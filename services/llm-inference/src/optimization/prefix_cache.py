"""
Cross-request prefix KV cache (serve-time prefix reuse).

Production LLM serving repeats long shared prefixes (system prompts, RAG
context). Reusing the prefilled KV cache for a matched prefix skips re-prefill
of those tokens — the same idea as vLLM Prefix Caching / SGLang radiata, and
exactly what ``research/inference_optimization/run_kv_cache_speedup.py``
measured (1.68–2.97× prefill speedup).

This module is the *policy*: prefix matching + LRU eviction over prompt
token-id prefixes. The actual KV tensors live in whatever cache object the
engine uses (HF ``DynamicCache``); this manager only stores and retrieves them
by prompt prefix. The engine wires it into its generation path (gated behind
``prefix_cache_enabled``).

Thread safety: ``lookup`` / ``store`` take an internal lock so concurrent
requests on a shared engine cannot corrupt the LRU dict. (The engine still
serializes model forwards with its own lock; this guards the policy state.)

Correctness note: ``model.generate`` mutates a ``past_key_values`` it is given
in place, so a cached KV must be **cloned** before being reused. On
transformers 5.x, ``model.generate`` does NOT accept a pre-populated
``past_key_values`` with external ``input_ids`` (it drops/ignores the input and
decodes from the cache tail). The HF engine therefore implements prefix reuse
with a forward-based prefill + manual decode loop, resuming via
``model(suffix, past_key_values=clone(cached))`` (a forward, which IS supported)
rather than ``model.generate(suffix, past_key_values=...)``.
"""

from threading import RLock
from typing import Dict, List, Optional, Tuple


def longest_prefix_len(prompt_ids: List[int], cached_ids: Tuple[int, ...]) -> int:
    """Length of the longest prefix ``prompt_ids`` shares with ``cached_ids``."""
    n = min(len(prompt_ids), len(cached_ids))
    i = 0
    while i < n and prompt_ids[i] == cached_ids[i]:
        i += 1
    return i


class PrefixCache:
    """LRU cache of prefilled KV caches keyed by prompt token-id prefix.

    ``lookup`` returns the cached entry whose key is a *prefix* of the new
    prompt (not merely a shared prefix), choosing the longest such key. This
    guarantees the returned KV corresponds exactly to the matched prefix length
    and is safe to resume from.
    """

    def __init__(self, capacity: int = 8):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        # insertion-ordered dict; end = most-recently-used
        self._store: "Dict[Tuple[int, ...], object]" = {}
        self._lock = RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def lookup(self, prompt_ids: List[int]) -> Tuple[int, Optional[object]]:
        """Return ``(matched_len, cache)`` for the longest cached *prefix* of ``prompt_ids``.

        ``matched_len == 0`` means no usable prefix (caller prefills normally).
        A key is a valid match only if it is a prefix of ``prompt_ids``
        (``prompt_ids[:len(key)] == key``), so the returned KV is exactly the
        matched prefix. Ties broken by most-recently-used.
        """
        with self._lock:
            best_len = 0
            best_key: Optional[Tuple[int, ...]] = None
            for key in self._store:  # insertion order
                if len(key) <= len(prompt_ids) and prompt_ids[: len(key)] == list(key):
                    if len(key) > best_len:
                        best_len = len(key)
                        best_key = key
            if best_key is None or best_len == 0:
                return 0, None
            # touch LRU: move to end
            val = self._store.pop(best_key)
            self._store[best_key] = val
            return best_len, val

    def store(self, prompt_ids: List[int], cache: object) -> None:
        """Cache ``cache`` under the full prompt prefix; evict LRU if over capacity."""
        key = tuple(prompt_ids)
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = cache
            while len(self._store) > self._capacity:
                oldest = next(iter(self._store))
                del self._store[oldest]

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def keys(self) -> List[Tuple[int, ...]]:
        with self._lock:
            return list(self._store.keys())
