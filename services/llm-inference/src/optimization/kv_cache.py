"""KV Cache optimization for LLM inference.

Implements:
1. KVCache: LRU-evicted key-value cache for token generation state
2. PrefixCache: Shared prefix matching for prompt caching
"""

import hashlib
from collections import OrderedDict
from typing import Any, Optional, Tuple, List


class KVCache:
    """LRU key-value cache for KV tensors.

    Stores intermediate KV cache states to avoid recomputation
    for shared prefixes across requests.
    """

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    def get(self, key: str) -> Optional[Any]:
        """Get cached value. Updates LRU order."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set cache value with LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def contains(self, key: str) -> bool:
        return key in self._cache

    def delete(self, key: str):
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


class PrefixCache:
    """Cache for shared prompt prefixes.

    Stores KV cache entries keyed by prompt prefix, allowing
    reuse when a new request shares a prefix with a cached one.
    """

    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._entries: OrderedDict = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._entries)

    def store(self, prefix: str, kv_data: Any):
        """Store KV data for a prefix."""
        key = self._hash(prefix)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = {"prefix": prefix, "kv": kv_data}
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def lookup(self, prefix: str) -> Optional[Any]:
        """Look up exact prefix match."""
        key = self._hash(prefix)
        entry = self._entries.get(key)
        if entry and entry["prefix"] == prefix:
            return entry["kv"]
        return None

    def match_prefix(self, text: str) -> Tuple[Optional[Any], float]:
        """Find the best matching prefix.

        Args:
            text: Input text to match.

        Returns:
            (cached_kv_data, match_score) where score is 0-1.
        """
        best_entry = None
        best_score = 0.0

        for entry in self._entries.values():
            prefix = entry["prefix"]
            score = self._compute_prefix_score(prefix, text)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score > 0.5:
            return best_entry["kv"], best_score
        return None, 0.0

    def clear(self):
        self._entries.clear()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _compute_prefix_score(prefix: str, text: str) -> float:
        """Compute how much of the prefix matches the start of text."""
        if not prefix or not text:
            return 0.0
        common = 0
        for a, b in zip(prefix, text):
            if a == b:
                common += 1
            else:
                break
        if common == 0:
            return 0.0
        return common / len(prefix)
