"""StreamingLLM-style KV-cache eviction (attention sink + sliding window).

Service-layer home of the algorithm benchmarked by
``research/inference_optimization/run_kv_eviction_quality.py``. The goal is
MEMORY, not speed: keeping only ``[sink tokens | recent window]`` of K/V per
layer lets a fixed GPU hold a far longer context / larger batch (StreamingLLM,
Xiao et al. 2023).

``SinkWindowCache`` is a ``DynamicCache`` subclass that evicts the middle of
the KV sequence during single-token decode steps, preserving the first
``sink_size`` positions (attention sink) and the last ``window_size``
(recent context). It is injected into generation by passing an instance as
``past_key_values``.

Correctness invariants (verified against transformers 5.14.1)
-------------------------------------------------------------
1. ``get_seq_length()`` / ``get_query_offset()`` return the LOGICAL cumulative
   length (including evicted tokens), never the physical size — otherwise
   Qwen3 derives wrong ``position_ids`` and RoPE positions reset.
2. ``get_mask_sizes(q, layer)`` returns ``(_physical_after(q, layer), 0)``
   where ``_physical_after`` = stored + q, capped at ``sink+window`` for
   single-token (decode) steps, uncapped during prefill. This MUST equal the
   seq-len of the tensor ``update()`` returns.
3. ``update()`` returns the truncated ``[sink|window]`` tensors during decode
   (real memory + compute savings, not just a mask). During prefill (q_len>1)
   nothing is evicted — the full context must be attended to build the cache.
4. ``get_max_length()`` returns ``sink_size + window_size``.
5. ``crop()`` raises: after eviction the physical layout is ``[sink|window]``,
   not a prefix, so prefix-based ``crop`` (used by assisted decoding /
   speculative-decoding rollback) would silently corrupt the cache.

Limitations
-----------
- Single sequence only (no padding/packing): the 2D attention-mask slicing in
  ``masking_utils`` assumes cached positions are a contiguous prefix, which
  does not hold for ``[sink|window]``.
- Prefill phase peak memory is NOT saved (eviction happens on the first decode
  step); only decode steady-state memory is saved.
- Multi-token speculative/assisted verification steps (q_len>1) do not evict.
"""

import torch
from transformers import DynamicCache


class SinkWindowCache(DynamicCache):
    """A DynamicCache that keeps ``[sink_size | window_size]`` KV positions.

    Drop-in for ``DynamicCache``: pass an instance as ``past_key_values`` to
    ``model.generate(...)`` (do NOT also pass ``cache_implementation`` —
    transformers raises if both are given).
    """

    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 512,
        **kwargs,
    ):
        if sink_size < 0:
            raise ValueError(f"sink_size must be >= 0, got {sink_size}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self.sink_size = sink_size
        self.window_size = window_size
        self._cap = sink_size + window_size
        self._logical_seq_len = 0  # cumulative tokens ever processed (monotonic)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Physical / logical length helpers
    # ------------------------------------------------------------------

    def _physical_len(self, layer_idx: int) -> int:
        """Number of K/V positions currently stored for a layer."""
        if layer_idx >= len(self.layers) or not self.layers[layer_idx].is_initialized:
            return 0
        return self.layers[layer_idx].keys.shape[-2]

    def _physical_after(self, query_length: int, layer_idx: int) -> int:
        """Seq-len the returned K/V will have after an update of ``query_length``.

        Decode (1 token) is capped at ``sink + window``; prefill (multi-token)
        is not capped (nothing is evicted during prefill).
        """
        after = self._physical_len(layer_idx) + query_length
        if query_length == 1:
            after = min(after, self._cap)
        return after

    # ------------------------------------------------------------------
    # Cache contract overrides
    # ------------------------------------------------------------------

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Append K/V; on single-token decode steps evict to ``[sink|window]``.

        Returns the tensors the attention layer should use, whose seq-len
        matches ``get_mask_sizes`` for this forward.
        """
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        # decode step (q_len==1) AND over capacity -> evict the middle
        if key_states.shape[-2] == 1 and keys.shape[-2] > self._cap:
            keys = torch.cat(
                [keys[..., : self.sink_size, :], keys[..., -self.window_size :, :]],
                dim=-2,
            )
            values = torch.cat(
                [values[..., : self.sink_size, :], values[..., -self.window_size :, :]],
                dim=-2,
            )
            self.layers[layer_idx].keys = keys
            self.layers[layer_idx].values = values
        # count tokens once per forward (layer 0 is updated first)
        if layer_idx == 0:
            self._logical_seq_len += key_states.shape[-2]
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Logical cumulative length (including evicted tokens)."""
        return self._logical_seq_len

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Logical cumulative length — makes every stored position visible."""
        return self._logical_seq_len

    def get_mask_sizes(self, query_length: int, layer_idx: int):
        """(kv_length, kv_offset) matching the tensors ``update`` returns."""
        if layer_idx >= len(self.layers):
            return query_length, 0  # first prefill: mask over the query only
        return self._physical_after(query_length, layer_idx), 0

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return self._cap

    def crop(self, maximum_length):
        raise NotImplementedError(
            "SinkWindowCache cannot be cropped: its physical layout is "
            "[sink|window], not a prefix. Do not use with assisted decoding "
            "or speculative-decoding KV rollback."
        )

    # ------------------------------------------------------------------
    # Introspection (used by the benchmark)
    # ------------------------------------------------------------------

    @property
    def logical_length(self) -> int:
        return self._logical_seq_len

    def physical_length(self, layer_idx: int = 0) -> int:
        return self._physical_len(layer_idx)

    def kv_bytes(self) -> int:
        """Exact bytes currently held by the K/V tensors across all layers."""
        return kv_cache_bytes(self)


def kv_cache_bytes(cache) -> int:
    """Exact bytes held by a cache's K/V tensors (works for any DynamicCache)."""
    total = 0
    for layer in cache.layers:
        if layer.keys is not None:
            total += (layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
    return total


def evict_to_sink_window(cache, sink_size: int, window_size: int):
    """Truncate a plain ``DynamicCache`` in place to ``[sink|window]``.

    Utility for tests / scripts that drive the model manually. NOTE: this does
    NOT fix ``get_seq_length()`` (it keeps returning the physical size), so the
    model must be given explicit ``position_ids`` or RoPE positions reset. Prefer
    ``SinkWindowCache`` whenever position_ids are derived from the cache.
    """
    cap = sink_size + window_size
    for layer in cache.layers:
        if layer.keys is not None and layer.keys.shape[-2] > cap:
            layer.keys = torch.cat(
                [layer.keys[..., :sink_size, :], layer.keys[..., -window_size:, :]],
                dim=-2,
            )
            layer.values = torch.cat(
                [layer.values[..., :sink_size, :], layer.values[..., -window_size:, :]],
                dim=-2,
            )
    return cache
