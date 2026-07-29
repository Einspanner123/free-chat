"""
EngineCacheAdapter: bridges KVCacheManager to inference engine interface.

The inference engine (HF/vLLM) needs:
  - initialize_sequence(seq_id, prompt_tokens): allocate KV blocks for prompt
  - on_token_generated(seq_id): allocate more blocks as sequence grows
  - finalize_sequence(seq_id): free all blocks when sequence ends
  - lookup_prefix / store_prefix: cache reuse across requests
  
This adapter wraps KVCacheManager with this engine-friendly API.
"""

from typing import List, Optional


class EngineCacheAdapter:
    """Adapts KVCacheManager for use by the inference engine."""

    def __init__(self, kv_cache_manager, blocks_per_token: float = 0.25):
        self._mgr = kv_cache_manager
        self.blocks_per_token = blocks_per_token
        self._sequences: set = set()

    def initialize_sequence(self, seq_id: str, prompt_tokens: int):
        """Allocate KV blocks for a new sequence's prompt.

        Args:
            seq_id: Unique sequence identifier.
            prompt_tokens: Number of tokens in the prompt.
        """
        self._sequences.add(seq_id)
        blocks_needed = max(1, int(prompt_tokens * self.blocks_per_token))
        self._mgr.allocate(seq_id, blocks_needed)

    def on_token_generated(self, seq_id: str):
        """Grow KV cache by one token's worth of blocks.

        Called after each generated token.
        """
        blocks_needed = max(1, int(self.blocks_per_token))
        self._mgr.allocate(seq_id, blocks_needed)

    def finalize_sequence(self, seq_id: str):
        """Free all KV blocks for a finished sequence."""
        self._sequences.discard(seq_id)
        self._mgr.free(seq_id)

    def lookup_prefix(self, prefix: str) -> Optional[List[int]]:
        """Look up cached KV blocks for a prompt prefix."""
        return self._mgr.lookup_prefix(prefix)

    def store_prefix(self, prefix: str, block_ids: List[int]):
        """Cache KV blocks for a prompt prefix."""
        self._mgr.store_prefix(prefix, block_ids)

    def stats(self) -> dict:
        """Return cache diagnostics."""
        s = self._mgr.stats()
        s["active_sequences"] = len(self._sequences)
        return s

    def clear(self):
        """Clear all cached data."""
        # Free all active sequences
        for seq_id in list(self._sequences):
            self.finalize_sequence(seq_id)
        self._mgr._prefix_cache.clear()
