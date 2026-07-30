"""
Tests: inject KVCacheManager into the inference engine.

The adapter wraps KVCacheManager with an interface compatible
with the engine's existing cache usage patterns.
"""

import os
import sys
import pytest

_mem = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory-manager")
if _mem not in sys.path:
    sys.path.insert(0, _mem)


class TestCacheAdapter:
    """Adapter that makes KVCacheManager look like the engine's cache interface."""

    def test_adapter_imports(self):
        from kv_cache_manager import KVCacheManager
        assert KVCacheManager is not None

    def test_adapter_wraps_basic_ops(self):
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter

        inner = KVCacheManager(total_blocks=32)
        adapter = EngineCacheAdapter(inner, blocks_per_token=0.25)
        adapter.initialize_sequence("seq_1", prompt_tokens=128)
        assert inner.used_blocks > 0

    def test_adapter_token_step(self):
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter

        inner = KVCacheManager(total_blocks=32)
        adapter = EngineCacheAdapter(inner, blocks_per_token=0.25)
        adapter.initialize_sequence("seq_1", prompt_tokens=64)
        before = inner.used_blocks
        adapter.on_token_generated("seq_1")
        after = inner.used_blocks
        assert after >= before

    def test_adapter_finalize(self):
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter

        inner = KVCacheManager(total_blocks=32)
        adapter = EngineCacheAdapter(inner)
        adapter.initialize_sequence("seq_1", prompt_tokens=64)
        adapter.finalize_sequence("seq_1")
        assert inner.used_blocks == 0

    def test_adapter_prefix_hit(self):
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter

        inner = KVCacheManager(total_blocks=32)
        adapter = EngineCacheAdapter(inner)
        adapter.initialize_sequence("seq_1", prompt_tokens=64)
        blocks = inner.allocate("tmp", num_blocks=2)
        inner.store_prefix("The quick brown fox", blocks)
        inner.free("tmp")

        hit = adapter.lookup_prefix("The quick brown fox")
        assert hit is not None
        miss = adapter.lookup_prefix("nonexistent prompt")
        assert miss is None

    def test_adapter_stats(self):
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter

        inner = KVCacheManager(total_blocks=16)
        adapter = EngineCacheAdapter(inner)
        adapter.initialize_sequence("s1", prompt_tokens=64)
        adapter.initialize_sequence("s2", prompt_tokens=64)
        stats = adapter.stats()
        assert stats["active_sequences"] == 2
        assert stats["used_blocks"] > 0

    def test_adapter_zero_prompt(self):
        """initialize_sequence with 0 prompt tokens."""
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter
        inner = KVCacheManager(total_blocks=16)
        adapter = EngineCacheAdapter(inner)
        adapter.initialize_sequence("s1", prompt_tokens=0)
        # Should allocate at least 1 block
        assert inner.used_blocks >= 1
        adapter.finalize_sequence("s1")

    def test_adapter_empty_stats(self):
        """Stats with no active sequences."""
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter
        inner = KVCacheManager(total_blocks=16)
        adapter = EngineCacheAdapter(inner)
        stats = adapter.stats()
        assert stats["active_sequences"] == 0
        assert stats["used_blocks"] == 0

    def test_adapter_prefix_miss(self):
        """Lookup for nonexistent prefix returns None."""
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter
        inner = KVCacheManager(total_blocks=16)
        adapter = EngineCacheAdapter(inner)
        assert adapter.lookup_prefix("nonexistent") is None

    def test_adapter_clear_sequences(self):
        """Clear all active sequences."""
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter
        inner = KVCacheManager(total_blocks=16)
        adapter = EngineCacheAdapter(inner)
        adapter.initialize_sequence("a", prompt_tokens=32)
        adapter.initialize_sequence("b", prompt_tokens=32)
        adapter.initialize_sequence("c", prompt_tokens=32)
        assert adapter.stats()["active_sequences"] == 3
        adapter.clear()
        assert adapter.stats()["active_sequences"] == 0
        assert inner.used_blocks == 0

    def test_adapter_pool_exhaustion(self):
        """Requesting more blocks than available should handle gracefully."""
        from kv_cache_manager import KVCacheManager
        from engine_cache_adapter import EngineCacheAdapter
        inner = KVCacheManager(total_blocks=4)
        adapter = EngineCacheAdapter(inner, blocks_per_token=0.5)
        adapter.initialize_sequence("big", prompt_tokens=100)  # requests ~50 blocks
        # Only 4 blocks available
        assert inner.used_blocks <= 4
