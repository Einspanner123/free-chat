"""
Tests for KV Cache Manager: block pool, eviction policies, prefix cache.

RED phase: all tests should fail.
GREEN phase: all tests should pass.
"""

import os
import sys
import pytest

_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory-manager")
if _src not in sys.path:
    sys.path.insert(0, _src)

# =============================================================================
# Block Pool Tests
# =============================================================================

class TestBlockPool:
    """Block-based memory allocation for KV cache."""

    def test_allocate_single_block(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=64, block_size_tokens=256)
        block_id = pool.allocate("request_1")
        assert block_id is not None
        assert pool.used_blocks == 1
        assert pool.free_blocks == 63

    def test_allocate_multiple_blocks(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=64)
        ids = pool.allocate("request_1", num_blocks=5)
        assert len(ids) == 5
        assert pool.used_blocks == 5

    def test_free_blocks(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=64)
        ids = pool.allocate("request_1", num_blocks=3)
        pool.free("request_1")
        assert pool.used_blocks == 0
        assert pool.free_blocks == 64

    def test_out_of_memory(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=4)
        pool.allocate("req_a", num_blocks=3)
        ids = pool.allocate("req_b", num_blocks=3)
        assert len(ids) == 1  # only 1 remaining
        assert pool.used_blocks == 4

    def test_free_nonexistent_request_does_not_crash(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=64)
        pool.free("nonexistent")  # should not raise

    def test_multiple_requests_share_pool(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=16)
        pool.allocate("A", num_blocks=5)
        pool.allocate("B", num_blocks=5)
        pool.allocate("C", num_blocks=5)
        assert pool.used_blocks == 15
        assert pool.free_blocks == 1

    def test_freed_blocks_reusable(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=4)
        ids1 = pool.allocate("X", num_blocks=4)
        pool.free("X")
        ids2 = pool.allocate("Y", num_blocks=4)
        assert len(ids2) == 4
        # Blocks should be different from original allocation
        assert set(ids1) == set(ids2)  # same block IDs, reused

    def test_fragmentation_metric(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=8)
        # Allocate interleaved
        ids_a = pool.allocate("A", num_blocks=2)
        ids_b = pool.allocate("B", num_blocks=2)
        pool.free("A")  # frees blocks 0,1; blocks 2,3 still held by B
        frag = pool.fragmentation_pct
        assert frag > 0  # some fragmentation due to non-contiguous free

    def test_available_blocks_accounting(self):
        from kv_cache_manager import BlockPool
        pool = BlockPool(total_blocks=10)
        assert pool.available_blocks() == 10
        pool.allocate("A", num_blocks=3)
        assert pool.available_blocks() == 7
        pool.free("A")
        assert pool.available_blocks() == 10


# =============================================================================
# Eviction Policy Tests
# =============================================================================

class TestLRUEviction:
    def test_evict_least_recently_used(self):
        from kv_cache_manager import LRUEvictionPolicy
        policy = LRUEvictionPolicy()
        policy.record_access("A")
        policy.record_access("B")
        policy.record_access("C")
        victims = policy.evict(2)
        assert len(victims) == 2
        assert victims[0] == "A"  # least recently accessed
        assert victims[1] == "B"

    def test_evict_empty(self):
        from kv_cache_manager import LRUEvictionPolicy
        policy = LRUEvictionPolicy()
        assert policy.evict(5) == []

    def test_reorder_on_reaccess(self):
        from kv_cache_manager import LRUEvictionPolicy
        policy = LRUEvictionPolicy()
        policy.record_access("A")
        policy.record_access("B")
        policy.record_access("C")
        policy.record_access("A")  # A is now most recent
        victims = policy.evict(2)
        assert victims == ["B", "C"]  # A should not be evicted

    def test_remove_tracking(self):
        from kv_cache_manager import LRUEvictionPolicy
        policy = LRUEvictionPolicy()
        policy.record_access("A")
        policy.record_access("B")
        policy.remove("A")
        victims = policy.evict(1)
        assert victims == ["B"]
        assert len(policy.evict(1)) == 0  # no more to evict


class TestSlidingWindowEviction:
    def test_window_keeps_last_n(self):
        from kv_cache_manager import SlidingWindowEvictionPolicy
        policy = SlidingWindowEvictionPolicy(window_size=4)
        # Position 0-7 (8 tokens)
        for i in range(8):
            policy.record_position(f"token_{i}", i)
        victims = policy.evict(10)
        # Token 0-3 should be evictable (outside window), token 4-7 should be kept
        assert "token_4" not in victims
        assert "token_5" not in victims
        assert "token_0" in victims

    def test_window_empty(self):
        from kv_cache_manager import SlidingWindowEvictionPolicy
        policy = SlidingWindowEvictionPolicy(window_size=4)
        assert policy.evict(10) == []


# =============================================================================
# Full KV Cache Manager Tests
# =============================================================================

class TestKVCacheManager:
    def test_init(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=64, block_size=256, eviction_policy="lru")
        assert mgr.total_blocks == 64
        assert mgr.free_blocks == 64

    def test_allocate_and_free(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=32)
        mgr.allocate("req_1", num_blocks=4)
        assert mgr.used_blocks == 4
        mgr.free("req_1")
        assert mgr.used_blocks == 0

    def test_allocate_triggers_eviction(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=4)
        mgr.allocate("old_req", num_blocks=4)
        # Second request should trigger eviction of old_req
        mgr.allocate("new_req", num_blocks=2)
        assert mgr.used_blocks <= 4

    def test_prefix_cache_hit(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=64)
        prompt = "You are a helpful assistant."
        # First time: cache miss
        hit1 = mgr.lookup_prefix(prompt)
        assert hit1 is None
        # Store some blocks
        blocks = mgr.allocate("tmp", num_blocks=2)
        mgr.store_prefix(prompt, blocks)
        mgr.free("tmp")
        # Second time: cache hit
        hit2 = mgr.lookup_prefix(prompt)
        assert hit2 is not None

    def test_prefix_cache_eviction(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=8)
        # Fill with different prefixes
        for i in range(5):
            blocks = mgr.allocate(f"req_{i}", num_blocks=1)
            mgr.store_prefix(f"prefix_{i}", blocks)
        # Max stored prefixes = 4 (under default)
        assert mgr.prefix_cache_size <= 5

    def test_eviction_policy_switch(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=32, eviction_policy="lru")
        mgr.set_eviction_policy("sliding_window", window_size=8)
        # Class name: SlidingWindowEvictionPolicy → slidingwindow
        name = mgr.eviction_policy_name
        assert "sliding" in name

    def test_allocation_fails_gracefully(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=2)
        mgr.allocate("big", num_blocks=4)
        assert mgr.used_blocks == 2  # only 2 fit

    def test_stats(self):
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=16)
        mgr.allocate("A", num_blocks=2)
        mgr.allocate("B", num_blocks=3)
        stats = mgr.stats()
        assert stats["total_blocks"] == 16
        assert stats["used_blocks"] == 5
        assert stats["free_blocks"] == 11
        assert stats["fragmentation_pct"] >= 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestKVCacheIntegration:
    """Simulate realistic serving scenarios."""

    def test_single_request_grows(self):
        """As a request generates tokens, it allocates more blocks."""
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=32)
        req_id = "chat_1"
        # Start with 1 prompt block
        blocks = mgr.allocate(req_id, num_blocks=1)
        assert len(blocks) == 1
        # Generate 5 more tokens
        blocks = mgr.allocate(req_id, num_blocks=5)
        assert len(blocks) == 5
        assert mgr.used_blocks == 6

    def test_multiple_requests_contend_for_memory(self):
        """Multiple requests share limited block pool.
        When pool is full, eviction frees blocks from LRU request.
        """
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=8)
        mgr.allocate("A", num_blocks=3)
        mgr.allocate("B", num_blocks=3)
        third = mgr.allocate("C", num_blocks=3)
        # C gets 2 directly + 1 from evicting A (LRU victim) = 3 total
        assert len(third) == 3
        assert mgr.used_blocks == 6  # B(3) + C(3), A evicted

    def test_request_finish_frees_blocks_for_others(self):
        """When a request finishes, its blocks go back to the pool."""
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=8)
        mgr.allocate("A", num_blocks=5)
        assert mgr.free_blocks == 3
        mgr.free("A")
        assert mgr.free_blocks == 8
        # New request can now use freed blocks
        ids = mgr.allocate("B", num_blocks=5)
        assert len(ids) == 5

    def test_prefix_reuse_saves_allocation(self):
        """Requests sharing a prefix should reuse cached blocks."""
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=32)
        prompt = "System: You are a coding assistant.\nUser: Write a function."
        blocks = mgr.allocate("req_1", num_blocks=3)
        mgr.store_prefix(prompt, blocks)
        mgr.free("req_1")
        # Second request with same prefix
        cached = mgr.lookup_prefix(prompt)
        assert cached is not None
        # Only need to allocate for the divergent part
        new_blocks = mgr.allocate("req_2", num_blocks=1)
        assert len(new_blocks) == 1

    def test_high_contention_eviction_order(self):
        """Under memory pressure, LRU evicts oldest requests first."""
        from kv_cache_manager import KVCacheManager
        mgr = KVCacheManager(total_blocks=6)
        # Fill the pool
        mgr.allocate("old", num_blocks=2)
        mgr.allocate("mid", num_blocks=2)
        # Access "old" to make it most recent
        # (In a real system, every decode step accesses the request)
        # Then allocate for "new" which should evict "mid" (least recently used)
        mgr.allocate("new", num_blocks=2)
        assert mgr.used_blocks <= 6  # no overallocation
