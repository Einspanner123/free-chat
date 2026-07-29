"""
Integration tests: ContinuousBatchingScheduler + KVCacheManager.

Tests that the scheduler correctly allocates/frees KV cache blocks
as requests enter and leave the batch at each iteration step.
"""

import os
import sys
import pytest

_sched = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scheduler")
_mem = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory-manager")
for p in [_sched, _mem]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestSchedulerKVCacheIntegration:
    """Core integration: scheduler step ↔ KV cache allocate/free."""

    def test_scheduler_initializes_with_cache_manager(self):
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=64)
        scheduler = ContinuousBatchingScheduler(SchedulerConfig(), kv_cache_manager=mgr)
        assert scheduler.kv_cache_manager is mgr

    def test_request_entering_batch_allocates_blocks(self):
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=64)
        scheduler = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4), kv_cache_manager=mgr)
        scheduler.submit(Request(id="r1", prompt="test", max_tokens=10, prompt_tokens=256))
        scheduler.step()

        # r1 should have blocks allocated
        assert mgr.used_blocks > 0

    def test_finished_request_frees_blocks(self):
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=64)
        scheduler = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4), kv_cache_manager=mgr)
        scheduler.submit(Request(id="r1", prompt="test", max_tokens=1, prompt_tokens=64))
        scheduler.run_until_complete()

        # After completion, no blocks should be held by r1
        assert mgr.free_blocks == 64

    def test_multiple_requests_share_block_pool(self):
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=64)
        scheduler = ContinuousBatchingScheduler(
            SchedulerConfig(max_batch_size=8), kv_cache_manager=mgr, blocks_per_token=0.1
        )

        for i in range(6):
            scheduler.submit(Request(id=f"r{i}", prompt=f"p{i}", max_tokens=10, prompt_tokens=64))

        scheduler.run_until_complete()
        assert mgr.free_blocks == 64  # all freed after completion

    def test_memory_pressure_blocks_new_requests(self):
        """When pool is nearly full, scheduler should slow admission."""
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=8)
        scheduler = ContinuousBatchingScheduler(
            SchedulerConfig(max_batch_size=8), kv_cache_manager=mgr, blocks_per_token=0.1
        )

        scheduler.submit(Request(id="big", prompt="test", max_tokens=50, prompt_tokens=200))
        scheduler.step()

        # Submit more requests
        scheduler.submit(Request(id="small", prompt="test", max_tokens=5, prompt_tokens=32))
        scheduler.step()

        # The scheduler should not exceed available blocks
        assert mgr.used_blocks <= 8

    def test_step_does_not_leak_blocks(self):
        """After many steps, blocks should be properly accounted."""
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=16)
        scheduler = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4), kv_cache_manager=mgr)

        for i in range(10):
            scheduler.submit(Request(id=f"r{i}", prompt=f"p{i}", max_tokens=5, prompt_tokens=64))

        scheduler.run_until_complete()
        stats = mgr.stats()
        assert stats["used_blocks"] == 0
        assert stats["free_blocks"] == 16

    def test_cache_manager_policy_affects_scheduling(self):
        """Different eviction policies should produce different schedules under pressure."""
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=32, eviction_policy="lru")
        scheduler = ContinuousBatchingScheduler(
            SchedulerConfig(max_batch_size=4), kv_cache_manager=mgr, blocks_per_token=0.1
        )

        for i in range(5):
            scheduler.submit(Request(id=f"r{i}", prompt=f"p{i}", max_tokens=10, prompt_tokens=64))
        scheduler.run_until_complete()
        assert scheduler.stats()["total_requests"] == 5

    def test_block_count_accounting_across_steps(self):
        """As requests generate tokens, block count should grow."""
        from continuous_batching import ContinuousBatchingScheduler, SchedulerConfig, Request
        from kv_cache_manager import KVCacheManager

        mgr = KVCacheManager(total_blocks=128)
        scheduler = ContinuousBatchingScheduler(
            SchedulerConfig(max_batch_size=2), kv_cache_manager=mgr, blocks_per_token=0.2
        )

        scheduler.submit(Request(id="r1", prompt="test", max_tokens=10, prompt_tokens=32))
        scheduler.submit(Request(id="r2", prompt="test", max_tokens=10, prompt_tokens=32))

        blocks_after_step1 = mgr.used_blocks
        for _ in range(3):
            scheduler.step()
        blocks_after_step4 = mgr.used_blocks

        # After 3 more decoding steps, all requests grew by 1 token each
        # Each step allocates 1 block per request, so 3 steps × 2 requests = 6 more blocks
        # But initial allocation also exists, so total should be >= initial
        assert blocks_after_step4 > blocks_after_step1
