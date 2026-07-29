"""
Continuous Batching Scheduler

Static batching waits for all sequences in a batch to finish before starting
new ones. Continuous batching (aka iteration-level scheduling) adds/removes
sequences at each decoding step, maximizing GPU utilization.

Reference: 
  Orca: A Distributed Serving System for Transformer-Based Generative Models
  (Yu et al., SOSP 2022)

Key idea:
  Static:    [A B C] all start together, all finish together → GPU idle waiting
  Continuous: A B C → A B D → A E D → each step reconfigures the batch
"""

import time
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Callable
from collections import deque


class RequestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    id: str
    prompt: str
    max_tokens: int = 128
    arrival_time: float = 0.0
    status: RequestStatus = RequestStatus.PENDING
    generated_tokens: int = 0
    prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    max_total_tokens: int = 4096  # across all sequences in a batch
    scheduling_policy: str = "fcfs"  # fcfs, shortest-first, longest-first


class ContinuousBatchingScheduler:
    """
    Iteration-level scheduler for LLM inference.
    
    At each step, the scheduler decides which requests to include in the batch.
    Requests can enter/leave at every decoding iteration, not just at batch boundaries.
    
    When kv_cache_manager is provided, each step triggers:
      - allocate blocks for new requests (prompt tokens)
      - grow blocks for running requests (generated tokens)
      - free blocks for finished requests
    """

    def __init__(
        self,
        config: SchedulerConfig,
        model_max_seq_len: int = 2048,
        kv_cache_manager=None,
        blocks_per_token: float = 0.25,
    ):
        self.config = config
        self.model_max_seq_len = model_max_seq_len
        self.kv_cache_manager = kv_cache_manager
        self.blocks_per_token = blocks_per_token  # blocks allocated per generated token
        self.pending_queue: deque = deque()
        self.running_batch: List[Request] = []
        self.finished: List[Request] = []
        self.total_steps = 0

    def submit(self, request: Request):
        """Submit a new request to the scheduler."""
        request.arrival_time = time.time()
        request.status = RequestStatus.PENDING
        self.pending_queue.append(request)

    def submit_batch(self, requests: List[Request]):
        for r in requests:
            self.submit(r)

    def step(self) -> List[Request]:
        """
        Execute one scheduling step.
        
        1. Free blocks for finished requests.
        2. Allocate blocks for newly admitted requests.
        3. Grow KV cache blocks for running requests.
        4. Advance token generation.
        
        Returns:
            The current batch of active requests after this step.
        """
        self.total_steps += 1

        # 1. Remove finished requests → free KV cache blocks
        still_running = []
        for req in self.running_batch:
            if req.status == RequestStatus.FINISHED:
                if self.kv_cache_manager:
                    self.kv_cache_manager.free(req.id)
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running_batch = still_running

        # 2. Admit new requests → allocate KV cache blocks for their prompt
        while self.pending_queue and self._can_add():
            req = self.pending_queue.popleft()
            req.status = RequestStatus.RUNNING
            self.running_batch.append(req)
            if self.kv_cache_manager:
                # Allocate blocks proportional to prompt length
                blocks_for_prompt = max(1, int(req.prompt_tokens * self.blocks_per_token))
                self.kv_cache_manager.allocate(req.id, blocks_for_prompt)

        # 3. Grow KV cache for each running request (one token per step)
        if self.kv_cache_manager:
            for req in self.running_batch:
                self.kv_cache_manager.allocate(req.id, 1)

        # 4. Advance token generation
        for req in self.running_batch:
            req.generated_tokens += 1
            # Check token limit
            if req.generated_tokens >= req.max_tokens:
                req.status = RequestStatus.FINISHED

        return self.running_batch

    def run_until_complete(self, generate_fn: Optional[Callable] = None) -> List[Request]:
        """
        Run scheduling loop until all requests are finished.
        
        Args:
            generate_fn: Optional custom generation function.
                         If None, uses internal simulation.
        
        Returns:
            List of finished requests.
        """
        max_steps = 100000  # safety limit
        steps = 0
        while (self.pending_queue or self.running_batch) and steps < max_steps:
            steps += 1
            before = len(self.running_batch)
            self.step()
            after = len(self.running_batch)
            # If nothing running and nothing can be added, break
            if after == 0 and before == 0:
                break
        return self.finished

    def _can_add(self) -> bool:
        """Check if there's capacity for another request in the current batch."""
        if len(self.running_batch) >= self.config.max_batch_size:
            return False
        
        # Token budget check
        current_tokens = sum(r.total_tokens for r in self.running_batch)
        next_req = self.pending_queue[0]
        would_add = current_tokens + next_req.prompt_tokens
        if would_add > self.config.max_total_tokens:
            return False
        
        # KV cache block check
        if self.kv_cache_manager:
            needed = max(1, int(next_req.prompt_tokens * self.blocks_per_token))
            if self.kv_cache_manager.free_blocks < needed:
                return False
        
        return True

    def stats(self) -> dict:
        """Return scheduling statistics."""
        if not self.finished:
            return {}
        
        total_time = max(r.arrival_time for r in self.finished) - min(r.arrival_time for r in self.finished)
        total_tokens = sum(r.generated_tokens for r in self.finished)
        
        stats = {
            "total_requests": len(self.finished),
            "total_time_sec": round(total_time, 3),
            "total_generated_tokens": total_tokens,
            "avg_throughput_tps": round(total_tokens / total_time, 1) if total_time > 0 else 0,
            "avg_batch_size": round(
                sum(r.generated_tokens for r in self.finished) / max(self.total_steps, 1), 1
            ),
            "scheduling_steps": self.total_steps,
        }
        
        if self.kv_cache_manager:
            stats["kv_cache"] = self.kv_cache_manager.stats()
        
        return stats


# =============================================================================
# Experiment: Static vs Continuous Batching Comparison
# =============================================================================

@dataclass
class BatchBenchResult:
    method: str
    num_requests: int
    total_time_sec: float
    throughput_tps: float
    avg_latency_sec: float
    p99_latency_sec: float


def bench_static_batching(requests: List[Request], batch_size: int, ms_per_token: float = 50.0) -> BatchBenchResult:
    """
    Static batching: divide requests into fixed batches, process each batch sequentially.
    
    Batch latency = max(request max_tokens) × ms_per_token (longest request determines batch time).
    Short requests waste GPU cycles waiting for the longest one.
    """
    batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
    
    total_time_ms = 0
    latencies = []
    
    for batch in batches:
        max_tokens = max(r.max_tokens for r in batch)
        batch_latency_ms = max_tokens * ms_per_token
        total_time_ms += batch_latency_ms
        for r in batch:
            r.generated_tokens = r.max_tokens
            r.status = RequestStatus.FINISHED
            latencies.append(batch_latency_ms / 1000.0)
    
    total_time_sec = total_time_ms / 1000.0
    total_tokens = sum(r.generated_tokens for r in requests)
    
    latencies.sort()
    p99 = latencies[int(len(latencies) * 0.99)]
    
    return BatchBenchResult(
        method="static",
        num_requests=len(requests),
        total_time_sec=round(total_time_sec, 3),
        throughput_tps=round(total_tokens / total_time_sec, 1),
        avg_latency_sec=round(sum(latencies) / len(latencies), 3),
        p99_latency_sec=round(p99, 3),
    )


def bench_continuous_batching(requests: List[Request], max_batch_size: int, ms_per_token: float = 50.0) -> BatchBenchResult:
    """
    Continuous batching: iteration-level scheduling.
    
    Requests enter/leave at each decoding step. Short requests finish early;
    new requests can start without waiting for the full batch to finish.
    """
    config = SchedulerConfig(max_batch_size=max_batch_size)
    scheduler = ContinuousBatchingScheduler(config)
    scheduler.submit_batch(requests)
    
    finish_tokens_at = {}
    current_time_ms = 0
    
    while scheduler.pending_queue or scheduler.running_batch:
        current_time_ms += ms_per_token
        batch = scheduler.step()
        for r in scheduler.finished:
            if r.id not in finish_tokens_at:
                finish_tokens_at[r.id] = current_time_ms
    
    total_time_sec = current_time_ms / 1000.0
    total_tokens = sum(r.generated_tokens for r in requests)
    
    latency_values = [v / 1000.0 for v in finish_tokens_at.values()]
    latency_values.sort()
    p99 = latency_values[int(len(latency_values) * 0.99)] if latency_values else 0
    
    return BatchBenchResult(
        method="continuous",
        num_requests=len(requests),
        total_time_sec=round(total_time_sec, 3),
        throughput_tps=round(total_tokens / total_time_sec, 1),
        avg_latency_sec=round(sum(latency_values) / len(latency_values), 3) if latency_values else 0,
        p99_latency_sec=round(p99, 3),
    )


def run_benchmark(num_requests: int = 32, max_tokens_range=(32, 256)) -> List[BatchBenchResult]:
    """
    Run static vs continuous batching comparison.
    
    Creates requests with varying max_tokens (simulating real workload),
    then benchmarks both scheduling strategies.
    """
    import random
    random.seed(42)
    
    requests = []
    for i in range(num_requests):
        req = Request(
            id=f"req-{i}",
            prompt=f"Prompt {i}",
            max_tokens=random.randint(*max_tokens_range),
            prompt_tokens=random.randint(64, 512),
        )
        requests.append(req)
    
    static_results = bench_static_batching(requests, batch_size=8)
    continuous_results = bench_continuous_batching(requests, max_batch_size=8)
    
    return [static_results, continuous_results]


def format_benchmark_table(results: List[BatchBenchResult]) -> str:
    lines = [
        "| Method | Requests | Total Time (s) | Throughput (t/s) | Avg Latency (s) | P99 Latency (s) |",
        "|--------|----------|---------------|-----------------|-----------------|----------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.method} | {r.num_requests} | {r.total_time_sec} | "
            f"{r.throughput_tps} | {r.avg_latency_sec} | {r.p99_latency_sec} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import random
    random.seed(42)
    
    print("=" * 70)
    print("Static vs Continuous Batching: Throughput & Latency")
    print("=" * 70)
    print(f"Model: ~7B params, 50ms/token generation")
    print(f"Workload: 32 requests, varying max_tokens (32-256)")
    print(f"Batch size / max concurrent: 8")
    print()
    
    results = run_benchmark(num_requests=32)
    print(format_benchmark_table(results))
    print()
    
    r1, r2 = results
    speedup = r2.throughput_tps / r1.throughput_tps
    print(f"Continuous batching improves throughput by {speedup:.1f}× over static.")
    print(f"P99 latency reduced from {r1.p99_latency_sec:.2f}s to {r2.p99_latency_sec:.2f}s.")
    print()
    
    # Schedule analysis  
    print("=" * 70)
    print("Inside the Scheduler: Batch Composition Over Time")
    print("=" * 70)
    scheduler = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=8))
    for i in range(20):
        scheduler.submit(Request(id=f"r{i}", prompt=f"p{i}", max_tokens=random.randint(16, 64), prompt_tokens=random.randint(64, 256)))
    scheduler.run_until_complete()
    stats = scheduler.stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total generated tokens: {stats['total_generated_tokens']}")
    print(f"Avg batch size: {stats['avg_batch_size']}")
    print(f"Scheduling steps: {stats['scheduling_steps']}")
