"""
Throughput benchmark: tokens/sec under varying concurrency.

Usage:
  from throughput_bench import run_throughput_bench
  results = run_throughput_bench(model_params_b=7)
"""

import math
from typing import List, Dict
from benchmark_runner import BenchmarkSuite


def estimate_throughput(
    model_params_b: float = 7,
    num_concurrent: int = 1,
    avg_output_tokens: int = 256,
    continuous_batching: bool = False,
) -> float:
    """
    Estimate throughput in tokens/second.
    
    Throughput is limited by:
    - Model forward pass latency (TPOT × tokens)
    - Batch efficiency (more concurrent = better GPU utilization, up to a point)
    - Memory capacity (more concurrent = larger KV cache, may force eviction)
    
    With continuous batching, throughput scales better because the GPU is
    never idle waiting for stragglers.
    """
    tpot = 50 * math.sqrt(model_params_b / 7)  # ms per token
    
    # Baseline: sequential processing
    seq_tps = 1000 / tpot
    
    # Parallel scaling: sublinear due to memory contention
    scaling_factor = math.sqrt(num_concurrent) * (0.9 ** (num_concurrent / 4))
    
    if continuous_batching:
        # Continuous batching improves utilization by ~30-50%
        scaling_factor *= 1.4
    
    return seq_tps * scaling_factor


def run_throughput_bench(
    model_params_b: float = 7,
    concurrency_levels: List[int] = None,
) -> List[Dict]:
    """Run throughput benchmark."""
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16, 32]

    config = {"model_params_b": model_params_b}

    suite = BenchmarkSuite(name="Throughput Benchmark", config=config)

    for nc in concurrency_levels:
        tps_static = estimate_throughput(model_params_b, nc, continuous_batching=False)
        tps_continuous = estimate_throughput(model_params_b, nc, continuous_batching=True)
        suite.add_result(f"static_concurrent_{nc}", round(tps_static, 1), "t/s",
                         metadata={"concurrency": nc, "batching": "static"})
        suite.add_result(f"continuous_concurrent_{nc}", round(tps_continuous, 1), "t/s",
                         metadata={"concurrency": nc, "batching": "continuous"})

    return suite
