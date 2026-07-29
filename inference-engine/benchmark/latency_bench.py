"""
Latency benchmark: TTFT (Time to First Token) and TPOT (Time Per Output Token).

Usage:
  from latency_bench import run_latency_bench
  results = run_latency_bench(model_params_b=7)
"""

import math
from typing import List, Dict
from benchmark_runner import BenchmarkSuite


def estimate_ttft(
    model_params_b: float = 7,
    batch_size: int = 1,
    prompt_tokens: int = 512,
    use_kv_cache: bool = True,
) -> float:
    """
    Estimate TTFT in milliseconds.
    
    TTFT = prompt_encoding_time + first_token_generation_time
    
    - Prompt encoding is ~O(L) in sequence length, with significant fixed overhead
    - First token generation is same as any token (one forward pass)
    - Without KV cache, TTFT grows O(L²) — intractable for long sequences
    """
    base_per_token_ms = 1.5 * math.sqrt(model_params_b)  # rough: 7B → ~4ms/token for prompt
    # Prompt encoding: roughly O(L) per token, parallelized across batch
    # But memory bandwidth contention means larger batch doesn't reduce latency much
    batch_overhead = 1 + 0.1 * math.log2(max(batch_size, 1))
    prompt_time = prompt_tokens * base_per_token_ms * batch_overhead
    
    first_token_time = 10 * math.sqrt(model_params_b)  # generate first output token
    
    kv_cache_overhead = 0 if use_kv_cache else prompt_time * 2
    
    return prompt_time + first_token_time + kv_cache_overhead


def estimate_tpot(
    model_params_b: float = 7,
    batch_size: int = 1,
    kv_cache_efficiency: float = 1.0,
) -> float:
    """
    Estimate TPOT in milliseconds per token.
    
    Each token requires one forward pass through the model.
    In a batch, all tokens are processed in parallel, so per-token time
    decreases with batch size (up to GPU memory limits).
    
    kv_cache_efficiency: 1.0 = full KV cache, 0.5 = 50% eviction overhead
    """
    base = 50 * math.sqrt(model_params_b / 7)  # ~50ms for 7B
    batched = base / math.sqrt(max(batch_size, 1))
    return batched / kv_cache_efficiency


def run_latency_bench(
    model_params_b: float = 7,
    batch_sizes: List[int] = None,
    prompt_lengths: List[int] = None,
) -> List[Dict]:
    """Run latency benchmark across batch sizes and prompt lengths."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    if prompt_lengths is None:
        prompt_lengths = [128, 512, 2048, 8192]

    config = {"model_params_b": model_params_b}

    suite = BenchmarkSuite(name="Latency Benchmark", config=config)

    for bs in batch_sizes:
        for pl in prompt_lengths:
            ttft = estimate_ttft(model_params_b, bs, pl)
            tpot = estimate_tpot(model_params_b, bs)
            suite.add_result(f"TTFT_bs{bs}_len{pl}", round(ttft, 1), "ms",
                             metadata={"batch_size": bs, "prompt_len": pl})
            suite.add_result(f"TPOT_bs{bs}", round(tpot, 1), "ms",
                             metadata={"batch_size": bs})

    return suite
