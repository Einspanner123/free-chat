"""
KV Cache Experiment

Compare:
1. Cache Off — no KV cache, full attention recomputation per token
2. Cache On — standard KV cache
3. Prefix Cache — shared prefix reuse across requests

Metrics:
- Memory (KV cache size across sequence lengths)
- Latency (time per token)
- Throughput improvement

The experiment simulates KV cache behavior using analytical models
based on published measurements for 7B-scale transformers.
"""

# =============================================================================
# WARNING: SIMULATION / ESTIMATION ONLY
#
# The numbers produced by this script are analytical estimates or simulated
# results, NOT real hardware measurements. Do NOT cite these numbers in
# documentation, README, or resumes as measured performance.
#
# Real measured results live in benchmarks/long_context/results/ and were
# obtained by running actual models (Qwen2.5-0.5B, Qwen3-0.6B) on NVIDIA RTX
# A6000 hardware.
# =============================================================================

import math
from dataclasses import dataclass
from typing import List


@dataclass
class KVCacheResult:
    method: str
    seq_len: int
    kv_cache_gb: float
    latency_ms_per_token: float
    speedup_vs_no_cache: float
    throughput_tps: float


# Model configuration (Qwen2.5-7B equivalent)
NUM_LAYERS = 28
NUM_HEADS = 28
HEAD_DIM = 128
HIDDEN_SIZE = 3584


def kv_cache_size(seq_len: int, batch_size: int = 1, dtype_bytes: int = 2) -> float:
    """KV cache size in GB for one sequence."""
    bytes_per = 2 * NUM_LAYERS * batch_size * seq_len * NUM_HEADS * HEAD_DIM * dtype_bytes
    return bytes_per / (1024 ** 3)


def estimate_latency(seq_len: int, cache_type: str) -> float:
    """
    Estimate per-token latency in milliseconds.

    Without KV cache:
        Each step recomputes attention over ALL previous tokens.
        Complexity per step: O(L²) where L grows with each token.
        Total for N generated tokens: O(N³) - completely intractable at length.

    With KV cache:
        Each step only computes attention for the new token.
        Complexity per step: O(L) where L is total sequence length.
        Total for N generated tokens: O(L·N) ≈ O(L²).

    With prefix cache:
        First request pays full KV cache cost. Subsequent requests with
        shared prefix only compute the divergent suffix.
    """
    base_per_token = 50.0  # ms for 7B model forward pass
    prompt_encoding_ms = seq_len * 0.5  # ~0.5ms per token for prompt encoding

    if cache_type == "off":
        # KV cache off: O(L²) per step
        attention_recompute = (seq_len ** 2) * 0.002  # ms
        return prompt_encoding_ms + base_per_token + attention_recompute

    elif cache_type == "on":
        # KV cache on: O(L) per step
        attention_overhead = seq_len * 0.05  # ms
        return prompt_encoding_ms + base_per_token + attention_overhead

    elif cache_type == "prefix":
        # Prefix cache: first request pays full cost, subsequent share prefix
        # For a single request, same as "on". Benefit shows at batch level.
        return prompt_encoding_ms + base_per_token + seq_len * 0.05


def run_experiment(seq_lens: List[int] = None) -> List[KVCacheResult]:
    if seq_lens is None:
        seq_lens = [128, 512, 1024, 4096, 8192, 16384, 32768]

    results = []

    for seq_len in seq_lens:
        cache_gb = kv_cache_size(seq_len)

        # No cache
        lat_off = estimate_latency(seq_len, "off")
        results.append(KVCacheResult(
            method="No Cache",
            seq_len=seq_len,
            kv_cache_gb=0.0,
            latency_ms_per_token=round(lat_off, 1),
            speedup_vs_no_cache=1.0,
            throughput_tps=round(1000 / lat_off, 1),
        ))

        # With KV cache
        lat_on = estimate_latency(seq_len, "on")
        results.append(KVCacheResult(
            method="KV Cache",
            seq_len=seq_len,
            kv_cache_gb=round(cache_gb, 2),
            latency_ms_per_token=round(lat_on, 1),
            speedup_vs_no_cache=round(lat_off / lat_on, 1),
            throughput_tps=round(1000 / lat_on, 1),
        ))

    # Prefix cache benefit: simulated as batch of 4 requests sharing 50% prompt
    for seq_len in seq_lens:
        lat_on = estimate_latency(seq_len, "on")
        # Prefix cache: first request pays full prompt encoding,
        # subsequent 3 requests only pay 50% (shared prefix)
        shared_prompt_save = seq_len * 0.5 * 0.5 * 3 / 4  # avg saving over batch
        lat_prefix = lat_on - shared_prompt_save * 0.3
        results.append(KVCacheResult(
            method="Prefix Cache (4-req batch)",
            seq_len=seq_len,
            kv_cache_gb=round(kv_cache_size(seq_len), 2),
            latency_ms_per_token=round(max(lat_prefix, 1), 1),
            speedup_vs_no_cache=round(lat_on / max(lat_prefix, 1), 1),  # vs KV cache, not vs no-cache
            throughput_tps=round(1000 / max(lat_prefix, 1), 1),
        ))

    return results


def format_table(results: List[KVCacheResult]) -> str:
    lines = [
        "| Method | Seq Len | KV Cache (GB) | Latency (ms/t) | Speedup | Throughput (t/s) |",
        "|--------|---------|---------------|----------------|---------|-----------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.method:<25} | {r.seq_len:<7} | {r.kv_cache_gb:<12.2f} | "
            f"{r.latency_ms_per_token:<14} | {r.speedup_vs_no_cache:<7.1f}x | "
            f"{r.throughput_tps:<15.1f} |"
        )
    return "\n".join(lines)


def format_analysis(results: List[KVCacheResult]) -> str:
    no_cache = [r for r in results if r.method == "No Cache"]
    with_cache = [r for r in results if r.method == "KV Cache"]

    lines = ["## Analysis", ""]

    # Key finding: no cache is intractable at length
    for nc, wc in zip(no_cache, with_cache):
        if nc.seq_len == 4096:
            lines.append(f"At seq_len=4096: No cache {nc.latency_ms_per_token}ms/t vs KV cache {wc.latency_ms_per_token}ms/t")
            lines.append(f"KV cache provides {wc.speedup_vs_no_cache:.0f}x speedup at {wc.kv_cache_gb:.1f}GB memory cost.")
            lines.append("")
            break

    # Scaling
    lines.append("KV cache scales linearly with sequence length:")
    for r in with_cache:
        lines.append(f"  seq_len={r.seq_len:<6}: {r.kv_cache_gb:.1f}GB")
    lines.append("")
    lines.append(f"At 32K context, KV cache alone consumes ~{with_cache[-1].kv_cache_gb:.0f}GB.")
    lines.append("This is why eviction strategies (LRU, H2O, Sliding Window) become necessary")
    lines.append("for long-context serving on consumer GPUs.")

    return "\n".join(lines)


def save_results(results_list: List[KVCacheResult], out_dir: str = "results"):
    import json, os
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "experiment": "kv_cache",
        "results": [
            {"method": r.method, "seq_len": r.seq_len, "kv_cache_gb": r.kv_cache_gb,
             "latency_ms_per_token": r.latency_ms_per_token, "speedup": r.speedup_vs_no_cache,
             "throughput_tps": r.throughput_tps}
            for r in results_list
        ],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(format_table(results_list))
        f.write("\n\n")
        f.write(format_analysis(results_list))
    print(f"Results saved to {out_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None)
    args = parser.parse_args()

    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
    from _env_info import capture
    env = capture()
    gpu_name = env["gpu"]["devices"][0]["name"] if env["gpu"]["available"] else "None (reference mode)"
    print(f"GPU: {gpu_name}")

    print("=" * 80)
    print("KV Cache Experiment: Memory × Latency × Throughput")
    print("=" * 80)
    print()

    results = run_experiment(seq_lens=args.seq_lens)
    save_results(results, args.out)
    print(format_table(results))
    print()
    print(format_analysis(results))


if __name__ == "__main__":
    main()
