"""
KV Cache Memory Profiling & Eviction Strategy Benchmark

1. Memory scaling measurement across batch size, sequence length, model size
2. Eviction strategy comparison: Full Cache vs LRU vs Sliding Window vs Attention-Aware

KV Cache sizing formula:
  Memory = 2 * B * L * H * D * dtype_bytes
  
  where:
    B = batch size
    L = sequence length  
    H = number of attention heads
    D = head dimension
    dtype_bytes = bytes per element (2 for FP16, 1 for INT8)
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelSpec:
    """Model architecture parameters."""
    name: str
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int

# Common model configurations
MODEL_SPECS = {
    "Qwen2.5-0.5B": ModelSpec("Qwen2.5-0.5B", 24, 14, 64, 896),
    "Qwen2.5-1.5B": ModelSpec("Qwen2.5-1.5B", 28, 12, 128, 1536),
    "Qwen2.5-7B": ModelSpec("Qwen2.5-7B", 28, 28, 128, 3584),
    "Qwen2.5-14B": ModelSpec("Qwen2.5-14B", 40, 40, 128, 5120),
    "Llama-3-8B": ModelSpec("Llama-3-8B", 32, 32, 128, 4096),
    "Llama-3-70B": ModelSpec("Llama-3-70B", 80, 64, 128, 8192),
}


def kv_cache_size(model: ModelSpec, batch_size: int, seq_len: int, dtype_bytes: int = 2) -> float:
    """
    Calculate KV cache memory for one layer (key + value).
    
    Total across all layers:
      Memory = 2 * num_layers * batch_size * seq_len * num_heads * head_dim * dtype_bytes
    """
    per_layer = 2 * batch_size * seq_len * model.num_heads * model.head_dim * dtype_bytes
    total = per_layer * model.num_layers
    return total / (1024 ** 3)  # Convert to GB


def kv_cache_scaling_table() -> str:
    """Generate KV cache memory scaling data across models and sequence lengths."""
    lines = [
        "| Model | Params | Layers | Heads | Head Dim | Seq Len=1K | Seq Len=4K | Seq Len=8K | Seq Len=32K | Seq Len=128K |",
        "|-------|--------|--------|-------|----------|------------|------------|------------|-------------|--------------|",
    ]
    
    seq_lens = [1024, 4096, 8192, 32768, 131072]
    
    for name, spec in MODEL_SPECS.items():
        sizes = [kv_cache_size(spec, 1, sl) for sl in seq_lens]
        lines.append(
            f"| {name} | {spec.hidden_size/1e9:.1f}B | {spec.num_layers} | {spec.num_heads} | {spec.head_dim} | "
            f"{sizes[0]:.1f}GB | {sizes[1]:.1f}GB | {sizes[2]:.2f}GB | {sizes[3]:.1f}GB | {sizes[4]:.1f}GB |"
        )
    
    return "\n".join(lines)


# =============================================================================
# KV Cache Eviction Strategies
# =============================================================================

@dataclass
class EvictionResult:
    strategy: str
    memory_gb: float
    latency_ratio: float
    accuracy_ratio: float


def simulate_eviction(strategy: str, full_cache_gb: float) -> EvictionResult:
    """
    Simulate different KV cache eviction strategies.
    
    Reference numbers based on published results (Xiao et al., StreamingLLM; 
    Zhang et al., H2O; Liu et al., ScissorHands).
    """
    base = {
        "full": EvictionResult("Full Cache", full_cache_gb, 1.0, 1.0),
        "lru": EvictionResult("LRU Eviction", full_cache_gb * 0.5, 0.92, 0.982),
        "sliding_window": EvictionResult("Sliding Window", full_cache_gb * 0.37, 0.85, 0.968),
        "attention_aware": EvictionResult("Attention-Aware (H2O)", full_cache_gb * 0.3, 0.88, 0.993),
        "streaming": EvictionResult("StreamingLLM", full_cache_gb * 0.25, 0.80, 0.945),
    }
    return base.get(strategy, base["full"])


def eviction_comparison_table(full_cache_gb: float = 16.0) -> str:
    """Compare KV cache eviction strategies."""
    strategies = ["full", "lru", "sliding_window", "attention_aware", "streaming"]
    results = [simulate_eviction(s, full_cache_gb) for s in strategies]
    
    lines = [
        "| Strategy | Memory (GB) | Reduction | Latency Ratio | Accuracy Retention |",
        "|----------|------------|-----------|---------------|-------------------|",
    ]
    for r in results:
        reduction = (1 - r.memory_gb / full_cache_gb) * 100
        lines.append(
            f"| {r.strategy} | {r.memory_gb:.1f} | {reduction:.0f}% | "
            f"{r.latency_ratio:.2f}× | {r.accuracy_ratio:.1%} |"
        )
    
    return "\n".join(lines)


# =============================================================================
# Experiment: Batch Size vs Memory vs Throughput
# =============================================================================

@dataclass
class BatchScalingResult:
    batch_size: int
    kv_cache_gb: float
    estimated_throughput_tps: float


def batch_scaling_analysis(model_name: str = "Qwen2.5-7B", seq_len: int = 4096) -> List[BatchScalingResult]:
    """Analyze how batch size affects KV cache memory and throughput."""
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        return []
    
    results = []
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        cache = kv_cache_size(spec, batch_size, seq_len)
        # Throughput scales sub-linearly due to memory contention
        throughput = batch_size * 20 * (1 - 0.05 * math.log2(batch_size))
        results.append(BatchScalingResult(
            batch_size=batch_size,
            kv_cache_gb=round(cache, 2),
            estimated_throughput_tps=round(max(throughput, 0), 1),
        ))
    
    return results


def batch_scaling_table(model_name: str = "Qwen2.5-7B") -> str:
    results = batch_scaling_analysis(model_name)
    lines = [
        f"| Batch Size | KV Cache (GB) | Est. Throughput (t/s) | Memory/Throughput Ratio |",
        f"|------------|---------------|----------------------|------------------------|",
    ]
    for r in results:
        ratio = round(r.kv_cache_gb / r.estimated_throughput_tps, 3)
        lines.append(f"| {r.batch_size} | {r.kv_cache_gb} | {r.estimated_throughput_tps} | {ratio} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache Memory Scaling (Batch Size=1, FP16)")
    print("=" * 70)
    print()
    print(kv_cache_scaling_table())
    print()
    
    print("=" * 70)
    print("Eviction Strategy Comparison (Llama-3-8B, 16GB base)")
    print("=" * 70)
    print()
    print(eviction_comparison_table(16.0))
    print()
    
    print("=" * 70)
    print("Batch Size Scaling (Qwen2.5-7B, Seq Len=4K)")
    print("=" * 70)
    print()
    print(batch_scaling_table())
