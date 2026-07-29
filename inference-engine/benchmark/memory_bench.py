"""
Memory benchmark: model memory, KV cache scaling, GPU fit analysis.

Usage:
  from memory_bench import run_memory_bench
  results = run_memory_bench()
"""

from typing import List, Dict, Optional
from benchmark_runner import BenchmarkSuite

# Quantization memory ratios relative to FP16
_QUANT_RATIOS = {
    "fp16": 1.0,
    "int8": 0.55,
    "fp8": 0.50,
    "awq_int4": 0.32,
    "gptq_int4": 0.35,
}

# Model layer configuration for KV cache estimation
_MODEL_LAYERS = {
    0.5: 24,   # 0.5B models
    1.5: 28,   # 1.5B
    7: 28,     # 7B
    13: 40,    # 13B
    30: 60,    # 30B
    70: 80,    # 70B
}

_MODEL_HEADS = {
    0.5: 14,
    1.5: 12,
    7: 28,
    13: 40,
    30: 60,
    70: 64,
}

_HEAD_DIM = 128  # standard for most LLaMA-family models


def model_memory_fp16(params_b: float) -> float:
    """Model weights in FP16: each param uses 2 bytes."""
    return params_b * 2  # GB


def model_memory(params_b: float, method: str = "fp16") -> float:
    """Model weights with quantization."""
    ratio = _QUANT_RATIOS.get(method, 1.0)
    return model_memory_fp16(params_b) * ratio


def kv_cache_memory(
    params_b: float,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """
    KV cache memory in GB.
    
    Formula: Memory = 2 × layers × batch_size × seq_len × heads × head_dim × dtype_bytes
    """
    layers = _MODEL_LAYERS.get(int(params_b) if params_b >= 1 else 0.5, 28)
    heads = _MODEL_HEADS.get(int(params_b) if params_b >= 1 else 0.5, 28)
    
    bytes_total = 2 * layers * batch_size * seq_len * heads * _HEAD_DIM * dtype_bytes
    return bytes_total / (1024 ** 3)


def total_memory(params_b: float, seq_len: int, batch_size: int = 1, method: str = "fp16") -> float:
    """Total GPU memory: model weights + KV cache."""
    return model_memory(params_b, method) + kv_cache_memory(params_b, seq_len, batch_size)


def model_fits_on_gpu(params_b: float, method: str, gpu_vram_gb: float) -> bool:
    """Check if a model fits on a given GPU."""
    mem = total_memory(params_b, seq_len=4096, batch_size=1, method=method)
    return mem < gpu_vram_gb * 0.85  # 85% utilization threshold


def run_memory_bench(
    models: List[float] = None,
    seq_lens: List[int] = None,
    batch_sizes: List[int] = None,
) -> List[Dict]:
    """Run memory benchmark."""
    if models is None:
        models = [0.5, 7, 13, 70]
    if seq_lens is None:
        seq_lens = [1024, 4096, 16384]
    if batch_sizes is None:
        batch_sizes = [1, 4]

    suite = BenchmarkSuite(name="Memory Benchmark")

    for p in models:
        model_only = model_memory_fp16(p)
        suite.add_result(f"model_{p}B_fp16", round(model_only, 1), "GB",
                         metadata={"params_b": p, "method": "fp16"})

        for method in ["fp16", "awq_int4"]:
            quantized = model_memory(p, method)
            suite.add_result(f"model_{p}B_{method}", round(quantized, 1), "GB",
                             metadata={"params_b": p, "method": method})

        for sl in seq_lens:
            for bs in batch_sizes:
                kv = kv_cache_memory(p, sl, bs)
                suite.add_result(f"kv_{p}B_seq{sl}_bs{bs}", round(kv, 2), "GB",
                                 metadata={"params_b": p, "seq_len": sl, "batch_size": bs})

    return suite
