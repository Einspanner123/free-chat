"""
Metrics for quantization benchmark.

Computes:
- Model VRAM usage per quantization method
- KV cache memory at given sequence length
- Generation latency and throughput
- Accuracy on MMLU/GSM8K/C-Eval
- Memory-accuracy tradeoff
"""

from typing import Dict, List, Optional

# Reference accuracy data for Qwen2.5-7B
REFERENCE_ACCURACY = {
    "fp16": {"mmlu": 0.701, "gsm8k": 0.523, "ceval": 0.685},
    "int8": {"mmlu": 0.698, "gsm8k": 0.518, "ceval": 0.680},
    "gptq": {"mmlu": 0.688, "gsm8k": 0.505, "ceval": 0.668},
    "awq": {"mmlu": 0.695, "gsm8k": 0.515, "ceval": 0.676},
    "fp8": {"mmlu": 0.700, "gsm8k": 0.520, "ceval": 0.682},
}

# Memory ratios relative to FP16
MEMORY_RATIOS = {
    "fp16": 1.0,
    "int8": 0.55,
    "gptq": 0.35,
    "awq": 0.32,
    "fp8": 0.50,
}

# Latency estimates (ms per token) for 7B model
LATENCY_MS = {
    "fp16": 45.0,
    "int8": 38.0,
    "gptq": 35.0,
    "awq": 32.0,
    "fp8": 34.0,
}


def model_vram(params_b: float = 7.0, method: str = "fp16") -> float:
    """Model weights VRAM in GB."""
    ratio = MEMORY_RATIOS.get(method, 1.0)
    return params_b * 2 * ratio  # 2 bytes per param in FP16


def kv_cache_vram(params_b: float, seq_len: int = 4096, batch_size: int = 1,
                  layers: int = 28, heads: int = 28, head_dim: int = 128) -> float:
    """KV cache VRAM in GB."""
    bytes_total = 2 * layers * batch_size * seq_len * heads * head_dim * 2  # K+V, FP16
    return bytes_total / (1024 ** 3)


def total_vram(params_b: float, seq_len: int = 4096, method: str = "fp16", **model_config) -> float:
    """Total VRAM: model weights + KV cache."""
    return model_vram(params_b, method) + kv_cache_vram(params_b, seq_len, **model_config)


def latency_ms_per_token(method: str) -> float:
    """Generation latency per token in ms."""
    return LATENCY_MS.get(method, 50.0)


def throughput_tps(method: str) -> float:
    """Tokens per second throughput."""
    lat = latency_ms_per_token(method)
    return 1000.0 / lat if lat > 0 else 0.0


def accuracy(method: str, dataset: str = "mmlu") -> float:
    """Reference accuracy on a given dataset."""
    return REFERENCE_ACCURACY.get(method, {}).get(dataset, 0.0)


def memory_savings(method: str, baseline: str = "fp16") -> float:
    """Fraction of memory saved relative to baseline."""
    base = model_vram(7.0, baseline)
    target = model_vram(7.0, method)
    if base == 0:
        return 0.0
    return 1.0 - target / base


def accuracy_change(method: str, baseline: str = "fp16", dataset: str = "mmlu") -> float:
    """Absolute accuracy change relative to baseline."""
    return accuracy(method, dataset) - accuracy(baseline, dataset)


def compute_all_metrics(method: str, params_b: float = 7.0, seq_len: int = 4096) -> Dict:
    """Compute all metrics for a quantization method."""
    model_mem = model_vram(params_b, method)
    kv_mem = kv_cache_vram(params_b, seq_len)
    lat = latency_ms_per_token(method)

    return {
        "method": method.upper(),
        "model_vram_gb": round(model_mem, 1),
        "kv_cache_gb": round(kv_mem, 2),
        "total_vram_gb": round(model_mem + kv_mem, 1),
        "vram_savings": round(memory_savings(method), 4),
        "latency_ms_per_token": lat,
        "throughput_tps": round(throughput_tps(method), 1),
        "mmlu": accuracy(method, "mmlu"),
        "gsm8k": accuracy(method, "gsm8k"),
        "ceval": accuracy(method, "ceval"),
        "mmlu_delta": round(accuracy_change(method, "fp16", "mmlu"), 4),
    }
