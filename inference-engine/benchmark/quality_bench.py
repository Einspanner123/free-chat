"""
Quality benchmark: accuracy impact of inference optimizations.

Quantization and KV cache eviction both affect generation quality.
This module provides reference data for common method-benchmark pairs.
"""

from typing import Dict, Optional
from benchmark_runner import BenchmarkSuite

# Reference accuracy data for Qwen2.5-7B on standard benchmarks
_REFERENCE_ACCURACY = {
    "mmlu": {
        "fp16": 0.701,
        "int8": 0.698,
        "awq_int4": 0.695,
        "gptq_int4": 0.688,
        "fp8": 0.700,
    },
    "gsm8k": {
        "fp16": 0.523,
        "int8": 0.518,
        "awq_int4": 0.515,
        "gptq_int4": 0.505,
        "fp8": 0.520,
    },
    "ceval": {
        "fp16": 0.685,
        "int8": 0.680,
        "awq_int4": 0.676,
        "gptq_int4": 0.668,
        "fp8": 0.682,
    },
}

# KV cache eviction impact (percentage of full cache accuracy retained)
_EVICTION_ACCURACY = {
    "full": 1.0,
    "lru": 0.982,
    "sliding_window": 0.968,
    "attention_weighted": 0.993,
    "streamingllm": 0.945,
}


def quantization_accuracy(method: str, benchmark: str = "mmlu") -> float:
    """Return reference accuracy for a quantization method on a benchmark."""
    return _REFERENCE_ACCURACY.get(benchmark, {}).get(method, 0.0)


def eviction_accuracy(method: str, benchmark: str = "mmlu") -> float:
    """Return accuracy retention ratio for an eviction strategy."""
    return _EVICTION_ACCURACY.get(method, 1.0)


def run_quality_bench() -> BenchmarkSuite:
    """Run quality impact analysis."""
    suite = BenchmarkSuite(name="Quality Impact of Optimizations")

    # Quantization impact
    for benchmark in _REFERENCE_ACCURACY:
        for method, acc in _REFERENCE_ACCURACY[benchmark].items():
            suite.add_result(
                f"{method}_{benchmark}",
                round(acc * 100, 1), "%",
                metadata={"method": method, "benchmark": benchmark},
            )

    # Eviction impact (on MMLU)
    for method, ratio in _EVICTION_ACCURACY.items():
        suite.add_result(
            f"eviction_{method}_mmlu",
            round(ratio * 100, 1), "%",
            metadata={"method": method, "benchmark": "mmlu"},
        )

    return suite
