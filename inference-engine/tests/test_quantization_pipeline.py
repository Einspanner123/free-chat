"""
Tests: quantization pipeline that runs actual model inference.

In CI mode (no GPU), falls back to reference data.
On GPU hardware, loads models with different quantization methods
and measures memory/latency/accuracy.
"""

import os
import sys
import pytest

_bench = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")
if _bench not in sys.path:
    sys.path.insert(0, _bench)


class TestQuantizationPipeline:
    """Quantization benchmark pipeline."""

    def test_pipeline_config(self):
        from quantization_pipeline import QuantBenchConfig
        cfg = QuantBenchConfig()
        assert cfg.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert "fp16" in cfg.methods
        assert "awq" in cfg.methods

    def test_benchmark_in_ci_mode(self):
        """Without GPU, returns reference data."""
        from quantization_pipeline import QuantBenchConfig, run_quantization_bench
        cfg = QuantBenchConfig(methods=["fp16", "awq"])
        results = run_quantization_bench(cfg)
        assert len(results) > 0
        for r in results:
            assert "method" in r
            assert "vram_gb" in r
            assert "mmlu" in r

    def test_format_result_table(self):
        from quantization_pipeline import QuantBenchConfig, run_quantization_bench, format_results
        cfg = QuantBenchConfig(methods=["fp16", "awq"])
        results = run_quantization_bench(cfg)
        table = format_results(results)
        assert "FP16" in table
        assert "AWQ" in table

    def test_memory_savings_calculation(self):
        from quantization_pipeline import compute_memory_savings
        savings = compute_memory_savings(method="awq")
        assert 0.5 < savings < 0.8  # AWQ saves 50-80% memory

    def test_accuracy_impact(self):
        from quantization_pipeline import compute_accuracy_impact
        impact = compute_accuracy_impact(method="awq")
        assert 0.0 < impact < 0.05  # <5% accuracy degradation
