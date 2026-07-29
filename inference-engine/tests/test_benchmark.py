"""
Tests for benchmark infrastructure.

Each benchmark script should:
1. Accept configurable parameters
2. Return structured results (dict with mean, std, units)
3. Output both human-readable and machine-readable formats
4. Run in CI mode (no GPU) with reference data
"""

import os
import sys
import tempfile
import json

import pytest

_bench = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")
if _bench not in sys.path:
    sys.path.insert(0, _bench)


class TestBenchmarkRunner:
    """Base benchmark runner utilities."""

    def test_result_dataclass(self):
        from benchmark_runner import BenchResult
        r = BenchResult(name="test", value=42.5, unit="ms", std=1.2)
        assert r.name == "test"
        assert r.value == 42.5
        assert r.unit == "ms"

    def test_result_to_dict(self):
        from benchmark_runner import BenchResult
        r = BenchResult(name="latency", value=10.0, unit="ms")
        d = r.to_dict()
        assert d["name"] == "latency"
        assert d["value"] == 10.0

    def test_benchmark_suite(self):
        from benchmark_runner import BenchmarkSuite
        suite = BenchmarkSuite(name="test_suite")
        suite.add_result("latency", 10.0, "ms")
        suite.add_result("throughput", 100.0, "t/s")
        assert len(suite.results) == 2
        assert suite.results[0].name == "latency"

    def test_benchmark_suite_to_json(self):
        from benchmark_runner import BenchmarkSuite
        suite = BenchmarkSuite(name="test")
        suite.add_result("a", 1.0, "ms")
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            suite.to_json(f.name)
            with open(f.name) as f2:
                data = json.load(f2)
        os.unlink(f.name)
        assert data["name"] == "test"
        assert len(data["results"]) == 1

    def test_benchmark_suite_table(self):
        from benchmark_runner import BenchmarkSuite
        suite = BenchmarkSuite(name="t", config={"model": "test"})
        suite.add_result("latency", 10.5, "ms")
        table = suite.to_table()
        assert "latency" in table
        assert "10.5" in table


class TestLatencyBench:
    """TTFT and TPOT measurement."""

    def test_estimate_ttft(self):
        from latency_bench import estimate_ttft
        # model_size_gb, batch_size, has_kv_cache → approximate TTFT
        ttft = estimate_ttft(model_params_b=7, batch_size=1, use_kv_cache=True)
        assert ttft > 0
        assert ttft < 5000  # ms, reasonable range

    def test_estimate_ttft_vs_batch_size(self):
        from latency_bench import estimate_ttft
        # Larger batch: same prompt encoding time (parallel), same first-token time
        # TTFT should be approximately similar across batch sizes
        ttft_1 = estimate_ttft(7, 1, prompt_tokens=512)
        ttft_4 = estimate_ttft(7, 4, prompt_tokens=512)
        # With larger batch, memory contention may increase TTFT slightly
        # But for this analytical model, they should be close
        ratio = ttft_4 / ttft_1
        assert 0.5 < ratio < 2.0  # not wildly different

    def test_estimate_tpot(self):
        from latency_bench import estimate_tpot
        tpot = estimate_tpot(model_params_b=7, batch_size=1)
        assert tpot > 0
        assert tpot < 200  # ms/token, reasonable range

    def test_run_latency_bench(self):
        from latency_bench import run_latency_bench
        suite = run_latency_bench(model_params_b=7, batch_sizes=[1, 2, 4])
        assert len(suite.results) > 0

    def test_latency_vs_sequence_length(self):
        from latency_bench import estimate_ttft
        # Longer prompt → higher TTFT
        short = estimate_ttft(7, prompt_tokens=128)
        long = estimate_ttft(7, prompt_tokens=4096)
        assert long > short


class TestThroughputBench:
    """Throughput measurement."""

    def test_estimate_throughput(self):
        from throughput_bench import estimate_throughput
        tps = estimate_throughput(model_params_b=7, num_concurrent=1)
        assert tps > 0

    def test_throughput_scaling(self):
        from throughput_bench import estimate_throughput
        tps_1 = estimate_throughput(7, 1)
        tps_4 = estimate_throughput(7, 4)
        # More concurrency → higher aggregate throughput
        assert tps_4 > tps_1

    def test_throughput_sublinear_scaling(self):
        from throughput_bench import estimate_throughput
        tps_1 = estimate_throughput(7, 1)
        tps_8 = estimate_throughput(7, 8)
        # Scaling should be sublinear due to memory contention
        assert tps_8 < tps_1 * 8

    def test_run_throughput_bench(self):
        from throughput_bench import run_throughput_bench
        suite = run_throughput_bench(model_params_b=7, concurrency_levels=[1, 2, 4])
        assert len(suite.results) >= 2


class TestMemoryBench:
    """Memory profiling."""

    def test_model_memory_fp16(self):
        from memory_bench import model_memory_fp16
        mem = model_memory_fp16(params_b=7)
        assert abs(mem - 14.0) < 1.0  # ~2 bytes/param * 7B

    def test_kv_cache_memory(self):
        from memory_bench import kv_cache_memory
        mem = kv_cache_memory(params_b=7, seq_len=4096, batch_size=1)
        assert mem > 0
        assert mem < 30  # GB

    def test_kv_cache_scales_with_batch(self):
        from memory_bench import kv_cache_memory
        mem_1 = kv_cache_memory(7, 4096, 1)
        mem_4 = kv_cache_memory(7, 4096, 4)
        assert abs(mem_4 - mem_1 * 4) < 1.0

    def test_kv_cache_scales_with_seq_len(self):
        from memory_bench import kv_cache_memory
        mem_1k = kv_cache_memory(7, 1024, 1)
        mem_4k = kv_cache_memory(7, 4096, 1)
        assert abs(mem_4k - mem_1k * 4) < 1.0

    def test_run_memory_bench(self):
        from memory_bench import run_memory_bench
        suite = run_memory_bench(seq_lens=[1024, 4096], batch_sizes=[1, 4])
        assert len(suite.results) >= 2

    def model_accessibility_map(self):
        """Which model sizes fit on which GPUs."""
        from memory_bench import model_fits_on_gpu
        # 7B FP16 on 24GB
        assert model_fits_on_gpu(params_b=7, method="fp16", gpu_vram_gb=24) is True
        # 70B FP16 doesn't fit on 24GB
        assert model_fits_on_gpu(params_b=70, method="fp16", gpu_vram_gb=24) is False
        # But 70B AWQ fits on 80GB
        assert model_fits_on_gpu(params_b=70, method="awq_int4", gpu_vram_gb=80) is True


class TestQualityBench:
    """Quality impact of optimizations."""

    def test_quantization_accuracy_impact(self):
        from quality_bench import quantization_accuracy
        acc = quantization_accuracy(method="fp16", benchmark="mmlu")
        assert 0 < acc < 1

    def test_eviction_accuracy_impact(self):
        from quality_bench import eviction_accuracy
        acc = eviction_accuracy(method="full", benchmark="mmlu")
        assert acc == 1.0  # full cache = no degradation
        acc_lru = eviction_accuracy(method="lru", benchmark="mmlu")
        assert acc_lru < 1.0  # LRU eviction has some degradation
