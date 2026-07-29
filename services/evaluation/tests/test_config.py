"""Tests for evaluation configuration."""

import os
import sys
import tempfile
import json

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestEvalConfig:
    def test_default_values(self):
        from config import EvalConfig
        cfg = EvalConfig()
        assert cfg.model_name == "Qwen/Qwen3-0.6B"
        assert cfg.batch_size == 8
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.0  # greedy for evaluation
        assert cfg.top_p == 1.0
        assert cfg.top_k == -1

    def test_custom_values(self):
        from config import EvalConfig
        cfg = EvalConfig(model_name="test-model", batch_size=16, temperature=0.2)
        assert cfg.model_name == "test-model"
        assert cfg.batch_size == 16
        assert cfg.temperature == 0.2

    def test_benchmark_list(self):
        from config import EvalConfig
        cfg = EvalConfig()
        assert "mmlu" in cfg.benchmarks
        assert "ceval" in cfg.benchmarks
        assert "gsm8k" in cfg.benchmarks
        assert "humaneval" in cfg.benchmarks

    def test_custom_benchmarks(self):
        from config import EvalConfig
        cfg = EvalConfig(benchmarks=["mmlu", "gsm8k"])
        assert len(cfg.benchmarks) == 2

    def test_to_dict(self):
        from config import EvalConfig
        cfg = EvalConfig(model_name="test")
        d = cfg.to_dict()
        assert d["model_name"] == "test"

    def test_from_dict(self):
        from config import EvalConfig
        d = {"model_name": "test", "batch_size": 16}
        cfg = EvalConfig.from_dict(d)
        assert cfg.model_name == "test"
        assert cfg.batch_size == 16


class TestBenchmarkConfig:
    def test_mmlu_config(self):
        from config import MMLUConfig
        cfg = MMLUConfig()
        assert cfg.num_few_shot == 5
        assert cfg.subjects is None  # all subjects

    def test_ceval_config(self):
        from config import CEvalConfig
        cfg = CEvalConfig()
        assert cfg.num_few_shot == 5
        assert cfg.subject is None

    def test_gsm8k_config(self):
        from config import GSM8KConfig
        cfg = GSM8KConfig()
        assert cfg.num_few_shot == 8
        assert cfg.test_only is True

    def test_humaneval_config(self):
        from config import HumanEvalConfig
        cfg = HumanEvalConfig()
        assert cfg.num_samples == 1  # greedy
        assert cfg.test_only is True


class TestReportConfig:
    def test_default_values(self):
        from config import ReportConfig
        cfg = ReportConfig()
        assert cfg.output_dir == "./eval_results"
        assert cfg.format == "markdown"
        assert cfg.include_plots is True

    def test_custom_format(self):
        from config import ReportConfig
        cfg = ReportConfig(format="json")
        assert cfg.format == "json"
        cfg2 = ReportConfig(format="html")
        assert cfg2.format == "html"

    def test_invalid_format_raises(self):
        from config import ReportConfig
        with pytest.raises(ValueError, match="format"):
            ReportConfig(format="invalid")
