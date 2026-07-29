"""Tests for the unified evaluator."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestEvaluator:
    def test_init_basic(self):
        from evaluator import Evaluator
        from config import EvalConfig
        eval = Evaluator(config=EvalConfig(model_name="test-model"))
        assert eval.config.model_name == "test-model"
        assert len(eval.benchmarks) > 0

    def test_init_with_custom_benchmarks(self):
        from evaluator import Evaluator
        from config import EvalConfig
        eval = Evaluator(config=EvalConfig(benchmarks=["mmlu", "gsm8k"]))
        names = [b.name().lower() for b in eval.benchmarks]
        assert "mmlu" in str(names)
        assert "gsm8k" in str(names)

    def test_register_custom_benchmark(self):
        from evaluator import Evaluator
        from benchmarks.base import BaseBenchmark
        from config import EvalConfig

        class CustomBM(BaseBenchmark):
            def name(self): return "custom"
            def run(self, model, config): return {"accuracy": 1.0}
            def get_metrics(self): return {"accuracy": 1.0}

        eval = Evaluator(config=EvalConfig())
        eval.register_benchmark(CustomBM())
        names = [b.name() for b in eval.benchmarks]
        assert "custom" in names

    def test_run_all_benchmarks(self):
        from evaluator import Evaluator
        from config import EvalConfig
        from benchmarks.base import BaseBenchmark

        class MockBM(BaseBenchmark):
            def name(self): return "mock"
            def run(self, model, config): return {"accuracy": 0.8, "num_questions": 10}
            def get_metrics(self): return {"accuracy": 0.8}

        mock_engine = MagicMock()
        mock_engine.generate.return_value = "test"

        eval = Evaluator(config=EvalConfig(benchmarks=[]))
        eval.register_benchmark(MockBM())
        results = eval.run_all(mock_engine)
        assert "mock" in results
        assert results["mock"]["accuracy"] == 0.8

    def test_run_single_benchmark(self):
        from evaluator import Evaluator
        from config import EvalConfig
        from benchmarks.base import BaseBenchmark

        class MockBM(BaseBenchmark):
            def name(self): return "single"
            def run(self, model, config): return {"accuracy": 0.9}
            def get_metrics(self): return {"accuracy": 0.9}

        mock_engine = MagicMock()
        eval = Evaluator(config=EvalConfig(benchmarks=[]))
        eval.register_benchmark(MockBM())
        result = eval.run("single", mock_engine)
        assert result["accuracy"] == 0.9

    def test_run_nonexistent_benchmark(self):
        from evaluator import Evaluator
        from config import EvalConfig
        eval = Evaluator(config=EvalConfig())
        with pytest.raises(ValueError, match="not found"):
            eval.run("nonexistent", MagicMock())

    def test_summary_empty(self):
        from evaluator import Evaluator
        from config import EvalConfig
        eval = Evaluator(config=EvalConfig())
        summary = eval.summary()
        assert isinstance(summary, dict)

    def test_compare_models(self):
        from evaluator import Evaluator
        from config import EvalConfig
        from benchmarks.base import BaseBenchmark

        class MockBM(BaseBenchmark):
            def name(self): return "mock"
            def run(self, model, config): return {"accuracy": 0.8}
            def get_metrics(self): return {"accuracy": 0.8}

        eval = Evaluator(config=EvalConfig(benchmarks=[]))
        eval.register_benchmark(MockBM())

        model_a = MagicMock()
        model_b = MagicMock()
        comparison = eval.compare_models({"model_a": model_a, "model_b": model_b})
        assert "model_a" in comparison
        assert "model_b" in comparison


class TestEvaluatorPersistence:
    def test_save_results(self):
        from evaluator import Evaluator
        from config import EvalConfig
        eval = Evaluator(config=EvalConfig())
        results = {"mmlu": {"accuracy": 0.7, "num_questions": 100}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            eval.save_results(results, path)
            assert os.path.exists(path)

            import json
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["mmlu"]["accuracy"] == 0.7

    def test_load_results(self):
        from evaluator import Evaluator
        from config import EvalConfig
        eval = Evaluator(config=EvalConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            import json
            with open(path, 'w') as f:
                json.dump({"mmlu": {"accuracy": 0.7}}, f)

            loaded = eval.load_results(path)
            assert loaded["mmlu"]["accuracy"] == 0.7
