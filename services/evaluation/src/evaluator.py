"""Unified evaluator for running benchmarks."""

import json
import os
from typing import List, Dict, Any, Optional

from loguru import logger

from config import EvalConfig
from benchmarks.base import BaseBenchmark
from benchmarks.mmlu import MMLUBenchmark
from benchmarks.ceval import CEvalBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.humaneval import HumanEvalBenchmark


_BENCHMARK_REGISTRY = {
    "mmlu": MMLUBenchmark,
    "ceval": CEvalBenchmark,
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
}


class Evaluator:
    """Unified benchmark evaluator.

    Runs multiple benchmarks on a model and collects results.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.benchmarks: List[BaseBenchmark] = []
        self._results: Dict[str, Any] = {}

        # Register default benchmarks from config
        for name in config.benchmarks:
            if name in _BENCHMARK_REGISTRY:
                self.benchmarks.append(_BENCHMARK_REGISTRY[name]())

    def register_benchmark(self, benchmark: BaseBenchmark):
        """Register a custom benchmark."""
        self.benchmarks.append(benchmark)

    def run(self, name: str, model) -> Dict[str, Any]:
        """Run a single benchmark by name.

        Args:
            name: Benchmark name.
            model: Engine with generate interface.

        Returns:
            Benchmark results dict.
        """
        for bm in self.benchmarks:
            if bm.name() == name:
                logger.info(f"Running benchmark: {name}")
                result = bm.run(model, self.config)
                self._results[name] = result
                return result
        raise ValueError(f"Benchmark '{name}' not found. Available: {[b.name() for b in self.benchmarks]}")

    def run_all(self, model) -> Dict[str, Dict[str, Any]]:
        """Run all registered benchmarks.

        Args:
            model: Engine with generate interface.

        Returns:
            Dict mapping benchmark names to results.
        """
        all_results = {}
        for bm in self.benchmarks:
            try:
                logger.info(f"Running benchmark: {bm.name()}")
                result = bm.run(model, self.config)
                all_results[bm.name()] = result
                self._results[bm.name()] = result
            except Exception as e:
                logger.error(f"Benchmark {bm.name()} failed: {e}")
                all_results[bm.name()] = {"error": str(e)}
        return all_results

    def compare_models(self, models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models on all benchmarks.

        Args:
            models: Dict mapping model names to model instances.

        Returns:
            Dict mapping model names to their benchmark results.
        """
        comparison = {}
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            comparison[model_name] = self.run_all(model)
        return comparison

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all results."""
        summary = {}
        for name, result in self._results.items():
            if "error" in result:
                summary[name] = {"error": result["error"]}
            else:
                metrics = result.copy()
                # Remove verbose fields
                for key in ["subjects", "problems"]:
                    metrics.pop(key, None)
                summary[name] = metrics
        return summary

    def save_results(self, results: Dict[str, Any], path: str):
        """Save results to a JSON file.

        Args:
            results: Results dict.
            path: Output file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")

    def load_results(self, path: str) -> Dict[str, Any]:
        """Load results from a JSON file.

        Args:
            path: Path to results file.

        Returns:
            Loaded results dict.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
