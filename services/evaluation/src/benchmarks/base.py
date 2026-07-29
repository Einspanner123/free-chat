"""Base benchmark class.

Defines the interface that all benchmarks must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    @abstractmethod
    def name(self) -> str:
        """Return the benchmark name."""
        ...

    @abstractmethod
    def run(self, model: Any, config: Any) -> Dict[str, Any]:
        """Run the benchmark on the given model.

        Args:
            model: An engine with generate/stream_generate interface.
            config: EvalConfig or per-benchmark config.

        Returns:
            Dict with evaluation results.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return computed metrics after running."""
        ...
