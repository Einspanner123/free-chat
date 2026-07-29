"""
Benchmark runner: common infrastructure for all benchmarks.

Provides:
  - BenchResult: single measurement (name, value, unit, std)
  - BenchmarkSuite: collection of results with metadata
  - CSV/JSON/table output
  - CI mode: returns reference data when no GPU available
"""

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class BenchResult:
    name: str
    value: float
    unit: str
    std: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "std": self.std,
            "metadata": self.metadata,
        }

    def __str__(self):
        return f"{self.name}: {self.value:.2f} ± {self.std:.2f} {self.unit}"


class BenchmarkSuite:
    """Collection of benchmark results with metadata."""

    def __init__(self, name: str = "", config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.results: List[BenchResult] = []

    def add_result(self, name: str, value: float, unit: str = "", std: float = 0.0, **metadata):
        self.results.append(BenchResult(
            name=name, value=value, unit=unit, std=std, metadata=metadata
        ))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_table(self) -> str:
        if not self.results:
            return "(no results)"
        lines = [f"# {self.name}", ""]
        if self.config:
            lines.append(f"Config: {json.dumps(self.config)}")
            lines.append("")
        lines.append(f"| {'Metric':<25} | {'Value':<12} | {'Std':<10} | {'Unit':<8} |")
        lines.append(f"|{'-'*27}|{'-'*14}|{'-'*12}|{'-'*10}|")
        for r in self.results:
            lines.append(f"| {r.name:<25} | {r.value:<12.2f} | {r.std:<10.2f} | {r.unit:<8} |")
        return "\n".join(lines)

    def to_csv(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["name", "value", "std", "unit"])
            for r in self.results:
                w.writerow([r.name, r.value, r.std, r.unit])
