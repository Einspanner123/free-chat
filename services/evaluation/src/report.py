"""Report generation for evaluation results."""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from loguru import logger


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    def generate_markdown(self, results: Dict[str, Any], model_name: str = "model") -> str:
        """Generate a Markdown report."""
        lines = [
            f"# Evaluation Report: {model_name}",
            "",
            "## Summary",
            "",
            "| Benchmark | Metric | Value |",
            "|-----------|--------|-------|",
        ]

        for name, metrics in sorted(results.items()):
            if "error" in metrics:
                lines.append(f"| {name} | error | {metrics['error']} |")
                continue
            for metric_key in ["accuracy", "pass@1"]:
                if metric_key in metrics:
                    lines.append(f"| {name} | {metric_key} | {self._format_pct(metrics[metric_key])} |")
            # Show other numeric metrics
            for key, val in metrics.items():
                if key not in ("accuracy", "pass@1", "correct", "total", "subjects", "problems") and isinstance(val, (int, float)):
                    lines.append(f"| {name} | {key} | {val} |")

        lines.extend(["", "## Detailed Results", ""])

        for name, metrics in sorted(results.items()):
            if "subjects" in metrics and isinstance(metrics["subjects"], dict):
                lines.extend([
                    f"### {name} - Per-Subject Breakdown",
                    "",
                    "| Subject | Accuracy |",
                    "|---------|----------|",
                ])
                for subject, sub in sorted(metrics["subjects"].items()):
                    if isinstance(sub, dict) and "accuracy" in sub:
                        lines.append(f"| {subject} | {self._format_pct(sub['accuracy'])} |")
                    elif isinstance(sub, (int, float)):
                        lines.append(f"| {subject} | {self._format_pct(sub)} |")
                lines.append("")

        return "\n".join(lines)

    def generate_comparison_markdown(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a comparison report for multiple models."""
        models = list(results.keys())
        if not models:
            return "No models to compare."

        # Collect all benchmark names
        all_benchmarks = set()
        for model_results in results.values():
            all_benchmarks.update(model_results.keys())

        lines = [
            "# Model Comparison Report",
            "",
            "| Benchmark | " + " | ".join(models) + " |",
            "|-----------|" + "|".join(["--------" for _ in models]) + "|",
        ]

        for bm in sorted(all_benchmarks):
            row_values = []
            for model_name in models:
                model_results = results.get(model_name, {})
                bm_results = model_results.get(bm, {})
                if "error" in bm_results:
                    row_values.append("ERROR")
                elif "accuracy" in bm_results:
                    row_values.append(self._format_pct(bm_results["accuracy"]))
                elif "pass@1" in bm_results:
                    row_values.append(self._format_pct(bm_results["pass@1"]))
                else:
                    row_values.append("-")
            lines.append(f"| {bm} | " + " | ".join(row_values) + " |")

        return "\n".join(lines)

    def generate_html(self, results: Dict[str, Any], model_name: str = "model") -> str:
        """Generate an HTML report."""
        md = self.generate_markdown(results, model_name)
        # Simple HTML wrapping
        html = "<!DOCTYPE html>\n<html>\n<head><title>Evaluation Report</title></head>\n<body>\n"
        html += f"<h1>Evaluation Report: {model_name}</h1>\n"
        html += "<table border='1'>\n<tr><th>Benchmark</th><th>Metric</th><th>Value</th></tr>\n"
        for name, metrics in sorted(results.items()):
            if "error" in metrics:
                html += f"<tr><td>{name}</td><td>error</td><td>{metrics['error']}</td></tr>\n"
                continue
            for key in ["accuracy", "pass@1"]:
                if key in metrics:
                    html += f"<tr><td>{name}</td><td>{key}</td><td>{self._format_pct(metrics[key])}</td></tr>\n"
        html += "</table>\n</body>\n</html>"
        return html

    def generate_json(self, results: Dict[str, Any], model_name: str = "model") -> str:
        """Generate a JSON report."""
        report = {
            "model": model_name,
            "results": results,
        }
        return json.dumps(report, indent=2, ensure_ascii=False)

    def save_report(self, results: Dict[str, Any], path: str, format: str = "markdown", model_name: str = "model"):
        """Generate and save a report.

        Args:
            results: Evaluation results.
            path: Output file path.
            format: "markdown", "html", or "json".
            model_name: Model name for the report.
        """
        if format == "markdown":
            content = self.generate_markdown(results, model_name)
        elif format == "html":
            content = self.generate_html(results, model_name)
        elif format == "json":
            content = self.generate_json(results, model_name)
        else:
            raise ValueError(f"Unsupported format: {format}")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Report saved to {path}")

    def generate_bar_chart(self, data: Dict[str, float], title: str = "") -> str:
        """Generate a simple text-based bar chart.

        Returns: ASCII bar chart string.
        """
        if not data:
            return ""
        max_val = max(data.values()) if data else 1
        max_name_len = max(len(k) for k in data.keys()) if data else 10
        bar_width = 30

        lines = [title, "=" * len(title), ""] if title else []
        for name, val in sorted(data.items(), key=lambda x: -x[1]):
            bar_len = int((val / max_val) * bar_width) if max_val > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{name:<{max_name_len + 2}} {bar} {val:.1%}")
        return "\n".join(lines)

    def generate_radar_chart(self, data: Dict[str, Dict[str, float]], title: str = "") -> str:
        """Generate a comparison chart (text-based).

        Returns: ASCII comparison string.
        """
        if not data:
            return ""
        lines = [title, "=" * len(title), ""] if title else ["Comparison:"]
        for model_name, metrics in data.items():
            lines.append(f"\n{model_name}:")
            for metric, val in sorted(metrics.items()):
                bar = "█" * int(val * 20)
                lines.append(f"  {metric:<15} {bar} {val:.1%}")
        return "\n".join(lines)

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value * 100:.1f}%"


@dataclass
class ExperimentEntry:
    title: str
    data: Dict[str, Any]


class ExperimentDoc:
    """Document experiments with results."""

    def __init__(self):
        self.experiments: List[ExperimentEntry] = []

    def add_experiment(self, title: str, data: Dict[str, Any]):
        """Add an experiment to the document."""
        self.experiments.append(ExperimentEntry(title=title, data=data))

    def generate(self) -> str:
        """Generate a Markdown document with all experiments."""
        lines = ["# Experiment Report", "", ""]
        for i, exp in enumerate(self.experiments, 1):
            lines.append(f"## Experiment {i}: {exp.title}")
            lines.append("")

            # Table for dict-of-dicts data
            data = exp.data
            if data and isinstance(next(iter(data.values())), dict):
                # Multi-model comparison
                models = list(data.keys())
                metrics = set()
                for v in data.values():
                    metrics.update(v.keys())
                metrics = sorted(metrics)

                headers = ["Metric"] + models
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join(["---" for _ in headers]) + "|")

                for metric in metrics:
                    row = [metric]
                    for model in models:
                        val = data.get(model, {}).get(metric, "-")
                        if isinstance(val, float):
                            row.append(f"{val:.1%}")
                        else:
                            row.append(str(val))
                    lines.append("| " + " | ".join(row) + " |")
            else:
                # Simple key-value
                for key, val in data.items():
                    if isinstance(val, float):
                        lines.append(f"- **{key}**: {val:.4f}")
                    else:
                        lines.append(f"- **{key}**: {val}")

            lines.append("")

        return "\n".join(lines)

    def save(self, path: str):
        """Save the experiment document to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        content = self.generate()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Experiment report saved to {path}")
