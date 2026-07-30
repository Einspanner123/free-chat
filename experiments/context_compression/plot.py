"""
Generate plots from experiment results.

Usage:
    python plot.py                              # from results/results.json
    python plot.py --input results/results.json
    python plot.py --output plots/

Output:
    plots/compression_ratio.png        — bar chart per method per budget
    plots/entity_recall.png            — entity recall comparison
    plots/latency_vs_budget.png        — latency reduction
    plots/tradeoff.png                 — recall vs compression scatter
"""

import argparse
import json
import os
import sys


def load_results(path: str = "results/results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def render_bar_chart(data: list, x_label: str, y_label: str, title: str,
                     filename: str, output_dir: str = "plots"):
    """Render a bar chart. Uses matplotlib if available, otherwise ASCII."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)

        labels = [d["label"] for d in data]
        groups = sorted(set(d["group"] for d in data))
        x = np.arange(len(labels))
        n_groups = len(groups)
        width = 0.8 / n_groups

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, group in enumerate(groups):
            values = [d["values"].get(group, 0) for d in data]
            offset = (i - n_groups / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=group)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {path}")

    except ImportError:
        _render_ascii(data, title, output_dir, filename)


def _render_ascii(data: list, title: str, output_dir: str, filename: str):
    """ASCII fallback when matplotlib is not available."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename.replace(".png", ".txt"))

    lines = [f"# {title}", ""]
    for d in data:
        bars = " ".join(f"{k}={v:.0%}" for k, v in d["values"].items())
        lines.append(f"  {d['label']:<20} {bars}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ASCII plot saved: {path}")


def generate_all_plots(results: dict, output_dir: str = "plots"):
    """Generate all standard plots from experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    budgets = results["config"]["budgets"]

    # Prepare data
    methods_map = {
        "full_context": "Full Context",
        "truncation": "Truncation",
        "hierarchical": "Hierarchical",
    }

    # Plot 1: Compression ratio per budget
    data = []
    for budget in budgets:
        for key, label in methods_map.items():
            entries = results["methods"].get(key, [])
            entry = next((e for e in entries if e["budget"] == budget), None)
            if entry:
                data.append({
                    "label": f"{label}\n{budget}",
                    "group": label,
                    "values": {"compression": entry["compression_ratio"],
                               "recall": entry["entity_recall"]},
                })
    render_bar_chart(data, "Method / Budget", "Ratio",
                     "Compression Ratio vs Entity Recall", "compression_ratio.png", output_dir)

    # Plot 2: Entity recall comparison
    data2 = []
    for budget in budgets:
        for key, label in methods_map.items():
            entries = results["methods"].get(key, [])
            entry = next((e for e in entries if e["budget"] == budget), None)
            if entry:
                data2.append({
                    "label": f"{budget}",
                    "group": label,
                    "values": {"recall": entry["entity_recall"]},
                })
    render_bar_chart(data2, "Budget (tokens)", "Entity Recall",
                     "Entity Recall by Budget", "entity_recall.png", output_dir)

    # Plot 3: Latency vs budget
    data3 = []
    for budget in budgets:
        for key, label in methods_map.items():
            entries = results["methods"].get(key, [])
            entry = next((e for e in entries if e["budget"] == budget), None)
            if entry:
                data3.append({
                    "label": f"{budget}",
                    "group": label,
                    "values": {"latency_ms": entry["compressed_latency_ms"] / 1000},
                })
    render_bar_chart(data3, "Budget (tokens)", "Latency (s)",
                     "Estimated Latency by Budget", "latency_vs_budget.png", output_dir)

    print(f"All plots generated in {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.json")
    parser.add_argument("--output", default="plots")
    args = parser.parse_args()

    results = load_results(args.input)
    generate_all_plots(results, args.output)


if __name__ == "__main__":
    main()
