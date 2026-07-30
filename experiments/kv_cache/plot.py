"""Generate plots from KV cache experiment results."""

import argparse
import json
import os

def load_results(path: str = "results/results.json"):
    with open(path) as f:
        return json.load(f)

def render_chart(data: list, title: str, filename: str, output_dir: str = "plots"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = sorted(set(d["method"] for d in data))
        seq_lens = sorted(set(d["seq_len"] for d in data))
        for method in methods:
            vals = [d["value"] for d in data if d["method"] == method]
            ax.plot(seq_lens[:len(vals)], vals, marker="o", label=method)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {output_dir}/{filename}")
    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.json")
    parser.add_argument("--output", default="plots")
    args = parser.parse_args()
    results = load_results(args.input)
    os.makedirs(args.output, exist_ok=True)
    for metric in ["kv_cache_gb", "latency_ms_per_token"]:
        data = []
        for r in results.get("results", []):
            if metric in r:
                data.append({"method": r["method"], "seq_len": r["seq_len"], "value": r[metric]})
        render_chart(data, f"KV Cache: {metric}", f"{metric}.png", args.output)

if __name__ == "__main__":
    main()
