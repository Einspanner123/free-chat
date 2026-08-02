"""
Zero-SCROLLS quality 对照实验：0.6B vs 7B

用相同压缩上下文，对比模型规模对准确率的影响。
判断是否到达理解边界：
- 7B 能答、0.6B 不能 → 0.6B 理解边界
- 两者都不能       → 框架/压缩丢失信息

用法：
    .venv/bin/python benchmarks/zero_scrolls/run_compare.py --model Qwen/Qwen2.5-7B-Instruct
    .venv/bin/python benchmarks/zero_scrolls/run_compare.py --model Qwen/Qwen3-0.6B
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "quality")
# 复用 run_quality.py 的策略函数
sys.path.insert(0, os.path.dirname(__file__))
from run_quality import load_items, extract_parts, choose_strategy, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    items = load_items()[:args.samples]
    print(f"Zero-SCROLLS quality: {len(items)} samples\n")

    if args.output is None:
        model_short = args.model.split("/")[-1].replace("-Instruct", "").replace(".", "_")
        args.output = f"results/quality_{model_short}.json"

    results = {"config": {"model": args.model, "samples": len(items), "budgets": args.budgets}, "strategies": []}
    strategies = ["truncation", "project_topic", "attention_sink", "sink_topic"]

    for budget in args.budgets:
        print(f"=== Budget: {budget} tokens ===")
        for strat in strategies:
            r = evaluate(model, tokenizer, device, items, strat, budget)
            print(f"  {strat:<20} acc={r['accuracy']:.1%} ({r['correct']}/{r['total']})  latency={r['avg_latency_s']}s")
            results["strategies"].append(r)
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
