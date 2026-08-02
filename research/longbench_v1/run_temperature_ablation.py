"""
Temperature ablation on LongBench passage_retrieval_en.

Motivation: YaRN attention temperature scaling sharpens attention on
long contexts. We test whether lower sampling temperature improves
retrieval accuracy with the same optimized context (BM25 top-1).

Setup:
- Context: BM25 top-1 paragraph (the framework's best strategy)
- Model: Qwen3-0.6B
- Variable: temperature {0.0, 0.3, 0.5, 0.7, 1.0}
- Metric: paragraph retrieval accuracy

Expected: lower temperature → sharper attention → higher accuracy
(if YaRN's intuition holds at inference sampling level).

Usage: .venv/bin/python research/longbench_v1/run_temperature_ablation.py
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

# 复用 context-engine 的 BM25 检索
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "services", "context-engine", "src"))
from pipeline import ContextPipeline, PipelineConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data")


def load_items() -> List[Dict]:
    with open(os.path.join(DATA_DIR, "passage_retrieval_en.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def evaluate(model, tokenizer, device, items, temperature, budget=2048) -> Dict:
    """用 BM25 top-1 上下文 + 指定 temperature 检索。"""
    pipe = ContextPipeline(PipelineConfig(strategy="bm25_top1", budget=budget, retriever="bm25", top_k=1))
    correct = 0
    times = []
    per_item = []

    for item in items:
        ctx = pipe.build(item["context"], tokenizer, query=item["input"])
        prompt = f"Passages:\n{ctx}\n\nFind the passage that matches: {item['input']}\n\nAnswer with the paragraph number:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            if temperature <= 0.0:
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=10, temperature=temperature, do_sample=True)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = re.search(r'(\d+)', resp)
        pred_n = pred.group(1) if pred else None
        gold = re.search(r'(\d+)', item["answers"][0]).group(1)
        is_correct = pred_n == gold
        if is_correct:
            correct += 1
        per_item.append({"gold": gold, "pred": pred_n, "correct": is_correct})

    return {
        "temperature": temperature,
        "accuracy": correct / len(items),
        "correct": correct, "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="results/temperature_ablation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    items = load_items()[:args.samples]
    print(f"passage_retrieval_en: {len(items)} samples\n")

    results = {
        "config": {"model": args.model, "samples": len(items), "temperatures": args.temperatures, "seeds": args.seeds},
        "runs": [],
        "summary": {},
    }

    # 对每个 temperature 跑多个 seed，统计均值/方差（消融严谨性）
    for temp in args.temperatures:
        accs = []
        run_results = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            r = evaluate(model, tokenizer, device, items, temp)
            run_results.append(r)
            accs.append(r["accuracy"])
            print(f"  temp={temp:.1f} seed={seed}: acc={r['accuracy']:.1%} ({r['correct']}/{r['total']})")
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        results["runs"].extend(run_results)
        results["summary"][str(temp)] = {"mean_acc": round(mean, 4), "std": round(std, 4), "per_seed": accs}
        print(f"  → temp={temp:.1f} 均值={mean:.1%} ± {std:.1%}")
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
