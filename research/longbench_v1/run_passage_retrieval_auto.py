"""
L2 端到端验证：intent-routing（strategy="auto"）在真实 LongBench 数据上。

与 run_passage_retrieval.py 的区别：这里直接复用生产 context-engine 的
ContextPipeline（含 Router/RuleClassifier），而非脚本内联实现。auto 策略
内部路由到哪条策略、置信度多少，都记录在 per_item 里——验证「意图路由 →
最优策略」链路端到端有效。

策略对比：auto（新）/ truncation / sink_topic / bm25_top1
指标：段落号预测准确率（与 README 口径一致）

用法：
  .venv/bin/python research/longbench_v1/run_passage_retrieval_auto.py \
      --samples 50 --budget 1024
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "services", "context-engine", "src")
sys.path.insert(0, SRC)

from pipeline import ContextPipeline, PipelineConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data")


def load_items(task: str = "passage_retrieval_en") -> List[Dict]:
    with open(os.path.join(DATA_DIR, f"{task}.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_context(pipe: ContextPipeline, text: str, tokenizer, query: str) -> Dict:
    """Run the real pipeline; returns metadata incl. routing for auto."""
    result = pipe.build_with_metadata(text, tokenizer, query=query)
    return result


def evaluate(model, tokenizer, device, items, strategy, budget, task="passage_retrieval_en",
             max_new=15) -> Dict:
    pipe = ContextPipeline(PipelineConfig(strategy=strategy, budget=budget))
    correct = 0
    times = []
    per_item = []
    routing = Counter()

    for item in items:
        query = item["input"]
        result = build_context(pipe, item["context"], tokenizer, query)
        ctx = result["context"]

        # 注意：不要给数字例子（如 "e.g., Paragraph 5"）——0.6B 会被锚定到例子数字。
        if task == "passage_retrieval_zh":
            prompt = f"段落列表：\n{ctx}\n\n找到与描述匹配的段落：{query}\n\n请回答段落编号："
        else:
            prompt = f"Passages:\n{ctx}\n\nFind the passage that matches: {query}\n\nAnswer with the paragraph number:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if task == "passage_retrieval_zh":
            m = re.search(r'段落\s*(\d+)', resp)
        else:
            m = re.search(r'(?:Paragraph|paragraph)?\s*(\d+)', resp)
        pred = m.group(1) if m else None
        gold = re.search(r'(\d+)', item["answers"][0]).group(1)
        is_correct = pred == gold
        if is_correct:
            correct += 1

        row = {
            "id": item["_id"], "gold": gold, "pred": pred, "correct": is_correct,
            "full_tokens": len(tokenizer.encode(item["context"], add_special_tokens=False)),
            "used_tokens": result["tokens"],
        }
        if result.get("routed_from") == "auto":
            row["intent"] = result.get("intent")
            row["confidence"] = result.get("confidence")
            row["routed_strategy"] = result["strategy"]
            # tuple key → str for JSON
            routing_key = f"{result.get('intent')}->{result['strategy']}"
            routing[routing_key] += 1
        per_item.append(row)

    return {
        "strategy": strategy, "budget": budget,
        "accuracy": correct / len(items), "correct": correct, "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
        "routing": dict(routing),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--task", default="passage_retrieval_en",
                        choices=["passage_retrieval_en", "passage_retrieval_zh"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--strategies", nargs="+",
                        default=["truncation", "sink_topic", "bm25_top1", "auto"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}, Device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    items = load_items(task=args.task)[:args.samples]
    print(f"LongBench {args.task}: {len(items)} samples, budget={args.budget}\n")

    base = os.path.dirname(os.path.abspath(__file__))
    if args.output is None:
        args.output = os.path.join(base, "results",
                                   f"{args.task}_{args.model.split('/')[-1].replace('-Instruct','')}_auto.json")
    elif not os.path.isabs(args.output):
        args.output = os.path.join(base, args.output)
    results = {"config": {"model": args.model, "samples": len(items), "budget": args.budget},
               "strategies": []}

    for strat in args.strategies:
        r = evaluate(model, tokenizer, device, items, strat, args.budget, task=args.task)
        print(f"  {strat:<20} acc={r['accuracy']:.1%} ({r['correct']}/{r['total']})  latency={r['avg_latency_s']}s")
        if r["routing"]:
            print(f"    routing: {r['routing']}")
        results["strategies"].append({k: v for k, v in r.items() if k != "per_item"})
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
