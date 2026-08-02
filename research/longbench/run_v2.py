"""
LongBench v2 评估（zai-org/LongBench-v2，阿里通义）

503 个超长上下文选择题样本（平均 218K tokens，max 400万）。
Qwen3-0.6B 上限 128K，大部分样本超限 → 必须压缩。

对比策略（同 token 预算）：
- Truncation：直接截断
- Project + Topic：话题感知压缩（保留含关键词句子）
- Attention Sink：关键信息前置布局

评估：选择题准确率（官方格式，A/B/C/D）

用法：.venv/bin/python benchmarks/longbench/run_v2.py --samples-per-domain 2
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PUBLISHED_BASELINES = {
    "GPT-4o": 0.513,
    "Claude-3.5-Sonnet": 0.496,
    "GPT-4o-mini": 0.459,
    "Qwen2.5-72B": 0.512,
}


def choose_strategy(text: str, tokenizer, budget: int, strategy: str, question: str) -> str:
    """按策略压缩文本到预算。"""
    if strategy == "truncation":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:budget] if len(text) > budget else text

    # 从问题提取关键实体（大写词）
    question_words = [w for w in re.findall(r'\b[A-Z][a-zA-Z]+\b', question) if len(w) > 2]
    key = [s for s in sentences if any(w.lower() in s.lower() for w in question_words)]
    other = [s for s in sentences if s not in key]

    if strategy == "project_topic":
        # 保留关键句 + 分级压缩其余
        result = list(key)
        total = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in result)
        for i, s in enumerate(reversed(other)):
            turn = i + 1
            ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
            if not ct:
                continue
            nt = len(tokenizer.encode(ct, add_special_tokens=False))
            if total + nt <= budget:
                result.insert(0, ct)
                total += nt
        return " ".join(result)

    elif strategy == "attention_sink":
        # sink → 关键句 → 压缩其余
        key_text = "\n\n".join(key)
        key_tok = len(tokenizer.encode(key_text, add_special_tokens=False))
        remaining = budget - key_tok - 2
        compressed = []
        if remaining > 0:
            for i, s in enumerate(reversed(other)):
                ct = s if i < 5 else (s[:100] if i < 20 else (s[:50] if i < 50 else ""))
                if not ct:
                    continue
                nt = len(tokenizer.encode(ct, add_special_tokens=False))
                if sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in compressed) + nt <= remaining:
                    compressed.insert(0, ct)
        return "\n\n" + key_text + "\n\n" + " ".join(compressed)

    return text


def build_prompt(context: str, item: Dict) -> str:
    """构造 LongBench v2 格式的选择题 prompt。"""
    q = item["question"]
    choices = f"""A. {item['choice_A']}
B. {item['choice_B']}
C. {item['choice_C']}
D. {item['choice_D']}"""
    return f"""Context:
{context}

Question: {q}

{choices}

Answer with only the letter (A/B/C/D):"""


def evaluate(model, tokenizer, device, items, strategy, budget, max_new=10) -> Dict:
    correct = 0
    times = []
    per_item = []

    for item in items:
        full_ctx = item["context"]
        ctx = choose_strategy(full_ctx, tokenizer, budget, strategy, item["question"])
        prompt = build_prompt(ctx, item)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
        # 提取答案字母
        pred = None
        for letter in "ABCD":
            if letter in resp[:3] or f"({letter})" in resp or f"{letter}." in resp[:3]:
                pred = letter
                break
        is_correct = pred == item["answer"]
        if is_correct:
            correct += 1
        per_item.append({
            "id": item["_id"], "domain": item["domain"], "answer": item["answer"],
            "predicted": pred, "correct": is_correct,
            "context_tokens": len(tokenizer.encode(full_ctx, add_special_tokens=False)),
            "used_tokens": len(tokenizer.encode(ctx, add_special_tokens=False)),
        })

    return {
        "strategy": strategy,
        "budget": budget,
        "accuracy": correct / len(items),
        "correct": correct,
        "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples-per-domain", type=int, default=2)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--output", default="results/longbench_v2.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_dataset('zai-org/LongBench-v2')['train']
    print(f"LongBench v2: {len(ds)} samples total")

    # 每个 domain 抽样
    from collections import defaultdict
    by_domain = defaultdict(list)
    for item in ds:
        by_domain[item["domain"]].append(item)

    selected = []
    for domain, items in by_domain.items():
        selected.extend(items[:args.samples_per_domain])
    print(f"Selected: {len(selected)} samples ({args.samples_per_domain} per domain x {len(by_domain)} domains)\n")

    results = {
        "config": {"model": args.model, "samples_per_domain": args.samples_per_domain, "budgets": args.budgets},
        "published_baselines": PUBLISHED_BASELINES,
        "strategies": [],
        "selected_ids": [s["_id"] for s in selected],
    }

    strategies = ["truncation", "project_topic", "attention_sink"]

    for budget in args.budgets:
        print(f"=== Budget: {budget} tokens ===")
        for strat in strategies:
            r = evaluate(model, tokenizer, device, selected, strat, budget)
            print(f"  {strat:<20} accuracy={r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            results["strategies"].append(r)
        print()

    # 按 domain 分析最佳策略
    print("=== Per-domain accuracy (best strategy) ===")
    best = max(results["strategies"], key=lambda r: r["accuracy"])
    domain_acc = defaultdict(lambda: [0, 0])
    for pi in best["per_item"]:
        domain_acc[pi["domain"]][0] += int(pi["correct"])
        domain_acc[pi["domain"]][1] += 1
    for domain, (c, t) in sorted(domain_acc.items()):
        print(f"  {domain:<40} {c}/{t} = {c/t:.0%}")

    print(f"\nPublished baselines: {json.dumps(PUBLISHED_BASELINES)}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
