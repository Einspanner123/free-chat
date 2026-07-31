"""
Zero-SCROLLS qasper 任务评估

qasper: 阅读科学文章回答抽象问题，答案简短，F1 自动评分。
input 12K chars (~3K tokens)，适合 0.6B 模型 + 压缩框架。

对比：truncation / project_topic / attention_sink / sink_topic
指标：answer token-level F1（Zero-SCROLLS 官方指标）

用法：.venv/bin/python benchmarks/zero_scrolls/run.py
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "qasper")


def load_items() -> List[Dict]:
    with open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def extract_context(item: Dict) -> str:
    """提取 input 中的文档部分（document_start_index 到 document_end_index）。"""
    return item["input"][item["document_start_index"]:item["document_end_index"]]


def extract_query(item: Dict) -> str:
    """提取问题部分。"""
    return item["input"][item["query_start_index"]:item["query_end_index"]]


def choose_strategy(text: str, tokenizer, budget: int, strategy: str, query: str) -> str:
    if strategy == "truncation":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)

    # 按句子分块
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:budget] if len(text) > budget else text

    # 从问题提取关键实体/词（英文）
    query_words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', query)
                   if w.lower() not in {'what', 'which', 'where', 'when', 'how', 'why',
                                         'the', 'that', 'this', 'these', 'those', 'with', 'from',
                                         'were', 'have', 'been', 'their', 'they', 'there', 'about',
                                         'according', 'article', 'question', 'answer', 'concise'}]
    key = [s for s in sentences if any(w.lower() in s.lower() for w in query_words)]
    other = [s for s in sentences if s not in key]

    if strategy == "project_topic":
        result = list(key)
        total = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in result)
        tail = []
        for i, s in enumerate(reversed(other)):
            turn = i + 1
            ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
            if not ct:
                continue
            nt = len(tokenizer.encode(ct, add_special_tokens=False))
            if total + nt <= budget:
                tail.insert(0, ct)
                total += nt
        return " ".join(result + tail)

    elif strategy == "attention_sink":
        key_text = " ".join(key)
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

    elif strategy == "sink_topic":
        # 组合：关键句前置 + sink + 压缩其余
        key_text = "\n\n".join(key)
        key_tok = len(tokenizer.encode(key_text, add_special_tokens=False))
        remaining = budget - key_tok - 2
        compressed = []
        if remaining > 0:
            for i, s in enumerate(reversed(other)):
                turn = i + 1
                ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
                if not ct:
                    continue
                nt = len(tokenizer.encode(ct, add_special_tokens=False))
                if sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in compressed) + nt <= remaining:
                    compressed.insert(0, ct)
        return "\n\n" + key_text + "\n\n" + " ".join(compressed)

    return text


def compute_f1(pred: str, ref: str) -> float:
    """Zero-SCROLLS 官方 F1：token 级。"""
    p_tokens = pred.lower().split()
    r_tokens = ref.lower().split()
    if not p_tokens or not r_tokens:
        return 0.0
    common = set(p_tokens) & set(r_tokens)
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(model, tokenizer, device, items, strategy, budget, max_new=50) -> Dict:
    f1_scores = []
    times = []
    per_item = []

    for item in items:
        doc = extract_context(item)
        query = extract_query(item)
        ctx = choose_strategy(doc, tokenizer, budget, strategy, query)
        used_tok = len(tokenizer.encode(ctx, add_special_tokens=False))

        prompt = f"Article:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        times.append(dt)

        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        f1 = compute_f1(resp, item["output"])
        f1_scores.append(f1)
        per_item.append({
            "id": item["id"], "f1": round(f1, 3),
            "full_tokens": len(tokenizer.encode(doc, add_special_tokens=False)),
            "used_tokens": used_tok,
            "gold": item["output"][:50], "pred": resp[:50],
        })

    return {
        "strategy": strategy, "budget": budget,
        "avg_f1": sum(f1_scores) / len(f1_scores),
        "avg_latency_s": round(sum(times) / len(times), 2),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--output", default="results/zero_scrolls_qasper.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    items = load_items()[:args.samples]
    print(f"Zero-SCROLLS qasper: {len(items)} samples\n")

    results = {"config": {"model": args.model, "samples": len(items), "budgets": args.budgets}, "strategies": []}
    strategies = ["truncation", "project_topic", "attention_sink", "sink_topic"]

    for budget in args.budgets:
        print(f"=== Budget: {budget} tokens ===")
        for strat in strategies:
            r = evaluate(model, tokenizer, device, items, strat, budget)
            print(f"  {strat:<20} F1={r['avg_f1']:.3f}  latency={r['avg_latency_s']}s")
            results["strategies"].append(r)
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
