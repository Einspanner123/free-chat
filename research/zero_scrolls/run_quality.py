"""
Zero-SCROLLS quality 任务评估（选择题，准确率可自动评估）

quality: 阅读故事回答选择题（A/B/C/D），validation 21 个带答案样本。
上下文 ~5K chars (~1.3K tokens)，适合 0.6B + 压缩框架。

对比：truncation / project_topic / attention_sink / sink_topic
指标：准确率

用法：.venv/bin/python benchmarks/zero_scrolls/run_quality.py
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "quality")


def load_items() -> List[Dict]:
    with open(os.path.join(DATA_DIR, "validation.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def extract_parts(item: Dict):
    """从 input 提取文档、问题和选项。"""
    doc = item["input"][item["document_start_index"]:item["document_end_index"]]
    qa = item["input"][item["query_start_index"]:]
    # 去掉开头的 'Question and Possible Answers:' 标记
    return doc, qa


def choose_strategy(text: str, tokenizer, budget: int, strategy: str, question: str) -> str:
    if strategy == "truncation":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:budget] if len(text) > budget else text

    # 从问题提取关键实体/专有名词
    question_words = [w for w in re.findall(r'\b[A-Z][a-z]+\b', question)
                      if w.lower() not in {'question', 'possible', 'answers', 'answer'}]
    key = [s for s in sentences if any(w in s for w in question_words)]
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


def evaluate(model, tokenizer, device, items, strategy, budget, max_new=10) -> Dict:
    correct = 0
    times = []
    per_item = []

    for item in items:
        doc, qa = extract_parts(item)
        ctx = choose_strategy(doc, tokenizer, budget, strategy, qa)
        used_tok = len(tokenizer.encode(ctx, add_special_tokens=False))

        prompt = f"Story:\n{ctx}\n\n{qa}\nAnswer with only the letter (A/B/C/D):"
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
            if letter in resp[:3] or f"({letter})" in resp:
                pred = letter
                break
        is_correct = pred == item["output"]
        if is_correct:
            correct += 1
        per_item.append({
            "id": item["id"], "gold": item["output"], "pred": pred, "correct": is_correct,
            "full_tokens": len(tokenizer.encode(doc, add_special_tokens=False)),
            "used_tokens": used_tok,
        })

    return {
        "strategy": strategy, "budget": budget,
        "accuracy": correct / len(items), "correct": correct, "total": len(items),
        "avg_latency_s": round(sum(times) / len(times), 2),
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--output", default="results/quality_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    items = load_items()[:args.samples]
    print(f"Zero-SCROLLS quality: {len(items)} samples\n")

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
