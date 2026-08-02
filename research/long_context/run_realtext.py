"""
真实文本长上下文 benchmark（RULER 风格 needle 类型扩展）。

数据源：Project Gutenberg 真实书籍（Pride and Prejudice, Moby Dick, War and Peace, Alice in Wonderland）

Needle 类型：
1. niah_single    — 单值检索："The secret code is CODE-000"
2. niah_multivalue— 多值检索："Alice's code is 111, Bob's code is 222" → 问 Bob 的
3. niah_multikey  — 多属性检索："Alice's code is 111. Alice's city is Paris" → 问其中一个属性
4. niah_multihop  — 多跳推理：需要结合两条信息推理

策略对比：Full Context / Truncation / Project+Topic / Attention Sink / RAG

用法：
    .venv/bin/python benchmarks/long_context/run_realtext.py --context-tokens 8192
"""

import argparse
import json
import os
import random
import re
import time
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_real_text(name: str = "pride_and_prejudice") -> str:
    path = os.path.join(DATA_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def prepare_real_context(text: str, target_tokens: int, tokenizer) -> str:
    """取真实文本的连续片段，截断到目标 token 数。"""
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Needle 生成器（RULER 风格）
# ---------------------------------------------------------------------------

def make_niah_single(context: str, needle_idx: int, pos: float, seed: int) -> Tuple[str, Dict]:
    """单值 needle。"""
    code = f"CODE{needle_idx:03d}"
    needle_text = f" The secret code is {code}. "
    return needle_text, {"type": "niah_single", "needle": code, "answer": code,
                          "question": f"What is the secret code?"}


def make_niah_multivalue(context: str, needle_idx: int, pos: float, seed: int) -> Tuple[str, Dict]:
    """多值 needle：多个对象各有 code，问其中某个对象的。"""
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"]
    objects = [names[(needle_idx * 3 + i) % len(names)] for i in range(3)]
    codes = [f"V{needle_idx}{i}" for i in range(3)]
    needle_text = f" The secret codes are: {objects[0]} has {codes[0]}, {objects[1]} has {codes[1]}, {objects[2]} has {codes[2]}. "
    target_obj = objects[1]
    target_code = codes[1]
    return needle_text, {"type": "niah_multivalue", "needle": target_code, "answer": target_code,
                          "question": f"What is the secret code of {target_obj}?"}


def make_niah_multikey(context: str, needle_idx: int, pos: float, seed: int) -> Tuple[str, Dict]:
    """多属性 needle：同一对象多个属性，问其中一个。"""
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"]
    obj = names[needle_idx % len(names)]
    code = f"K{needle_idx:03d}"
    city = ["Paris", "Rome", "Tokyo", "Cairo", "Berlin", "Madrid", "Oslo", "Lima"][needle_idx % 8]
    needle_text = f" {obj}'s secret code is {code}. {obj}'s favorite city is {city}. "
    return needle_text, {"type": "niah_multikey", "needle": code, "answer": code,
                          "question": f"What is {obj}'s secret code?"}


def make_niah_multihop(context: str, needle_idx: int, pos: float, seed: int) -> Tuple[str, Dict]:
    """多跳 needle：需要结合两条信息。"""
    cities = ["Paris", "Rome", "Tokyo", "Cairo", "Berlin", "Madrid", "Oslo", "Lima"]
    countries = {"Paris": "France", "Rome": "Italy", "Tokyo": "Japan", "Cairo": "Egypt",
                 "Berlin": "Germany", "Madrid": "Spain", "Oslo": "Norway", "Lima": "Peru"}
    city = cities[needle_idx % len(cities)]
    country = countries[city]
    person = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"][needle_idx % 8]
    needle_text = f" {person} lives in {city}. "
    return needle_text, {"type": "niah_multihop", "needle": country, "answer": country,
                          "question": f"{person} lives in a city. What country is that city in?"}


NEEDLE_MAKERS = {
    "niah_single": make_niah_single,
    "niah_multivalue": make_niah_multivalue,
    "niah_multikey": make_niah_multikey,
    "niah_multihop": make_niah_multihop,
}


def insert_needles_mixed(context: str, num_needles: int, types: List[str], seed: int = 42) -> Tuple[str, List[Dict]]:
    """在真实文本中插入多种类型的 needle（在句子边界插入）。"""
    random.seed(seed)
    # 按句子切分真实文本，保持插入在句号后
    sentences = re.split(r'(?<=[.!?])\s+', context)
    if not sentences:
        sentences = context.split(". ")

    positions = sorted([(i + 1) / (num_needles + 1) for i in range(num_needles)])
    out = []
    needles = []
    prev_sent = 0

    for i, pos in enumerate(positions):
        sent_idx = int(pos * len(sentences))
        sent_idx = max(prev_sent, min(sent_idx, len(sentences) - 1))
        ntype = types[i % len(types)]
        maker = NEEDLE_MAKERS[ntype]
        needle_text, info = maker(context, i, pos, seed)
        info["position"] = round(pos, 3)
        out.append(" ".join(sentences[prev_sent:sent_idx]))
        out.append(needle_text)
        needles.append(info)
        prev_sent = sent_idx

    out.append(" ".join(sentences[prev_sent:]))
    return "".join(out), needles


# ---------------------------------------------------------------------------
# 压缩策略
# ---------------------------------------------------------------------------

def full(text: str, tok, budget: int) -> str:
    return text


def truncation(text: str, tok, budget: int) -> str:
    t = tok.encode(text, add_special_tokens=False)
    if len(t) <= budget:
        return text
    return tok.decode(t[-budget:], skip_special_tokens=True)


def project_topic(text: str, tok, budget: int) -> str:
    """话题保留 + 分级压缩（关键词匹配 needle 句子）。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    key = [s for s in sentences if any(kw in s for kw in ["code is", "lives in", "favorite city", "has V", "has K"])]
    other = [s for s in sentences if s not in key]

    result = list(key)
    total = sum(len(tok.encode(s, add_special_tokens=False)) for s in result)
    for i, s in enumerate(reversed(other)):
        turn = i + 1
        ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
        if not ct:
            continue
        nt = len(tok.encode(ct, add_special_tokens=False))
        if total + nt <= budget:
            result.insert(0, ct)
            total += nt
    return " ".join(result)


def attention_sink(text: str, tok, budget: int) -> str:
    """Attention Sink 布局：sink → 关键信息 → 压缩其他。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    key = [s for s in sentences if any(kw in s for kw in ["code is", "lives in", "favorite city", "has V", "has K"])]
    other = [s for s in sentences if s not in key]

    key_text = "\n\n".join(key)
    key_tok = len(tok.encode(key_text, add_special_tokens=False))
    remaining = budget - key_tok - 2

    compressed_other = []
    if remaining > 0:
        for i, s in enumerate(reversed(other)):
            turn = i + 1
            ct = s if turn <= 5 else (s[:100] if turn <= 20 else (s[:50] if turn <= 50 else ""))
            if not ct:
                continue
            nt = len(tok.encode(ct, add_special_tokens=False))
            if sum(len(tok.encode(x, add_special_tokens=False)) for x in compressed_other) + nt <= remaining:
                compressed_other.insert(0, ct)

    other_text = " ".join(compressed_other)
    return "\n\n" + key_text + "\n\n" + other_text if other_text else "\n\n" + key_text


STRATEGIES = {
    "full": ("Full Context", full),
    "truncation": ("Truncation", truncation),
    "topic": ("Project + Topic", project_topic),
    "attention_sink": ("Attention Sink", attention_sink),
}


def eval_strategy(model, tok, device, context, needles, name, fn, budget):
    processed = fn(context, tok, budget)
    ptok = len(tok.encode(processed, add_special_tokens=False))
    if ptok < budget and fn != full:
        filler = " ".join(["The weather in London is often rainy in winter."] * ((budget - ptok) // 10))
        processed = processed + "\n\n" + filler
        ptok = len(tok.encode(processed, add_special_tokens=False))

    ftok = len(tok.encode(context, add_special_tokens=False))
    ratio = round(1 - ptok / ftok, 3) if ftok > 0 else 0

    correct = 0
    results = []
    times = []

    for n in needles:
        q = n["question"]
        prompt = processed + "\n\n" + q
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        dt = time.time() - t0
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        times.append(dt)

        ans = n["answer"]
        is_correct = ans.lower() in resp.lower()
        if is_correct:
            correct += 1
        results.append({"type": n["type"], "position": n["position"], "correct": is_correct, "response": resp.strip()[:60]})

    return {
        "strategy": name,
        "actual_tokens": ptok,
        "compression_ratio": ratio,
        "recall": correct / len(needles),
        "avg_latency_s": round(sum(times) / len(times), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--book", default="pride_and_prejudice")
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--num-needles", type=int, default=8)
    parser.add_argument("--types", nargs="+", default=["niah_single", "niah_multivalue", "niah_multikey", "niah_multihop"])
    parser.add_argument("--out", default="results/realtext_ablation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Book: {args.book}")
    book_text = load_real_text(args.book)
    context = prepare_real_context(book_text, args.context_tokens, tok)
    context_with_needles, needles = insert_needles_mixed(context, args.num_needles, args.types)
    ftok = len(tok.encode(context_with_needles, add_special_tokens=False))
    print(f"Real text context: {ftok} tokens, {args.num_needles} needles")

    # 按类型统计
    type_counts = {}
    for n in needles:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    print(f"Needle types: {type_counts}\n")

    results = {"config": {"model": args.model, "book": args.book, "context_tokens": ftok, "budgets": args.budgets, "num_needles": args.num_needles, "types": args.types}, "strategies": []}

    r = eval_strategy(model, tok, device, context_with_needles, needles, "Full Context", full, ftok)
    print(f"Full Context: recall={r['recall']:.0%}  latency={r['avg_latency_s']:.2f}s")
    results["strategies"].append(r)

    for budget in args.budgets:
        ratio = 1 - budget / ftok
        print(f"\nBudget: {budget} tokens ({ratio:.0%} compression)")
        for key, (label, fn) in STRATEGIES.items():
            if key == "full":
                continue
            r = eval_strategy(model, tok, device, context_with_needles, needles, label, fn, budget)
            marker = "✓" if r["recall"] >= 0.5 else "✗"
            print(f"  {marker} {label:<22} recall={r['recall']:.0%}  latency={r['avg_latency_s']:.2f}s  tok={r['actual_tokens']}")
            results["strategies"].append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
