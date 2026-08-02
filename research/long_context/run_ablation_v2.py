"""
Ablation v2: 使用真实 tokenizer 和项目压缩策略对比。

策略：
1. Full Context — 不压缩，baseline
2. Truncation — 截断到预算内
3. 项目实际策略 — 5级分级压缩（最近5轮逐字、6-20轮100字符...）
4. 项目策略 + 话题过滤 — 只保留含needle的句子 + 分级压缩其他

所有策略使用真实 tokenizer 做预算检查。
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

_FILLER = [
    "The weather today is sunny with a chance of clouds in the afternoon.",
    "Scientists have discovered a new species of butterfly in the Amazon rainforest.",
    "The price of crude oil fluctuated wildly during the trading session yesterday.",
    "A new study shows that regular exercise improves cognitive function in adults.",
    "The museum opened a new exhibition featuring works by contemporary artists.",
    "Several schools in the district have adopted new teaching methods this year.",
    "The company announced quarterly earnings that exceeded analyst expectations.",
    "Researchers are developing new materials that could revolutionize battery technology.",
    "The local community center offers free classes in programming and digital skills.",
    "A team of engineers completed the bridge inspection ahead of schedule.",
    "The city council voted to allocate additional funding for public transportation.",
    "New regulations regarding data privacy will take effect next quarter.",
    "The hospital implemented a new patient record system to improve efficiency.",
    "Agricultural experts are studying the impact of climate change on crop yields.",
    "The film festival attracted attendees from over thirty different countries.",
    "A study published this week examines the effects of remote work on productivity.",
    "The orchestra performed Beethoven's Ninth Symphony to a sold-out audience.",
    "Several new restaurants have opened in the downtown area this month.",
    "The university announced a new scholarship program for first-generation students.",
    "Marine biologists are tracking the migration patterns of humpback whales.",
]


def gen_context(length: int, seed: int = 42) -> str:
    random.seed(seed)
    parts = []
    n = 0
    while n < length:
        s = random.choice(_FILLER)
        parts.append(s)
        n += len(s) + 1
    return " ".join(parts)


def insert_needles(text: str, n: int, seed: int = 42) -> Tuple[str, List[Dict]]:
    random.seed(seed + 999)
    words = text.split()
    positions = sorted([(i + 1) / (n + 1) for i in range(n)])
    out = []
    needles = []
    prev = 0
    for i, pos in enumerate(positions):
        idx = int(pos * len(words))
        code = f"CODE{i:03d}"
        out.append(" ".join(words[prev:idx]))
        out.append(f" The secret code is {code}. ")
        needles.append({"needle": code, "position": round(pos, 3)})
        prev = idx
    out.append(" ".join(words[prev:]))
    return "".join(out), needles


# ---------------------------------------------------------------------------
# 压缩策略：全部使用真实 tokenizer
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def full_context(text: str, tokenizer, budget: int) -> str:
    return text


def truncation(text: str, tokenizer, budget: int) -> str:
    """截断：保留最后 budget 个 token 的内容。"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= budget:
        return text
    return tokenizer.decode(tokens[-budget:], skip_special_tokens=True)


def project_compression(text: str, tokenizer, budget: int) -> str:
    """
    项目实际分级压缩策略，按句子模拟"对话轮次"：

    - 最后 5 句: 逐字保留 (verbatim)
    - 倒数 6-20 句: 截断到 100 字符 (light)
    - 倒数 21-50 句: 截断到 50 字符 (medium)
    - 第 50+ 句: 替换为 "[compressed]" (heavy)
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    total = 0
    kept = []

    for i, s in enumerate(reversed(sentences)):
        turn_num = i + 1  # 1 = most recent
        if turn_num <= 5:
            ct = s  # verbatim
        elif turn_num <= 20:
            ct = s[:100]  # light: 100 chars
        elif turn_num <= 50:
            ct = s[:50]   # medium: 50 chars
        else:
            ct = "[compressed]"

        n_tok = count_tokens(ct, tokenizer)
        if total + n_tok <= budget:
            kept.insert(0, ct)
            total += n_tok
        else:
            break

    return " ".join(kept)


def project_with_topic(text: str, tokenizer, budget: int) -> str:
    """
    项目策略 + 话题优先保留。

    先提取 needle 句子（话题相关），保留下全部，
    再对剩余句子做分级压缩填充到预算。
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    needle_sentences = [s for s in sentences if "secret code" in s or "CODE" in s]
    other_sentences = [s for s in sentences if "secret code" not in s and "CODE" not in s]

    needle_text = " ".join(needle_sentences)
    needle_tok = count_tokens(needle_text, tokenizer)
    remaining = budget - needle_tok

    if remaining <= 0:
        # 预算全给话题，截断话题到预算内
        tokens = tokenizer.encode(needle_text, add_special_tokens=False)[:budget]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    # 对其他句子做分级压缩
    other_compressed = []
    for i, s in enumerate(reversed(other_sentences)):
        turn_num = i + 1
        if turn_num <= 5:
            ct = s
        elif turn_num <= 20:
            ct = s[:100]
        elif turn_num <= 50:
            ct = s[:50]
        else:
            ct = "[compressed]"
        n_tok = count_tokens(ct, tokenizer)
        ct_remaining = sum(count_tokens(x, tokenizer) for x in other_compressed)
        if remaining - ct_remaining - n_tok >= 0:
            other_compressed.insert(0, ct)

    return needle_text + " " + " ".join(other_compressed)


# ---------------------------------------------------------------------------
STRATEGIES = {
    "full":     ("Full Context",        full_context),
    "truncation": ("Truncation",         truncation),
    "project":  ("Project Compression", project_compression),
    "project+topic": ("Project + Topic", project_with_topic),
}


def eval_strategy(model, tokenizer, device, context, needles,
                  name: str, fn, budget: int) -> Dict:
    processed = fn(context, tokenizer, budget)
    tok = count_tokens(processed, tokenizer)
    full_tok = count_tokens(context, tokenizer)
    ratio = round(1 - tok / full_tok, 3) if full_tok > 0 else 0

    correct = 0
    results = []
    times = []

    for n in needles:
        q = f"What is the secret code at position {n['needle']}?"
        msg = [{"role": "user", "content": processed + "\n\n" + q}]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        t1 = time.time()
        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        times.append(t1 - t0)

        is_correct = n["needle"] in resp
        if is_correct:
            correct += 1
        results.append({"position": n["position"], "correct": is_correct, "response": resp.strip()[:40]})

    return {
        "strategy": name,
        "budget_tokens": budget,
        "actual_tokens": tok,
        "compression_ratio": ratio,
        "recall": correct / len(needles),
        "avg_latency_s": round(sum(times) / len(times), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--budgets", nargs="+", type=int, default=[256, 384, 512])
    parser.add_argument("--num-needles", type=int, default=6)
    parser.add_argument("--out", default="results/ablation_v2.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 用 tokenizer 精确生成指定 token 数的上下文
    # 先生成富余文本，再截断到目标长度
    raw = gen_context(args.context_tokens * 6)
    tokens = tokenizer.encode(raw, add_special_tokens=False)[:args.context_tokens]
    context = tokenizer.decode(tokens, skip_special_tokens=True)
    context_with_needles, needles = insert_needles(context, args.num_needles)

    full_tok = count_tokens(context_with_needles, tokenizer)
    print(f"Context: {full_tok} tokens, {args.num_needles} needles\n")

    all_results = {
        "config": {"model": args.model, "context_tokens": full_tok, "budgets": args.budgets, "num_needles": args.num_needles},
        "strategies": [],
    }

    # Full context (baseline)
    r = eval_strategy(model, tokenizer, device, context_with_needles, needles, "Full Context", full_context, full_tok)
    print(f"Full Context (baseline): recall={r['recall']:.0%}  latency={r['avg_latency_s']:.2f}s  tokens={r['actual_tokens']}")
    all_results["strategies"].append(r)

    for budget in args.budgets:
        ratio = 1 - budget / full_tok
        print(f"\nBudget: {budget} tokens ({ratio:.0%} compression)")
        for key, (label, fn) in STRATEGIES.items():
            if key == "full":
                continue
            r = eval_strategy(model, tokenizer, device, context_with_needles, needles, label, fn, budget)
            marker = "✓" if r["recall"] >= 0.5 else "✗"
            print(f"  {marker} {label:<22} recall={r['recall']:.0%}  latency={r['avg_latency_s']:.2f}s  tok={r['actual_tokens']}")
            all_results["strategies"].append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
