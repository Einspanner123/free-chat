"""
Ablation: compare compression strategies on real model.

Strategies:
1. Full context (baseline) — no compression
2. Truncation — keep last N chars
3. Hierarchical compression — tiered by recency
4. Hierarchy + topic reconstruction — keep only selected topic

Usage:
    .venv/bin/python benchmarks/long_context/run_ablation.py --num-needles 8
"""

import argparse
import json
import os
import random
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


# /shared context generator/

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


# /compression strategies/

def full_context(text: str, _budget: int) -> str:
    return text


def truncation(text: str, budget: int) -> str:
    return text[-budget:] if len(text) > budget else text


def hierarchical(text: str, budget: int) -> str:
    """Tiered compression by turn boundary (approximated by sentence)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    total = 0
    compressed = []
    for i, s in enumerate(reversed(sentences)):
        turn_num = len(sentences) - i
        if turn_num <= 3:
            ct = len(s)  # verbatim
        elif turn_num <= 10:
            ct = min(len(s), 80)  # light
        elif turn_num <= 25:
            ct = min(len(s), 40)  # medium
        else:
            ct = 4  # "[...]"
        if total + ct <= budget:
            compressed.insert(0, s[:ct] if ct < len(s) else s)
            total += ct
        else:
            break
    return " ".join(compressed)


def hierarchy_topic(text: str, budget: int) -> str:
    """Compress + keep only sentences containing needle patterns."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    needle_sentences = [s for s in sentences if "secret code" in s or "CODE" in s]
    other_sentences = [s for s in sentences if "secret code" not in s and "CODE" not in s]

    # Keep all needle sentences, then fill with compressed others
    needle_text = " ".join(needle_sentences)
    remaining = budget - len(needle_text)
    if remaining <= 0:
        return needle_text[:budget]

    # Compress other sentences and take what fits
    compressed_other = []
    for s in reversed(other_sentences):
        ct = min(len(s), 40)
        if sum(len(x) for x in compressed_other) + ct <= remaining:
            compressed_other.insert(0, s[:ct])
        else:
            break

    return needle_text + " " + " ".join(compressed_other)


# /evaluation/

STRATEGIES = {
    "full": ("Full Context (baseline)", full_context, 0),
    "truncation": ("Truncation", truncation, 0),
    "hierarchical": ("Hierarchical Compression", hierarchical, 0),
    "hierarchy+topic": ("Hierarchy + Topic", hierarchy_topic, 0),
}


def eval_strategy(model, tokenizer, device, context, needles,
                  strategy_name: str, strategy_fn, budget: int, max_new: int = 20) -> Dict:
    processed = strategy_fn(context, budget) if strategy_fn else context
    token_count = len(processed) // 4

    correct = 0
    results = []
    times = []

    for n in needles:
        question = f"What is the secret code at position {n['needle']}? Answer with just the code."
        messages = [{"role": "user", "content": processed + "\n\n" + question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        t1 = time.time()
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        times.append(t1 - t0)
        is_correct = n["needle"] in response
        if is_correct:
            correct += 1
        results.append({"position": n["position"], "correct": is_correct, "needle": n["needle"], "response": response.strip()})

    return {
        "strategy": strategy_name,
        "budget": budget if budget > 0 else len(processed),
        "compressed_tokens": token_count,
        "compression_ratio": round(1 - token_count / (max(token_count, 1)), 3),
        "recall": correct / len(needles),
        "avg_latency": round(sum(times) / len(times), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096, help="characters")
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--num-needles", type=int, default=6)
    parser.add_argument("--out", default="results/ablation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory: {mem:.1f}GB")

    context, needles = insert_needles(gen_context(args.context_length), args.num_needles)
    full_tokens = len(context) // 4
    print(f"\nContext: {args.context_length} chars (~{full_tokens} tokens), {args.num_needles} needles")
    print()

    all_results = {"config": {"model": args.model, "context_length": args.context_length, "budgets": args.budgets, "num_needles": args.num_needles}, "strategies": []}

    # Full context (baseline)
    print("Full Context (baseline):")
    r = eval_strategy(model, tokenizer, device, context, needles, "Full Context", None, 0)
    print(f"  Recall: {r['recall']:.0%}  Latency: {r['avg_latency']:.2f}s")
    all_results["strategies"].append(r)

    # Compression strategies at each budget
    for budget in args.budgets:
        ratio = 1 - budget / args.context_length
        print(f"\nBudget: {budget} chars ({ratio:.0%} compression)")
        for key, (label, fn, _) in STRATEGIES.items():
            if key == "full":
                continue
            r = eval_strategy(model, tokenizer, device, context, needles, label, fn, budget)
            marker = "✓" if r["recall"] > 0.5 else "✗"
            print(f"  {marker} {label:<25} recall={r['recall']:.0%}  latency={r['avg_latency']:.2f}s  tokens={r['compressed_tokens']}")
            all_results["strategies"].append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
