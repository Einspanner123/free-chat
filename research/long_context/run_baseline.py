"""
Baseline: run Needle-in-a-Haystack on a real small model.

Downloads model, generates long context with inserted needles,
queries model at each position, reports recall by position.

Usage:
    .venv/bin/python benchmarks/long_context/run_baseline.py --model Qwen/Qwen2.5-0.5B-Instruct
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


_FILLER_SENTENCES = [
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


def generate_context(length: int, seed: int = 42) -> str:
    random.seed(seed)
    sentences = []
    total = 0
    while total < length:
        s = random.choice(_FILLER_SENTENCES)
        sentences.append(s)
        total += len(s) + 1
    return " ".join(sentences)


def insert_needles(context: str, n: int, seed: int = 42) -> Tuple[str, List[Dict]]:
    random.seed(seed + 999)
    words = context.split()
    positions = sorted([(i+1)/(n+1) for i in range(n)])
    result_parts = []
    needles = []
    prev = 0
    for i, pos in enumerate(positions):
        idx = int(pos * len(words))
        code = f"CODE{i:03d}"
        result_parts.append(" ".join(words[prev:idx]))
        result_parts.append(f" The secret code is {code}. ")
        needles.append({"needle": code, "position": round(pos, 3)})
        prev = idx
    result_parts.append(" ".join(words[prev:]))
    return "".join(result_parts), needles


def make_prompt(context: str, question: str) -> str:
    return f"""Context: {context}

Question: {question}

Answer the question based only on the context provided."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-needles", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.1f}s, {model.num_parameters()/1e6:.0f}M params")

    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after load: {mem:.1f}GB")

    print(f"\nGenerating {args.context_length}-char context with {args.num_needles} needles...")
    context, needles = insert_needles(
        generate_context(args.context_length), args.num_needles
    )
    print(f"Context length: {len(context)} chars, ~{len(context)//4} tokens")

    results = []
    total_time = 0
    correct = 0

    for n in needles:
        needle = n["needle"]
        prompt = make_prompt(context, f"What is the secret code at position {needle}?")
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=False,
            )
        t1 = time.time()
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        total_time += t1 - t0
        is_correct = needle in response
        if is_correct:
            correct += 1

        marker = "✓" if is_correct else "✗"
        print(f"  pos={n['position']:.2f} {marker} needle={needle} resp={response.strip()[:40]}")
        results.append({"position": n["position"], "correct": is_correct, "needle": needle, "response": response.strip()})

    recall = correct / max(len(needles), 1)
    avg_latency = total_time / max(len(needles), 1)

    print(f"\nResults:")
    print(f"  Overall recall: {recall:.1%} ({correct}/{len(needles)})")
    print(f"  Avg latency: {avg_latency:.2f}s per query")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    output = {
        "model": args.model,
        "context_length": args.context_length,
        "num_needles": args.num_needles,
        "recall": recall,
        "avg_latency_s": round(avg_latency, 3),
        "gpu_memory_gb": round(mem, 1),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
