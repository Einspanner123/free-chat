"""
NTK rope_scaling extension test (YaRN-style position extension).

Qwen3-0.6B trains to max_position_embeddings=40960 with rope_theta=1e6.
Beyond 40K positions is the EXTRAPOLATION zone where attention degrades.
NTK-aware scaling extends the effective window without fine-tuning.

Experiment:
- Task: long-context passage retrieval (needle in the haystack style)
- Context: real book text, padded to positions beyond 40K
- Baseline: default rope (no scaling) — expect degradation past 40K
- NTK: rope_scaling ntk with factor 2 (extends ~80K)
- Metric: retrieval accuracy at 10K / 30K / 45K / 60K positions

Usage: .venv/bin/python research/inference_optimization/run_ntk_extension.py
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
BOOK = os.path.join("research", "long_context", "data", "moby_dick.txt")


def load_book():
    with open(BOOK, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_needle_sample(text: str, needle_pos: int, target_len_tokens: int) -> Dict:
    """Insert a needle paragraph at a given position in a long context."""
    needle = "The magic number is 472913. The treasure is buried at this number."
    # tokenize approximations: use chars as proxy (roughly 3.5 chars/token)
    chars_per_token = 3.5
    total_chars = int(target_len_tokens * chars_per_token)
    needle_char_pos = int(needle_pos * chars_per_token)

    # build context: [before][needle][after] from book text
    before = text[needle_char_pos - len(needle) // 2: needle_char_pos] if needle_char_pos > 0 else ""
    after_avail = max(0, total_chars - needle_char_pos - len(needle))
    after = text[needle_char_pos: needle_char_pos + after_avail]
    context = before + needle + after

    question = "What is the magic number mentioned in the text?"
    answer = "472913"
    return {"context": context, "question": question, "answer": answer, "needle_pos": needle_pos}


def evaluate(model, tokenizer, device, samples, rope_mode: str) -> Dict:
    correct = 0
    results = []
    for s in samples:
        msgs = [{"role": "user", "content": f"Context:\n{s['context']}\n\n{s['question']}"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        n_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = s["answer"] in resp
        correct += int(ok)
        results.append({"needle_pos": s["needle_pos"], "tokens": n_tokens, "resp": resp[:60], "correct": ok})
    return {"mode": rope_mode, "accuracy": correct / len(samples), "correct": correct, "total": len(samples), "per_sample": results}


def load_model(rope_mode: str, ntk_factor: float):
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if rope_mode == "ntk":
        # Qwen3 supports 'yarn' (YaRN) — the proper long-context extension
        config.rope_parameters = {
            "rope_type": "yarn",
            "factor": ntk_factor,
            "original_max_position_embeddings": config.max_position_embeddings,
            "rope_theta": 1000000.0,  # Qwen3 uses rope_theta=1e6
            "attention_factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--needle-positions", nargs="+", type=int, default=[10000, 30000, 45000, 60000])
    parser.add_argument("--samples-per-pos", type=int, default=3)
    parser.add_argument("--ntk-factor", type=float, default=2.0)
    parser.add_argument("--output", default="results/ntk_extension.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_book()
    print(f"Book loaded: {len(text)} chars")
    print(f"Needle positions (tokens): {args.needle_positions}")
    print(f"NTK factor: {args.ntk_factor}\n")

    results = {"config": {"model": args.model, "needle_positions": args.needle_positions, "ntk_factor": args.ntk_factor}, "runs": []}

    for mode in ["default", "ntk"]:
        print(f"=== Mode: {mode} ===")
        model, tokenizer = load_model(mode, args.ntk_factor)
        model.to(device); model.eval()
        print(f"  rope_type: {model.config.rope_parameters.get('rope_type')}")

        for pos in args.needle_positions:
            samples = [build_needle_sample(text, pos, pos + 5000) for _ in range(args.samples_per_pos)]
            r = evaluate(model, tokenizer, device, samples, f"{mode}_{pos}")
            print(f"  needle@{pos:>6} tokens: acc={r['accuracy']:.0%} ({r['correct']}/{r['total']})")
            results["runs"].append(r)
        del model
        torch.cuda.empty_cache()
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
