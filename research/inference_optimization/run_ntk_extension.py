"""
YaRN rope extension test (position extrapolation) — corrected version.

Fixes over v1:
1. Independent samples: each sample uses a DIFFERENT needle number and
   a different insertion offset — no repeated inputs (v1 repeated the
   same context 3x, making 3/3 meaningless).
2. Reports ACTUAL token positions (measured from tokenizer output),
   not char-estimated positions.
3. Standard YaRN params: attention_factor left to transformers'
   default computation (get_mscale(factor)) instead of forcing 1.0.

Qwen3-0.6B trains to max_position_embeddings=40960 with rope_theta=1e6.
Beyond 40K is the extrapolation zone. We test whether YaRN (factor=2)
changes retrieval accuracy vs default rope.

Usage: .venv/bin/python research/inference_optimization/run_ntk_extension.py
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
BOOK = os.path.join("research", "long_context", "data", "moby_dick.txt")


def load_book():
    with open(BOOK, encoding="utf-8", errors="ignore") as f:
        return f.read()


class NeedleSample:
    """One needle sample with a unique number and unique context window."""

    _counter = 0

    def __init__(self, text: str, target_needle_pos: int, total_tokens: int):
        NeedleSample._counter += 1
        self.number = 400000 + NeedleSample._counter * 7  # unique per sample
        self.needle = f"The magic number is {self.number}. The treasure is buried at this number."

        # Unique context window: start offset varies per sample (independent draws)
        chars_per_token = 3.5
        total_chars = int(total_tokens * chars_per_token)
        window_start = max(0, min(len(text) - total_chars, abs(hash(f"win{NeedleSample._counter}")) % max(1, len(text) - total_chars)))
        window = text[window_start: window_start + total_chars]

        needle_char_pos = int(target_needle_pos * chars_per_token)
        needle_char_pos = min(needle_char_pos, len(window) - len(self.needle) - 10)

        before = window[:needle_char_pos]
        after = window[needle_char_pos + len(self.needle):]
        self.context = before + self.needle + after

        self.question = "What is the magic number mentioned in the text?"
        self.answer = str(self.number)

    def build(self):
        return {"context": self.context, "question": self.question, "answer": self.answer}


def evaluate(model, tokenizer, device, samples, rope_mode: str, target_pos: int) -> Dict:
    correct = 0
    per_sample = []
    for s in samples:
        msgs = [{"role": "user", "content": f"Context:\n{s['context']}\n\n{s['question']}"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        actual_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = s["answer"] in resp
        correct += int(ok)
        per_sample.append({"answer": s["answer"], "actual_tokens": actual_tokens, "resp": resp[:50], "correct": ok})
    return {"mode": rope_mode, "target_needle_pos": target_pos, "accuracy": correct / len(samples), "correct": correct, "total": len(samples), "per_sample": per_sample}


def load_model(rope_mode: str, yarn_factor: float):
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if rope_mode == "yarn":
        # YaRN: attention_factor/mscale left to transformers' default
        # computation (get_mscale(factor)) — standard YaRN params
        config.rope_parameters = {
            "rope_type": "yarn",
            "factor": yarn_factor,
            "original_max_position_embeddings": config.max_position_embeddings,
            "rope_theta": 1000000.0,
        }
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--needle-positions", nargs="+", type=int, default=[10000, 30000, 45000, 60000, 80000])
    parser.add_argument("--samples-per-pos", type=int, default=4)
    parser.add_argument("--yarn-factor", type=float, default=2.0)
    parser.add_argument("--output", default="results/ntk_extension.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_book()
    print(f"Book: {len(text)} chars, target needle positions (tokens): {args.needle_positions}")
    print(f"YaRN factor: {args.yarn_factor}\n")

    results = {"config": {"model": args.model, "needle_positions": args.needle_positions, "yarn_factor": args.yarn_factor}, "runs": []}

    for mode in ["default", "yarn"]:
        print(f"=== Mode: {mode} ===")
        model, tokenizer = load_model(mode, args.yarn_factor)
        model.to(device); model.eval()
        print(f"  rope_type: {model.config.rope_parameters.get('rope_type')}")

        for pos in args.needle_positions:
            # Independent samples: unique numbers + unique windows (no repeats)
            samples = [NeedleSample(text, pos, pos + 5000).build() for _ in range(args.samples_per_pos)]
            r = evaluate(model, tokenizer, device, samples, f"{mode}_{pos}", pos)
            actual = [p["actual_tokens"] for p in r["per_sample"]]
            print(f"  needle@~{pos:>6}: acc={r['accuracy']:.0%} ({r['correct']}/{r['total']}) "
                  f"actual_tokens={actual}")
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
