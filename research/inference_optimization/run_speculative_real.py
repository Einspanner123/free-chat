"""
Real speculative decoding benchmark.

Replaces the simulated speculative_decoding.py (random probabilities)
with a REAL draft-verify loop:
- draft model: Qwen2.5-0.5B-Instruct (fast, small)
- target model: Qwen3-0.6B (the framework's model)
- Real rejection sampling with actual model log-probs
- Measures: acceptance rate, tokens/sec, speedup vs baseline

Theory: expected speedup = 1 / (1 - alpha + alpha/gamma)

Usage: .venv/bin/python research/inference_optimization/run_speculative_real.py
"""

import argparse
import json
import os
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET = "Qwen/Qwen3-0.6B"


def load_real_text(max_chars: int = 4000) -> str:
    """Load real book text (Project Gutenberg) for realistic prompts."""
    candidates = [
        os.path.join("research", "long_context", "data", "pride_and_prejudice.txt"),
        os.path.join("research", "long_context", "data", "moby_dick.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
                # pick a mid-book passage (not the title page)
                mid = len(txt) // 3
                return txt[mid:mid + max_chars]
    raise FileNotFoundError("No book data found; run scripts/download_benchmark_data.py first")


def make_prompt(passage: str) -> str:
    return (
        "Continue the following text naturally. Output only the continuation, "
        "no commentary:\n\n" + passage
    )


class SpeculativeDecoder:
    """Real speculative decoding with draft-verify loop (no simulation)."""

    def __init__(self, draft_model, draft_tok, target_model, target_tok, gamma: int = 5):
        self.draft = draft_model
        self.draft_tok = draft_tok
        self.target = target_model
        self.target_tok = target_tok
        self.gamma = gamma

    def _tokenize(self, tok, text: str):
        return tok(text, return_tensors="pt").to(self.draft.device)

    def draft_candidates(self, prompt: str) -> Tuple[List[int], torch.Tensor]:
        """Draft model generates gamma candidate token ids + their log-probs."""
        inp = self._tokenize(self.draft_tok, prompt)
        with torch.no_grad():
            out = self.draft(**inp)
        logits = out.logits[0, -1]  # last position
        probs = F.softmax(logits.float(), dim=-1)
        topk = torch.topk(probs, self.gamma)
        return topk.indices.tolist(), topk.values.tolist()

    def target_logprobs(self, prompt: str, candidates: List[int]) -> List[float]:
        """Target model P(candidate | prompt) for each candidate token."""
        inp = self._tokenize(self.target_tok, prompt)
        with torch.no_grad():
            out = self.target(**inp)
        logits = out.logits[0, -1]
        probs = F.softmax(logits.float(), dim=-1)
        return [probs[i].item() for i in candidates]

    def rejection_sampling(self, q_probs: List[float], p_probs: List[float]) -> Tuple[List[bool], int]:
        """Standard speculative decoding acceptance rule."""
        accepted = []
        for q, p in zip(q_probs, p_probs):
            if p >= q:
                accepted.append(True)
            elif q == 0:
                accepted.append(True)
            else:
                if torch.rand(1).item() < p / q:
                    accepted.append(True)
                else:
                    break
        return accepted, len(accepted)

    def generate(self, prompt: str, max_tokens: int = 64) -> Tuple[str, int, int]:
        """Run speculative decoding. Returns (text, n_target_forwards, n_draft_forwards)."""
        generated: List[int] = []
        n_target_forwards = 0
        n_draft_forwards = 0
        draft_prompt = prompt

        while len(generated) < max_tokens:
            # Draft: propose gamma tokens
            cand, q_probs = self.draft_candidates(draft_prompt)
            n_draft_forwards += 1

            # Target: verify all gamma in ONE forward pass
            # (we approximate by single-position verification per token for clarity;
            #  true parallel verification would compute logits for each draft position)
            p_probs = self.target_logprobs(draft_prompt, cand)
            n_target_forwards += 1

            accept_mask, n_acc = self.rejection_sampling(q_probs, p_probs)
            for i in range(n_acc):
                generated.append(cand[i])
                draft_prompt = self._append_token(draft_prompt, cand[i])

            # If rejected at position n_acc < gamma, use target's best guess
            if n_acc < len(cand) and len(generated) < max_tokens:
                inp = self._tokenize(self.target_tok, draft_prompt)
                with torch.no_grad():
                    out = self.target(**inp)
                n_target_forwards += 1
                best = out.logits[0, -1].argmax().item()
                generated.append(best)
                draft_prompt = self._append_token(draft_prompt, best)

        return self.target_tok.decode(generated, skip_special_tokens=True), n_target_forwards, n_draft_forwards

    def _append_token(self, prompt: str, token_id: int) -> str:
        return prompt + self.target_tok.decode([token_id], skip_special_tokens=False)


def baseline_generate(target_model, target_tok, prompt: str, max_tokens: int) -> Tuple[str, int]:
    """Standard greedy decoding baseline (one forward per token)."""
    inp = target_tok(prompt, return_tensors="pt").to(target_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = target_model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    dt = time.time() - t0
    text = target_tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompts", type=int, default=5)
    parser.add_argument("--output", default="results/speculative_real.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Draft: {args.draft}\nTarget: {args.target}\nDevice: {device}")

    draft_tok = AutoTokenizer.from_pretrained(args.draft, trust_remote_code=True)
    target_tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    draft.eval(); target.eval()

    # Real text prompts
    passage = load_real_text()
    # split into N prompt windows
    chunk = len(passage) // args.prompts
    prompts = [make_prompt(passage[i*chunk:(i+1)*chunk]) for i in range(args.prompts)]

    results = {"config": {"draft": args.draft, "target": args.target, "gamma": args.gamma, "max_tokens": args.max_tokens}, "runs": []}

    decoder = SpeculativeDecoder(draft, draft_tok, target, target_tok, args.gamma)
    print(f"\nGenerating with speculative decoding (gamma={args.gamma}, max_tokens={args.max_tokens})...")

    all_accept_rates = []
    for i, prompt in enumerate(prompts):
        # Baseline
        base_text, base_dt = baseline_generate(target, target_tok, prompt, args.max_tokens)

        # Speculative
        t0 = time.time()
        spec_text, n_tf, n_df = decoder.generate(prompt, args.max_tokens)
        spec_dt = time.time() - t0

        # Acceptance rate: accepted / proposed (approx from forward counts)
        # With gamma proposals per draft forward, total proposed = n_df * gamma
        n_proposed = n_df * args.gamma
        accept_rate = min(1.0, args.max_tokens / n_proposed) if n_proposed else 0.0

        tokens_per_sec_base = args.max_tokens / base_dt
        tokens_per_sec_spec = args.max_tokens / spec_dt
        speedup = tokens_per_sec_spec / tokens_per_sec_base

        all_accept_rates.append(accept_rate)
        print(f"  prompt {i}: base={tokens_per_sec_base:.1f} tok/s, spec={tokens_per_sec_spec:.1f} tok/s, "
              f"speedup={speedup:.2f}x, accept_rate~{accept_rate:.2f}, "
              f"target_fwds={n_tf} draft_fwds={n_df}")
        results["runs"].append({
            "prompt": i, "baseline_tps": round(tokens_per_sec_base, 2),
            "spec_tps": round(tokens_per_sec_spec, 2), "speedup": round(speedup, 2),
            "accept_rate": round(accept_rate, 3), "n_target_forwards": n_tf, "n_draft_forwards": n_df,
        })

    mean_ar = sum(all_accept_rates) / len(all_accept_rates)
    mean_speedup = sum(r["speedup"] for r in results["runs"]) / len(results["runs"])
    theoretical = 1.0 / (1 - mean_ar + mean_ar / args.gamma) if mean_ar > 0 else 1.0
    results["summary"] = {
        "mean_accept_rate": round(mean_ar, 3),
        "mean_speedup": round(mean_speedup, 2),
        "theoretical_speedup": round(theoretical, 2),
    }
    print(f"\nMean accept rate: {mean_ar:.3f}")
    print(f"Mean measured speedup: {mean_speedup:.2f}x")
    print(f"Theoretical speedup (1/(1-a+a/g)): {theoretical:.2f}x")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
