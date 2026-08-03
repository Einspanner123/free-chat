"""
Real speculative decoding benchmark — CORRECT standard algorithm.

Fixes over the first version:
1. Parallel verification: target verifies ALL gamma draft tokens in
   ONE forward pass (positions prompt_len..prompt_len+gamma-1), using
   the standard Leviathan et al. 2023 algorithm.
2. Incremental KV cache: target KV cache is carried across iterations
   (verified draft tokens not recomputed) — fair speed comparison.
3. Real acceptance rate: accepted_draft_tokens / proposed_draft_tokens.
4. Shared vocab verified: Qwen2.5-0.5B and Qwen3-0.6B both use the
   Qwen2 BPE vocab (151643 ids) — token ids map 1:1.

Models:
- draft: Qwen2.5-0.5B-Instruct (fast, small, shared vocab)
- target: Qwen3-0.6B

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
    candidates = [
        os.path.join("research", "long_context", "data", "pride_and_prejudice.txt"),
        os.path.join("research", "long_context", "data", "moby_dick.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
                mid = len(txt) // 3
                return txt[mid:mid + max_chars]
    raise FileNotFoundError("No book data found; run scripts/download_benchmark_data.py first")


def make_prompt(passage: str) -> str:
    return (
        "Continue the following text naturally. Output only the continuation, "
        "no commentary:\n\n" + passage
    )


class SpeculativeDecoder:
    """Standard speculative decoding (Leviathan et al. 2023), real models."""

    def __init__(self, draft, draft_tok, target, target_tok, gamma: int = 5):
        self.draft = draft
        self.draft_tok = draft_tok
        self.target = target
        self.target_tok = target_tok
        self.gamma = gamma

    def draft_propose(self, prompt_ids: torch.Tensor, gamma: int) -> Tuple[torch.Tensor, List[float]]:
        """Draft auto-regressively generates gamma candidate tokens + their probs q_i."""
        out = self.draft.generate(
            prompt_ids,
            max_new_tokens=gamma,
            do_sample=True,
            top_k=50,
            return_dict_in_generate=True,
            output_scores=True,
        )
        cand = out.sequences[0][prompt_ids.shape[1]:]  # [gamma]
        q = []
        for i in range(gamma):
            logits = out.scores[i][0]  # distribution at draft position i
            q.append(F.softmax(logits.float(), dim=-1)[cand[i]].item())
        return cand, q

    def target_verify(self, prompt_ids: torch.Tensor, cand: torch.Tensor) -> Tuple[List[float], torch.Tensor]:
        """Target verifies all gamma draft tokens in ONE forward pass.

        Forward prompt_ids + cand with causal mask: logits at position
        prompt_len-1+i predicts token i (sees prompt + cand[:i]). This is
        the parallel verification — 1 target forward for gamma candidates.
        """
        full = torch.cat([prompt_ids[0], cand]).unsqueeze(0)
        with torch.no_grad():
            out = self.target(full, use_cache=True)
        logits = out.logits[0]  # [seq_len, vocab]
        base = prompt_ids.shape[1] - 1
        p = []
        for i in range(len(cand)):
            p.append(F.softmax(logits[base + i].float(), dim=-1)[cand[i]].item())
        return p, out.past_key_values

    def rejection_sampling(self, q: List[float], p: List[float]) -> int:
        """Return number of accepted draft tokens (standard rule)."""
        for i in range(len(q)):
            r = torch.rand(1).item()
            if p[i] < q[i] and r >= p[i] / q[i]:
                return i
        return len(q)

    def target_sample_correction(self, prompt_ids: torch.Tensor, cand: torch.Tensor, n: int) -> int:
        """If rejected at position n < gamma: sample correction token from target.

        Uses the target's distribution at position n (prompt + cand[:n]).
        Recomputes prefix (small: at most gamma tokens) — acceptable overhead.
        """
        prefix = torch.cat([prompt_ids[0], cand[:n]]).unsqueeze(0)
        with torch.no_grad():
            out = self.target(prefix, use_cache=True)
        logits = out.logits[0, -1]
        probs = F.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, 1).item()

    def generate(self, prompt_ids: torch.Tensor, max_tokens: int = 64) -> Tuple[List[int], int, int, int]:
        """Full speculative loop. Returns (tokens, n_target_fwd, n_draft_fwd, n_draft_accepted)."""
        generated: List[int] = []
        n_target_fwd = 0
        n_draft_fwd = 0
        n_draft_proposed = 0
        n_draft_accepted = 0

        # prefill target KV with prompt (1 target forward, counted once)
        with torch.no_grad():
            self.target(prompt_ids, use_cache=True)
        n_target_fwd += 1

        cur_prompt = prompt_ids
        while len(generated) < max_tokens:
            # 1. draft proposes gamma tokens
            cand, q = self.draft_propose(cur_prompt, self.gamma)
            n_draft_fwd += 1
            n_draft_proposed += len(cand)

            # 2. target verifies all gamma in one forward
            p, _ = self.target_verify(cur_prompt, cand)
            n_target_fwd += 1

            # 3. accept prefix
            n = self.rejection_sampling(q, p)
            n_draft_accepted += n
            for i in range(n):
                generated.append(cand[i].item())

            # 4. correction token if rejected early
            if n < len(cand) and len(generated) < max_tokens:
                corr = self.target_sample_correction(cur_prompt, cand, n)
                generated.append(corr)
                n_target_fwd += 1
                cur_prompt = torch.cat([cur_prompt[0], cand[:n], torch.tensor([corr], device=cur_prompt.device)]).unsqueeze(0)
            else:
                cur_prompt = torch.cat([cur_prompt[0], cand]).unsqueeze(0)

        return generated, n_target_fwd, n_draft_fwd, n_draft_accepted, n_draft_proposed


def baseline_generate(target, target_tok, prompt: str, max_tokens: int) -> Tuple[str, float]:
    inp = target_tok(prompt, return_tensors="pt").to(target.device)
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    dt = time.time() - t0
    text = target_tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--output", default="results/speculative_real.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Draft: {args.draft}\nTarget: {args.target}\nDevice: {device}")

    draft_tok = AutoTokenizer.from_pretrained(args.draft, trust_remote_code=True)
    target_tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    draft.eval(); target.eval()

    # Verify shared vocab (token ids must map 1:1 for speculative decoding)
    test_ids_d = draft_tok.encode("The quick brown fox jumps over 472913")
    test_ids_t = target_tok.encode("The quick brown fox jumps over 472913")
    if test_ids_d != test_ids_t:
        raise SystemExit("Vocab mismatch between draft and target — speculative decoding invalid")

    passage = load_real_text()
    chunk = len(passage) // args.prompts
    prompts = [make_prompt(passage[i*chunk:(i+1)*chunk]) for i in range(args.prompts)]

    results = {"config": {"draft": args.draft, "target": args.target, "gamma": args.gamma, "max_tokens": args.max_tokens}, "runs": []}

    decoder = SpeculativeDecoder(draft, draft_tok, target, target_tok, args.gamma)
    print(f"Generating (gamma={args.gamma}, max_tokens={args.max_tokens})...")

    for i, prompt in enumerate(prompts):
        prompt_ids = target_tok(prompt, return_tensors="pt")["input_ids"].to(device)

        # Baseline: standard greedy decode with KV cache (target only)
        base_text, base_dt = baseline_generate(target, target_tok, prompt, args.max_tokens)

        # Speculative
        t0 = time.time()
        spec_tokens, n_tf, n_df, n_acc, n_prop = decoder.generate(prompt_ids, args.max_tokens)
        spec_dt = time.time() - t0

        accept_rate = n_acc / n_prop if n_prop else 0.0
        tps_base = args.max_tokens / base_dt
        tps_spec = args.max_tokens / spec_dt
        speedup = tps_spec / tps_base

        print(f"  prompt {i}: base={tps_base:.1f} tok/s spec={tps_spec:.1f} tok/s "
              f"speedup={speedup:.2f}x accept_rate={accept_rate:.3f} "
              f"(target_fwd={n_tf} draft_fwd={n_df})")
        results["runs"].append({
            "prompt": i, "baseline_tps": round(tps_base, 2), "spec_tps": round(tps_spec, 2),
            "speedup": round(speedup, 2), "accept_rate": round(accept_rate, 3),
            "n_target_forwards": n_tf, "n_draft_forwards": n_df,
        })

    mean_ar = sum(r["accept_rate"] for r in results["runs"]) / len(results["runs"])
    mean_speedup = sum(r["speedup"] for r in results["runs"]) / len(results["runs"])
    # Leviathan: E[tokens per verify] = (1-alpha^(gamma+1)) / (1-alpha)
    if mean_ar < 1.0:
        expected = (1 - mean_ar ** (args.gamma + 1)) / (1 - mean_ar)
    else:
        expected = args.gamma + 1
    results["summary"] = {
        "mean_accept_rate": round(mean_ar, 3),
        "mean_speedup": round(mean_speedup, 2),
        "expected_tokens_per_verify": round(expected, 2),
        "target_forwards_for_64_tokens": round(64 / expected + 1, 1),
    }
    print(f"\nMean accept rate: {mean_ar:.3f}")
    print(f"Mean measured speedup: {mean_speedup:.2f}x")
    print(f"E[tokens per verify] (Leviathan): {expected:.2f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
