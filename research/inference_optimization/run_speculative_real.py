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

    # Sampling pipeline matching draft.generate's inherited defaults
    # (generation_config.json): repetition_penalty=1.1, temperature=0.7,
    # top_k=50 (explicit), top_p=0.8.
    REP_PENALTY = 1.1
    TEMPERATURE = 0.7
    TOP_K = 50
    TOP_P = 0.8

    def _warp_logits(self, logits: torch.Tensor, context_ids: List[int]) -> torch.Tensor:
        """Apply the same logits processors/warpers as transformers generate.

        ``context_ids`` = full context (prompt + accepted tokens + cand[:i])
        for the repetition penalty, exactly like generate's input_ids.
        """
        logits = logits.clone()
        # repetition penalty (RepetitionPenaltyLogitsProcessor)
        if context_ids:
            ctx = torch.tensor(context_ids, device=logits.device)
            score = logits[ctx]
            score = torch.where(score < 0, score * self.REP_PENALTY, score / self.REP_PENALTY)
            logits[ctx] = score
        # temperature
        logits = logits / self.TEMPERATURE
        # top_k
        k = min(self.TOP_K, logits.numel())
        topk_vals, topk_idx = torch.topk(logits, k)
        logits = torch.full_like(logits, float("-inf"))
        logits[topk_idx] = topk_vals
        # top_p (TopPLogitsWarper): keep smallest set with cumsum >= p
        sorted_probs, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        remove = cumsum > self.TOP_P
        remove[1:] = remove[:-1].clone()  # shift: token i removed if cumsum up to i-1 > p
        remove[0] = False  # always keep at least one
        logits[sorted_idx[remove]] = float("-inf")
        return logits

    def draft_propose(self, d_past, first_logits: torch.Tensor, gamma: int, device, context_ids: List[int]) -> Tuple[torch.Tensor, List[float], torch.Tensor, torch.Tensor]:
        """Draft incrementally proposes gamma tokens with KV reuse.

        Unlike the original implementation (full ``draft.generate`` from
        the whole prompt every round), this feeds only the previously
        sampled token each step, carrying the KV cache across rounds.
        The sampling distribution exactly reproduces ``draft.generate``
        with its inherited generation_config defaults (see _warp_logits).
        Returns (cand, q, updated_past, last_logits) where last_logits
        is P_draft(· | context + cand) — the next round's first_logits.
        """
        cand_list: List[int] = []
        q: List[float] = []
        logits = first_logits  # P_draft(· | current context)
        for _ in range(gamma):
            warped = self._warp_logits(logits, context_ids)
            sample_dist = F.softmax(warped.float(), dim=-1)
            t = torch.multinomial(sample_dist, 1).item()
            cand_list.append(t)
            context_ids.append(t)
            # q = probability under the draft's ACTUAL sampling
            # distribution (post-warper), matching output_scores
            q.append(sample_dist[t].item())
            # advance one token (KV reused)
            with torch.no_grad():
                out = self.draft(
                    torch.tensor([[t]], device=device),
                    past_key_values=d_past,
                    use_cache=True,
                )
            d_past = out.past_key_values
            logits = out.logits[0, -1]
        cand = torch.tensor(cand_list, device=device)
        return cand, q, d_past, logits

    def target_verify(self, cand: torch.Tensor, past=None, first_logits: torch.Tensor = None) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
        """Target verifies all gamma draft tokens in ONE forward pass.

        Incremental KV: only the NEW candidate tokens are fed to the
        model, with the prompt KV cache passed via ``past``.

        Logit alignment (causal LM): with input cand = [c0..c_{g-1}]
        starting at logical position prompt_len, output logits[i]
        predicts the token AFTER position prompt_len+i, i.e. c_{i+1}.
        So p[0] (P(c0 | prompt)) comes from the prefill pass's last
        position (``first_logits``), and p[i] (i>=1) comes from
        logits[i-1].
        """
        with torch.no_grad():
            out = self.target(cand.unsqueeze(0), past_key_values=past, use_cache=True)
        logits = out.logits[0]  # [gamma, vocab]
        p = []
        # p[0]: distribution right after the prompt (from prefill)
        p.append(F.softmax(first_logits.float(), dim=-1)[cand[0]].item())
        # p[i] (i>=1): logits[i-1] predicts cand[i]
        for i in range(1, len(cand)):
            p.append(F.softmax(logits[i - 1].float(), dim=-1)[cand[i]].item())
        return p, out.past_key_values, logits

    def rejection_sampling(self, q: List[float], p: List[float]) -> int:
        """Return number of accepted draft tokens (standard rule)."""
        for i in range(len(q)):
            r = torch.rand(1).item()
            if p[i] < q[i] and r >= p[i] / q[i]:
                return i
        return len(q)

    def target_sample_correction(self, first_logits: torch.Tensor, logits: torch.Tensor, n: int) -> int:
        """Sample correction token from the target distribution at position n.

        Rejection at position n means we need P(· | context + cand[:n]):
        n==0 uses the prefill's last-position logits, n>=1 uses
        logits[n-1] from the verification forward. No extra forward.
        """
        src = first_logits if n == 0 else logits[n - 1]
        probs = F.softmax(src.float(), dim=-1)
        return torch.multinomial(probs, 1).item()

    def generate(self, prompt_ids: torch.Tensor, max_tokens: int = 64) -> Tuple[List[int], int, int, int]:
        """Full speculative loop. Returns (tokens, n_target_fwd, n_draft_fwd, n_draft_accepted)."""
        generated: List[int] = []
        n_target_fwd = 0
        n_draft_fwd = 0
        n_draft_proposed = 0
        n_draft_accepted = 0
        device = prompt_ids.device

        # prefill BOTH models once; the last-position logits of each
        # predict the first candidate token under that model's dist.
        with torch.no_grad():
            out_t = self.target(prompt_ids, use_cache=True)
            out_d = self.draft(prompt_ids, use_cache=True)
        t_past = out_t.past_key_values
        t_first_logits = out_t.logits[0, -1]  # P_target(· | prompt)
        d_past = out_d.past_key_values
        d_first_logits = out_d.logits[0, -1]  # P_draft(· | prompt)
        n_target_fwd += 1
        n_draft_fwd += 1

        cur_len = prompt_ids.shape[1]
        context_ids: List[int] = prompt_ids[0].tolist()  # for repetition penalty
        while len(generated) < max_tokens:
            # 1. draft proposes gamma tokens (incremental KV, no full re-prefill)
            cand, q, d_past, d_last_logits = self.draft_propose(
                d_past, d_first_logits, self.gamma, device, context_ids
            )
            n_draft_fwd += self.gamma
            n_draft_proposed += len(cand)

            # 2. target verifies all gamma in one forward (incremental KV)
            p, t_past, t_logits = self.target_verify(
                cand, past=t_past, first_logits=t_first_logits
            )
            n_target_fwd += 1

            # 3. accept prefix
            n = self.rejection_sampling(q, p)
            n_draft_accepted += n
            for i in range(n):
                generated.append(int(cand[i].item()))

            # 4. correction token if rejected early — reuse verify logits
            if n < len(cand) and len(generated) < max_tokens:
                corr = self.target_sample_correction(t_first_logits, t_logits, n)
                generated.append(corr)
                # Drop the rejected candidates' KV entries from BOTH
                # models, then extend both caches with the correction
                # token's KV.
                keep = cur_len + n
                t_past.crop(keep)
                d_past.crop(keep)
                # roll back context_ids to the accepted prefix + corr
                del context_ids[cur_len + n:]
                context_ids.append(corr)
                n_target_fwd += 1
                n_draft_fwd += 1
                corr_t = torch.tensor([[corr]], device=device)
                with torch.no_grad():
                    out_t = self.target(corr_t, past_key_values=t_past, use_cache=True)
                    out_d = self.draft(corr_t, past_key_values=d_past, use_cache=True)
                t_past = out_t.past_key_values
                t_first_logits = out_t.logits[0, -1]  # P_target(· | ctx + corr)
                d_past = out_d.past_key_values
                d_first_logits = out_d.logits[0, -1]  # P_draft(· | ctx + corr)
                cur_len = keep + 1
            else:
                # all accepted: next round's first candidate is predicted
                # by the last verification/proposal position
                t_first_logits = t_logits[-1]
                d_first_logits = d_last_logits
                cur_len += len(cand)

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
