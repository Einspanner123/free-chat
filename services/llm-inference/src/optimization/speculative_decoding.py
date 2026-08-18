"""Real speculative decoding (Leviathan et al. 2023) for HF models.

This is the service-layer home of the algorithm validated in
``research/inference_optimization/run_speculative_real.py``. It is a REAL
implementation: it loads a small draft model, proposes ``gamma`` candidate
tokens with incremental KV-cache reuse, verifies all of them against the
target model in ONE forward pass, accepts the longest matching prefix via
rejection sampling, and rolls the KV cache back with ``crop()`` on rejection.

Unlike the research benchmark (which compared against a greedy baseline),
this service port warps BOTH the draft and the target with the SAME logits
processors (repetition penalty -> temperature -> top_k -> top_p), gated
exactly like ``transformers``' ``generate()``. That makes the spec output
distribution-preserving: it matches the non-speculative ``do_sample=True``
path of the engine.

Correctness invariants
----------------------
1. ``q`` must be the draft's POST-WARPER probability (its actual sampling
   distribution), not the raw logits softmax.
2. ``p`` must be the target's POST-WARPER probability with the same warpers.
3. Repetition-penalty context is the FULL running sequence (prompt +
   accepted + candidates so far), passed explicitly because incremental
   decoding feeds only one new token to the model.
4. Logit alignment (the classic off-by-one): with input ``cand`` starting at
   logical position ``cur_len``, ``logits[i]`` predicts the token AFTER
   position ``cur_len + i``. So ``p[0]`` comes from the prefill's last
   position (``t_first``) and ``p[i]`` (i>=1) from ``logits[i-1]``.
5. On rejection, BOTH models' KV caches are rolled back with ``crop()`` and
   ``context_ids`` is trimmed before the correction token is appended.
6. Draft and target must share a tokenizer vocab (ids map 1:1); validated at
   construction.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

from loguru import logger


def _as_list(x) -> List[int]:
    """Normalize an EOS token id to a list (accepts int / list / None)."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


@dataclass
class SpeculativeStats:
    """Forward-pass accounting for a speculative generation."""

    n_target_forwards: int = 0
    n_draft_forwards: int = 0
    n_draft_proposed: int = 0
    n_draft_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Accepted draft tokens / proposed draft tokens (real acceptance rate)."""
        if self.n_draft_proposed == 0:
            return 0.0
        return self.n_draft_accepted / self.n_draft_proposed

    def expected_tokens_per_verify(self, gamma: int) -> float:
        """Leviathan et al.: E[tokens per verify] = (1 - alpha^(gamma+1)) / (1 - alpha)."""
        ar = self.acceptance_rate
        if ar >= 1.0:
            return float(gamma + 1)
        return (1.0 - ar ** (gamma + 1)) / (1.0 - ar)


class SpeculativeDecoder:
    """Standard speculative decoding with a real draft-verify loop.

    Both ``draft_model`` and ``target_model`` must be HF ``PreTrainedModel``
    instances (the engine passes its loaded target as ``target_model``).
    Every call to ``generate``/``stream_tokens`` creates fresh KV state, so
    one decoder instance must not be shared across concurrent requests.
    """

    def __init__(
        self,
        *,
        draft_model,
        draft_tokenizer,
        target_model,
        target_tokenizer,
        gamma: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.05,
        device: Optional[torch.device] = None,
    ):
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.gamma = gamma
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.device = device or getattr(draft_model, "device", None)

        if self.gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {self.gamma}")
        if self.temperature == 0:
            raise ValueError(
                "Speculative decoding requires temperature > 0; "
                "greedy (temperature=0) sampling is unsupported in the spec loop"
            )
        self._validate_shared_vocab()

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _validate_shared_vocab(self):
        """Token ids must map 1:1 between draft and target (same BPE vocab).

        The hard requirement is probe-string id equality: real text must
        encode to the same ids under both tokenizers. ``len()`` equality is
        NOT enforced because sibling models (e.g. Qwen2.5 vs Qwen3) share the
        base BPE vocab but append a few model-specific added tokens to the
        tail, producing different ``len(tokenizer)`` values while all ids that
        matter map 1:1. A tail mismatch is logged as a warning only.
        """
        draft_vocab = len(self.draft_tokenizer)
        target_vocab = len(self.target_tokenizer)
        if draft_vocab != target_vocab:
            logger.warning(
                f"Speculative decoding: draft vocab={draft_vocab} != target "
                f"vocab={target_vocab}; continuing if probe-string ids match. "
                f"A shared base BPE vocab with a different added-token tail is "
                f"acceptable (e.g. Qwen2.5 draft for Qwen3 target)."
            )
        probe = "The quick brown fox jumps over 472913"
        draft_ids = self.draft_tokenizer.encode(probe)
        target_ids = self.target_tokenizer.encode(probe)
        if draft_ids != target_ids:
            raise ValueError(
                "Speculative decoding requires draft and target to share a "
                f"tokenizer vocab; got different ids for probe string: "
                f"draft={draft_ids}, target={target_ids}"
            )

    # ------------------------------------------------------------------
    # Logits warping (identical warpers for draft and target)
    # ------------------------------------------------------------------

    def _warp_logits(self, logits: torch.Tensor, context_ids: List[int]) -> torch.Tensor:
        """Apply repetition penalty -> temperature -> top_k -> top_p.

        The gating matches ``transformers``' ``_sample``: repetition penalty
        is skipped at 1.0, top_k at <=0, top_p at >=1.0. ``context_ids`` is the
        full running sequence used by the repetition penalty.
        """
        logits = logits.clone()

        # repetition penalty (RepetitionPenaltyLogitsProcessor)
        if self.repetition_penalty != 1.0 and context_ids:
            ctx = torch.tensor(context_ids, device=logits.device)
            score = logits[ctx]
            score = torch.where(
                score < 0,
                score * self.repetition_penalty,
                score / self.repetition_penalty,
            )
            logits[ctx] = score

        # temperature
        logits = logits / self.temperature

        # top_k
        if self.top_k > 0:
            k = max(1, min(self.top_k, logits.numel()))
            topk_vals, topk_idx = torch.topk(logits, k)
            logits = torch.full_like(logits, float("-inf"))
            logits[topk_idx] = topk_vals

        # top_p (TopPLogitsWarper): keep the smallest set with cumsum >= top_p
        if self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(
                F.softmax(logits, dim=-1), descending=True
            )
            cumsum = torch.cumsum(sorted_probs, dim=0)
            remove = cumsum > self.top_p
            # token i is removed if the cumulative sum up to i-1 already
            # exceeded top_p (shift by one); always keep at least one token
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_idx[remove]] = float("-inf")

        return logits

    # ------------------------------------------------------------------
    # Draft / verify / rejection primitives
    # ------------------------------------------------------------------

    def _draft_propose(
        self,
        d_past,
        first_logits: torch.Tensor,
        context_ids: List[int],
        cur_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[float], object, torch.Tensor]:
        """Draft incrementally proposes ``gamma`` tokens with KV reuse.

        Returns ``(cand, q, d_past, last_logits)`` where ``q`` holds the
        draft's POST-WARPER probability of each candidate (its actual sampling
        distribution) and ``last_logits`` is the next round's ``first_logits``.
        """
        cand_list: List[int] = []
        q: List[float] = []
        logits = first_logits
        for _ in range(self.gamma):
            ctx = context_ids[: cur_len + len(cand_list)]
            warped = self._warp_logits(logits, ctx)
            dist = F.softmax(warped.float(), dim=-1)
            t = torch.multinomial(dist, 1).item()
            cand_list.append(t)
            context_ids.append(t)
            q.append(dist[t].item())
            # advance one token (KV reused)
            with torch.no_grad():
                out = self.draft_model(
                    torch.tensor([[t]], device=device),
                    past_key_values=d_past,
                    use_cache=True,
                )
            d_past = out.past_key_values
            logits = out.logits[0, -1]
        cand = torch.tensor(cand_list, device=device)
        return cand, q, d_past, logits

    def _target_verify(
        self,
        cand: torch.Tensor,
        t_past,
        t_first: torch.Tensor,
        context_ids: List[int],
        cur_len: int,
    ) -> Tuple[List[float], object, torch.Tensor]:
        """Target verifies all ``gamma`` draft tokens in ONE forward pass.

        Logit alignment (causal LM): with ``cand = [c0..c_{g-1}]`` starting at
        logical position ``cur_len``, ``logits[i]`` predicts the token AFTER
        position ``cur_len + i``, i.e. ``c_{i+1}``. So ``p[0]`` (P(c0 | prompt))
        comes from the prefill's last position (``t_first``) and ``p[i]``
        (i>=1) from ``logits[i-1]``. Both are warped with the context that the
        model actually conditioned on.
        """
        with torch.no_grad():
            out = self.target_model(
                cand.unsqueeze(0), past_key_values=t_past, use_cache=True
            )
        logits = out.logits[0]  # [gamma, vocab]

        p: List[float] = []
        p.append(
            F.softmax(
                self._warp_logits(t_first, context_ids[:cur_len]).float(), dim=-1
            )[int(cand[0])].item()
        )
        for i in range(1, len(cand)):
            p.append(
                F.softmax(
                    self._warp_logits(logits[i - 1], context_ids[: cur_len + i]).float(),
                    dim=-1,
                )[int(cand[i])].item()
            )
        return p, out.past_key_values, logits

    @staticmethod
    def rejection_sampling(q: Sequence[float], p: Sequence[float], rng=None) -> int:
        """Return the number of accepted draft tokens (longest prefix).

        Standard rule: at each position i, accept iff ``q[i] == 0`` (token was
        never proposed) or ``p[i] >= q[i]`` (target at least as confident), else
        accept with probability ``p[i] / q[i]``. Returns the index of the first
        rejection, or ``len(q)`` if all are accepted.

        ``rng`` is injectable for tests (default: one uniform sample per step).
        """
        if rng is None:
            rng = lambda: torch.rand(1).item()
        for i in range(len(q)):
            if q[i] == 0.0 or p[i] >= q[i]:
                continue  # accept
            if rng() >= p[i] / q[i]:
                return i  # first rejection
        return len(q)

    def _target_sample_correction(
        self,
        t_first: torch.Tensor,
        t_logits: torch.Tensor,
        n: int,
        context_ids: List[int],
        cur_len: int,
    ) -> int:
        """Sample the correction token from the target's warped distribution at n."""
        src = t_first if n == 0 else t_logits[n - 1]
        ctx = context_ids[: cur_len + n]
        warped = self._warp_logits(src, ctx)
        probs = F.softmax(warped.float(), dim=-1)
        return torch.multinomial(probs, 1).item()

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------

    def _rounds(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int,
        eos_token_id=None,
    ) -> Iterator[Tuple[List[int], SpeculativeStats]]:
        """Yield the new token ids produced per speculative round.

        KV state is local to this generator (fresh prefill on first iteration).
        Terminates on ``max_tokens`` or an EOS token (if given).
        """
        device = prompt_ids.device
        cur_len = prompt_ids.shape[1]
        context_ids: List[int] = prompt_ids[0].tolist()
        eos_ids = _as_list(eos_token_id)
        stats = SpeculativeStats()
        generated = 0

        # Prefill BOTH models once; each model's last-position logits predict
        # the first candidate token under that model's distribution.
        with torch.no_grad():
            out_t = self.target_model(prompt_ids, use_cache=True)
            out_d = self.draft_model(prompt_ids, use_cache=True)
        t_past = out_t.past_key_values
        t_first = out_t.logits[0, -1]  # P_target(· | prompt)
        d_past = out_d.past_key_values
        d_first = out_d.logits[0, -1]  # P_draft(· | prompt)
        stats.n_target_forwards += 1
        stats.n_draft_forwards += 1

        if not hasattr(t_past, "crop") or not hasattr(d_past, "crop"):
            raise NotImplementedError(
                "Speculative decoding requires KV caches with crop() for "
                "rollback on rejection; got "
                f"target={type(t_past).__name__}, draft={type(d_past).__name__}"
            )

        while generated < max_tokens:
            # 1. Draft proposes gamma candidates (incremental KV, no re-prefill)
            cand, q, d_past, d_last = self._draft_propose(
                d_past, d_first, context_ids, cur_len, device
            )
            stats.n_draft_forwards += self.gamma
            stats.n_draft_proposed += len(cand)

            # 2. Target verifies all gamma in ONE forward (incremental KV)
            p, t_past, t_logits = self._target_verify(
                cand, t_past, t_first, context_ids, cur_len
            )
            stats.n_target_forwards += 1

            # 3. Accept the longest prefix
            n = self.rejection_sampling(q, p)
            stats.n_draft_accepted += n

            new_ids: List[int] = []
            stop = False
            for i in range(n):
                if generated >= max_tokens:
                    stop = True
                    break
                tok = int(cand[i])
                new_ids.append(tok)
                generated += 1
                if eos_ids and tok in eos_ids:
                    stop = True
                    break

            if stop or generated >= max_tokens:
                yield new_ids, stats
                return

            if n < len(cand):
                # 4. Correction token from the target's warped distribution at n.
                #    Reuses prefill/verify logits — no extra forward.
                corr = self._target_sample_correction(
                    t_first, t_logits, n, context_ids, cur_len
                )
                new_ids.append(corr)
                generated += 1

                # 5. Drop the rejected candidates' KV from BOTH models, then
                #    extend both caches with the correction token's KV.
                keep = cur_len + n
                t_past.crop(keep)
                d_past.crop(keep)
                # roll back context_ids to the accepted prefix + corr
                del context_ids[cur_len + n:]
                context_ids.append(corr)
                corr_t = torch.tensor([[corr]], device=device)
                with torch.no_grad():
                    out_t = self.target_model(
                        corr_t, past_key_values=t_past, use_cache=True
                    )
                    out_d = self.draft_model(
                        corr_t, past_key_values=d_past, use_cache=True
                    )
                t_past = out_t.past_key_values
                t_first = out_t.logits[0, -1]  # P_target(· | ctx + corr)
                d_past = out_d.past_key_values
                d_first = out_d.logits[0, -1]  # P_draft(· | ctx + corr)
                stats.n_target_forwards += 1
                stats.n_draft_forwards += 1
                cur_len = keep + 1
            else:
                # all accepted: next round's first candidate is predicted by
                # the last verification/proposal position
                t_first = t_logits[-1]
                d_first = d_last
                cur_len += len(cand)

            yield new_ids, stats
            if eos_ids and new_ids and new_ids[-1] in eos_ids:
                return

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int,
        eos_token_id=None,
    ) -> Tuple[List[int], SpeculativeStats]:
        """Run the full speculative loop. Returns (token_ids, stats)."""
        generated: List[int] = []
        stats = SpeculativeStats()
        for new_ids, s in self._rounds(prompt_ids, max_tokens, eos_token_id):
            generated.extend(new_ids)
            stats = s
        return generated, stats

    def stream_tokens(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int,
        eos_token_id=None,
    ) -> Iterator[Tuple[List[int], SpeculativeStats]]:
        """Yield ``(new_ids, stats)`` per speculative round for streaming."""
        for new_ids, stats in self._rounds(prompt_ids, max_tokens, eos_token_id):
            yield new_ids, stats

    # ------------------------------------------------------------------
    # Expected speedup (Leviathan et al. 2023)
    # ------------------------------------------------------------------

    def estimate_speedup(self, acceptance_rate: float = 0.8) -> float:
        """Expected speedup: 1 / (1 - alpha + alpha/gamma)."""
        if self.gamma <= 1:
            return 1.0
        alpha = acceptance_rate
        gamma = self.gamma
        return 1.0 / (1.0 - alpha + alpha / gamma)
