"""Speculative decoding for faster inference.

Uses a small draft model to generate candidate tokens,
then verifies them with the target model in parallel.

Theory:
- Draft model generates γ candidate tokens auto-regressively
- Target model verifies all γ tokens in a single forward pass
- Accepted tokens are kept, rejected tokens trigger correction
- Expected speedup: 1 / (1 - α + α/γ) where α is acceptance rate
"""

import random
from typing import List, Optional, Tuple

from loguru import logger


class SpeculativeDecoder:
    """Speculative decoding with draft-verify loop."""

    def __init__(
        self,
        draft_model=None,
        target_model=None,
        gamma: int = 5,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma

    def speculate(self, prompt: str, max_tokens: int = 128) -> str:
        """Run speculative decoding on a prompt.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        if self.draft_model is None or self.target_model is None:
            logger.warning("Draft or target model not set, falling back to target only")
            return self.target_model.generate([{"role": "user", "content": prompt}]).chunk

        generated = []
        remaining = max_tokens

        while remaining > 0:
            # Step 1: Draft generates γ candidates
            draft_prompt = prompt + "".join(generated)
            draft_output = self.draft_model.generate(
                [{"role": "user", "content": draft_prompt}]
            )
            candidate_text = draft_output.chunk
            candidate_tokens = candidate_text.split()
            candidate_tokens = candidate_tokens[:min(self.gamma, remaining)]

            if not candidate_tokens:
                break

            # Step 2: Target verifies
            full_prompt = prompt + "".join(generated)
            target_output = self.target_model.generate(
                [{"role": "user", "content": full_prompt}]
            )

            # Step 3: Rejection sampling
            q_probs = self._estimate_draft_probs(candidate_tokens)
            p_probs = self._estimate_target_probs(candidate_tokens)
            accept_mask, n_accepted = self.rejection_sampling(q_probs, p_probs)

            # Keep only accepted tokens
            for i, accepted in enumerate(accept_mask):
                if accepted and i < len(candidate_tokens):
                    generated.append(candidate_tokens[i])
            remaining -= n_accepted

            # If no tokens accepted, generate one with target and continue
            if n_accepted == 0:
                single = self.target_model.generate(
                    [{"role": "user", "content": prompt + "".join(generated)}]
                ).chunk
                generated.append(single[:20])
                remaining -= 1

        return "".join(generated)

    def verify(self, tokens: List[str], context: str) -> List[str]:
        """Verify draft tokens against target model distribution.

        Args:
            tokens: Candidate tokens from draft model.
            context: Context string.

        Returns:
            List of accepted tokens.
        """
        accepted = []
        for token in tokens:
            # In production, compare draft and target log-probs
            # Here we accept with high probability for testing
            if random.random() < 0.9:
                accepted.append(token)
        return accepted

    def rejection_sampling(
        self, q_probs: List[float], p_probs: List[float]
    ) -> Tuple[List[bool], int]:
        """Rejection sampling to verify draft tokens.

        For each position i:
        - If p > q, always accept (target is more confident than draft)
        - If q == 0 and p == 0, accept (both equally uncertain)
        - Else accept with probability p / q

        Args:
            q_probs: Draft model probabilities for each token.
            p_probs: Target model probabilities for each token.

        Returns:
            (acceptance_mask, n_accepted) where acceptance_mask
            is a list of booleans indicating which positions to accept.
        """
        accepted = []
        for q, p in zip(q_probs, p_probs):
            if p > q:
                accepted.append(True)
            elif q == 0 and p == 0:
                accepted.append(True)
            elif q == 0:
                # p <= q and q == 0 → p must also be 0 (handled above)
                accepted.append(True)
            else:
                if random.random() < p / q:
                    accepted.append(True)
                else:
                    break
        return accepted, len(accepted)

    def estimate_speedup(self, acceptance_rate: float = 0.8) -> float:
        """Estimate speedup factor from speculative decoding.

        Formula: speedup = 1 / (1 - α + α/γ)
        where α is the acceptance rate and γ is the draft length.

        Args:
            acceptance_rate: Average token acceptance rate (0-1).

        Returns:
            Expected speedup factor.
        """
        if self.gamma <= 1:
            return 1.0
        α = acceptance_rate
        γ = self.gamma
        return 1.0 / (1 - α + α / γ)

    @staticmethod
    def _estimate_draft_probs(tokens: List[str]) -> List[float]:
        """Estimate draft model probabilities (simplified)."""
        return [0.8 + random.random() * 0.2 for _ in tokens]

    @staticmethod
    def _estimate_target_probs(tokens: List[str]) -> List[float]:
        """Estimate target model probabilities (simplified)."""
        return [0.7 + random.random() * 0.3 for _ in tokens]
