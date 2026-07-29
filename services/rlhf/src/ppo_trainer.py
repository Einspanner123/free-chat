"""PPO (Proximal Policy Optimization) trainer for RLHF.

Implements the PPO-Clip algorithm for fine-tuning language models
with reinforcement learning from human feedback.

The PPO loss:
    L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
      + c1 * (V - R)² - c2 * KL(πθ || πref)

where ratio = πθ(a|s) / πθ_old(a|s), A is advantage, V is value,
R is return, ε is clip range, c1 is value coefficient, c2 is KL penalty.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger


class PPOTrainer:
    """PPO trainer for fine-tuning language models."""

    def __init__(self, policy_model, reward_model, tokenizer):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Advantage Estimation with GAE
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """Compute advantages using Generalized Advantage Estimation (GAE).

        GAE(γ, λ) = Σ(γλ)^t * δ_{t+1}
        where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)

        Args:
            rewards: List of rewards at each step.
            values: List of value function estimates.
            gamma: Discount factor.
            lam: GAE lambda parameter.

        Returns:
            (advantages, returns)
        """
        n = len(rewards)
        advantages = [0.0] * n
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0  # Assume 0 for terminal state
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        # Compute returns (target values)
        returns = [adv + val for adv, val in zip(advantages, values)]

        return advantages, returns

    # ------------------------------------------------------------------
    # KL Divergence
    # ------------------------------------------------------------------

    @staticmethod
    def compute_kl_divergence(
        log_probs_current: List[float],
        log_probs_ref: List[float],
    ) -> float:
        """Compute approximate KL divergence between current and reference policy.

        KL(πθ || πref) ≈ E[log πθ(a|s) - log πref(a|s)]

        Args:
            log_probs_current: Log probabilities from current policy.
            log_probs_ref: Log probabilities from reference policy.

        Returns:
            KL divergence estimate.
        """
        if not log_probs_current or not log_probs_ref:
            return 0.0
        diffs = [c - r for c, r in zip(log_probs_current, log_probs_ref)]
        return sum(diffs) / len(diffs)

    # ------------------------------------------------------------------
    # PPO Loss
    # ------------------------------------------------------------------

    def compute_ppo_loss(
        self,
        log_probs: List[float],
        old_log_probs: List[float],
        advantages: List[float],
        clip_range: float = 0.2,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute PPO-Clip loss.

        L = -E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

        where r(θ) = exp(log πθ - log πθ_old)

        Args:
            log_probs: Current policy log probabilities.
            old_log_probs: Old policy log probabilities.
            advantages: Advantage estimates.
            clip_range: PPO clip range ε.
            epsilon: Small constant for numerical stability.

        Returns:
            PPO loss value.
        """
        total_loss = 0.0
        n = min(len(log_probs), len(old_log_probs), len(advantages))

        for lp, old_lp, adv in zip(log_probs[:n], old_log_probs[:n], advantages[:n]):
            # Importance sampling ratio
            ratio = math.exp(lp - old_lp)

            # Clipped ratio
            clipped_ratio = max(min(ratio, 1.0 + clip_range), 1.0 - clip_range)

            # Surrogate loss
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv

            # Policy loss (negative for gradient ascent)
            policy_loss = -min(surr1, surr2)
            total_loss += policy_loss

        return total_loss / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # KL Penalty
    # ------------------------------------------------------------------

    @staticmethod
    def apply_kl_penalty(reward: float, kl: float, kl_penalty: float = 0.05) -> float:
        """Apply KL penalty to reward.

        penalized_reward = reward - kl_penalty * KL

        Args:
            reward: Raw reward from reward model.
            kl: KL divergence estimate.
            kl_penalty: KL penalty coefficient.

        Returns:
            KL-penalized reward.
        """
        return reward - kl_penalty * kl

    @staticmethod
    def adapt_kl_coefficient(
        current_coef: float,
        current_kl: float,
        target_kl: float,
    ) -> float:
        """Adapt KL penalty coefficient based on current KL.

        If KL is too high, increase penalty; if too low, decrease.

        Args:
            current_coef: Current KL penalty coefficient.
            current_kl: Current KL divergence.
            target_kl: Target KL divergence.

        Returns:
            Updated KL coefficient.
        """
        ratio = current_kl / target_kl if target_kl > 0 else 1.0
        if ratio > 1.5:
            return current_coef * 1.2
        elif ratio < 0.5:
            return current_coef * 0.8
        return current_coef

    # ------------------------------------------------------------------
    # Rollout Generation
    # ------------------------------------------------------------------

    def generate_rollout(self, prompt: str) -> Dict[str, Any]:
        """Generate a single rollout: response + reward.

        Args:
            prompt: Input prompt.

        Returns:
            Dict with prompt, response, reward, log_probs, values.
        """
        response = self.policy_model.generate(
            [{"role": "user", "content": prompt}]
        )

        response_text = response.chunk
        reward = self.reward_model.score(response_text)

        # Encode to get token-level log probs (simplified)
        tokens = self.tokenizer.encode(response_text) if hasattr(self.tokenizer, 'encode') else [0] * len(response_text.split())
        n_tokens = len(tokens)
        log_probs = [-math.log(n_tokens)] * n_tokens  # uniform approximation
        values = [reward * 0.5] * n_tokens  # simplified value estimate

        return {
            "prompt": prompt,
            "response": response_text,
            "reward": float(reward) if not isinstance(reward, (int, float)) else reward,
            "log_probs": log_probs,
            "values": values,
            "tokens": tokens,
            "n_tokens": n_tokens,
        }

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """Run a single PPO training step.

        Args:
            prompts: List of training prompts.

        Returns:
            Training metrics.
        """
        # Collect rollouts
        rollouts = [self.generate_rollout(p) for p in prompts]

        # Compute advantages
        all_rewards = [r["reward"] for r in rollouts]
        all_values = [v for r in rollouts for v in r["values"]]
        all_log_probs = [lp for r in rollouts for lp in r["log_probs"]]

        # Single-step advantages (simplified for non-sequential case)
        advantages = [r - 0.5 for r in all_rewards]  # simplified
        returns = [r for r in all_rewards]

        # Compute KL penalty
        # In production, this compares current vs. reference policy
        # Here we use a simplified approximation
        kl_div = 0.0

        # PPO loss
        ppo_loss = self.compute_ppo_loss(
            all_log_probs,
            all_log_probs,  # same as old in this simplified version
            advantages,
            clip_range=0.2,
        )

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        avg_advantage = sum(advantages) / len(advantages) if advantages else 0.0

        metrics = {
            "ppo_loss": ppo_loss,
            "avg_reward": avg_reward,
            "avg_advantage": avg_advantage,
            "kl_divergence": kl_div,
            "num_rollouts": len(rollouts),
        }

        return metrics
