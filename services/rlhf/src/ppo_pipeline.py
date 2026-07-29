"""Full RLHF pipeline: collect rollouts → PPO update → evaluate."""

import os
from typing import List, Dict, Any, Optional

from loguru import logger

from config import RLHFConfig
from ppo_trainer import PPOTrainer


class RLHFPipeline:
    """End-to-end RLHF fine-tuning pipeline.

    Pipeline:
        1. Generate responses (rollouts) using current policy
        2. Score responses using reward model
        3. Compute advantages using GAE
        4. Update policy using PPO-Clip
        5. Repeat
    """

    def __init__(self, config: RLHFConfig):
        self.config = config
        self._policy = None
        self._reward = None
        self._tokenizer = None
        self._trainer = None
        self._iteration = 0

    def _init_models(self):
        """Initialize models. Override in production."""
        pass

    def set_models(self, policy, reward, tokenizer):
        """Set models externally."""
        self._policy = policy
        self._reward = reward
        self._tokenizer = tokenizer
        self._trainer = PPOTrainer(policy, reward, tokenizer)

    def collect_rollouts(self, prompts: List[str]) -> List[Dict]:
        """Collect rollouts for a batch of prompts.

        Args:
            prompts: List of prompt strings.

        Returns:
            List of rollout dicts.
        """
        if self._trainer is None:
            raise ValueError("Models not set. Call set_models() first.")
        rollouts = []
        for prompt in prompts:
            rollout = self._trainer.generate_rollout(prompt)
            rollouts.append(rollout)
        logger.info(f"Collected {len(rollouts)} rollouts")
        return rollouts

    def train_iteration(self, prompts: List[str]) -> Dict[str, float]:
        """Run one training iteration.

        Args:
            prompts: Batch of training prompts.

        Returns:
            Training metrics.
        """
        if self._trainer is None:
            raise ValueError("Models not set.")
        metrics = self._trainer.train_step(prompts)
        self._iteration += 1
        logger.info(f"Iteration {self._iteration}: {metrics}")
        return metrics

    def train(
        self,
        dataset: List[Dict],
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Run full PPO training.

        Args:
            dataset: List of {prompt: ...} dicts.
            num_iterations: Number of PPO iterations.

        Returns:
            Final training results.
        """
        if self._trainer is None:
            self._init_models()

        results = {
            "iterations": num_iterations,
            "final_reward": 0.0,
            "reward_progress": [],
        }

        for i in range(num_iterations):
            # Sample batch from dataset
            batch = dataset[i % len(dataset):(i + 1) % len(dataset)] if len(dataset) > 1 else dataset
            prompts = [d.get("prompt", "") for d in batch]

            if not prompts:
                continue

            metrics = self.train_iteration(prompts)
            results["reward_progress"].append(metrics.get("avg_reward", 0.0))

            if i == num_iterations - 1:
                results["final_reward"] = metrics.get("avg_reward", 0.0)

        logger.info(f"Training completed. Final reward: {results['final_reward']}")
        return results

    def evaluate(self, eval_data: List[Dict]) -> Dict[str, float]:
        """Evaluate the current policy.

        Args:
            eval_data: List of {prompt, chosen, rejected} dicts.

        Returns:
            Evaluation metrics.
        """
        if self._trainer is None:
            raise ValueError("Models not set.")

        correct = 0
        total = 0

        for item in eval_data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            if not prompt:
                continue

            # Generate response
            response = self._policy.generate(
                [{"role": "user", "content": prompt}]
            ).chunk

            # Score with reward model
            score = self._reward.score(response)

            total += 1

        return {
            "num_eval": total,
        }

    def save_policy(self, path: str):
        """Save the policy model.

        Args:
            path: Output directory.
        """
        os.makedirs(path, exist_ok=True)
        if hasattr(self._policy, 'save_pretrained'):
            self._policy.save_pretrained(path)
        logger.info(f"Policy saved to {path}")
