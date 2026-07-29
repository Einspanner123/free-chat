"""
Reward Model for preference learning.

Implements a reward model that scores text pairs for DPO/RLHF.
Architecture: base LM + learned linear head that outputs a scalar reward.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from loguru import logger

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = object

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


@dataclass
class RewardModelConfig:
    """Configuration for the Reward Model."""
    base_model: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 4096
    dropout: float = 0.1
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4


class RewardModel:
    """Reward Model that scores text with a scalar value.

    Architecture:
        Base LM → [CLS] token hidden state → Linear head → scalar reward
    """

    def __init__(self, config: RewardModelConfig):
        self.config = config
        self._closed = False
        self.base_model = None
        self.tokenizer = None
        self.reward_head = None

        if AutoModel is None:
            raise ImportError("transformers is required for RewardModel")

        logger.info(f"Loading reward model base: {config.base_model}")

        # Load base model (without LM head)
        self.base_model = AutoModel.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch is not None else "auto",
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reward head: single linear layer
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

        logger.info("Reward model initialized")

    def score(self, text: str) -> float:
        """Score a single text.

        Args:
            text: Input text to score.

        Returns:
            Scalar reward value.
        """
        if self.base_model is None or self.tokenizer is None:
            return 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.base_model(**inputs)
            # Use the last hidden state of the last token
            last_hidden = outputs.last_hidden_state[:, -1, :]
            reward = self.reward_head(last_hidden)
            return reward.item()

    def score_pair(self, chosen: str, rejected: str) -> Tuple[float, float]:
        """Score a chosen/rejected pair.

        Args:
            chosen: Preferred response.
            rejected: Dispreferred response.

        Returns:
            (chosen_score, rejected_score)
        """
        return self.score(chosen), self.score(rejected)

    def close(self):
        self._closed = True
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Reward model resources released")


class RewardModelTrainer:
    """Trainer for the Reward Model."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-0.6B",
        config: Optional[RewardModelConfig] = None,
    ):
        self.config = config or RewardModelConfig(base_model=base_model)
        self.model = RewardModel(self.config)
        self._closed = False

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Train the reward model on preference data.

        Args:
            train_data: List of {"chosen": str, "rejected": str}.
            eval_data: Optional list of evaluation pairs.

        Returns:
            Training metrics.
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        # Training logic would go here (Bradley-Terry loss, etc.)
        # For now, this is a placeholder that returns mock metrics
        return {"loss": 0.5, "accuracy": 0.75}

    def close(self):
        self._closed = True
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
