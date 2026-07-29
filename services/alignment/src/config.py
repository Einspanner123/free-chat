"""
Configuration for DPO alignment pipeline.

Defines DPO hyperparameters (beta, loss type, label smoothing)
and the unified DPOPipelineConfig.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List

_VALID_LOSS_TYPES = {"sigmoid", "ipo", "kto_pair"}


@dataclass
class DPOConfig:
    """DPO (Direct Preference Optimization) hyperparameters.

    Attributes:
        beta: Temperature parameter for DPO loss. Higher = more
              focus on preference margin.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate for DPO training.
        max_length: Maximum total sequence length.
        max_prompt_length: Maximum prompt (input) length.
        reference_model: Optional separate reference model.
                         If None, uses a frozen copy of the policy model.
        loss_type: Loss variant: "sigmoid" (standard DPO),
                   "ipo" (Identity Preference Optimization),
                   "kto_pair" (Kahneman-Tversky Optimization).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
    """
    beta: float = 0.1
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-6
    max_length: int = 2048
    max_prompt_length: int = 1024
    reference_model: Optional[str] = None
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {_VALID_LOSS_TYPES}, "
                f"got '{self.loss_type}'"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DPOConfig":
        valid_keys = {
            "beta", "num_epochs", "batch_size", "learning_rate",
            "max_length", "max_prompt_length", "reference_model",
            "loss_type", "label_smoothing",
        }
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class DPOPipelineConfig:
    """Unified configuration for DPO training pipeline."""
    base_model: str = "Qwen/Qwen3-0.6B"
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "dpo": asdict(self.dpo),
        }
