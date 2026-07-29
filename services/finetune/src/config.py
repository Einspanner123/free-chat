"""
Fine-tuning configuration module.

Defines training hyperparameters, LoRA/QLoRA configs, and
a unified TrainingPipelineConfig for the full pipeline.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union


# ---------------------------------------------------------------------------
# FineTuneConfig
# ---------------------------------------------------------------------------


@dataclass
class FineTuneConfig:
    """Training hyperparameters for fine-tuning."""
    base_model: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = False
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    report_to: str = "none"
    deepspeed: Optional[str] = None

    def __post_init__(self):
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.max_seq_length < 1:
            raise ValueError(
                f"max_seq_length must be >= 1, got {self.max_seq_length}"
            )

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "FineTuneConfig":
        valid_keys = {
            "base_model", "output_dir", "num_epochs", "batch_size",
            "gradient_accumulation_steps", "learning_rate", "warmup_ratio",
            "max_seq_length", "save_steps", "logging_steps", "eval_steps",
            "save_total_limit", "gradient_checkpointing", "optimizer",
            "lr_scheduler", "report_to", "deepspeed",
        }
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def save_yaml(self, path: str):
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def load_yaml(cls, path: str) -> "FineTuneConfig":
        import yaml
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def __repr__(self):
        return f"FineTuneConfig(model={self.base_model}, epochs={self.num_epochs}, lr={self.learning_rate})"


# ---------------------------------------------------------------------------
# LoraConfig
# ---------------------------------------------------------------------------

_VALID_TARGET_MODULES_STR = {"all-linear"}


@dataclass
class LoraConfig:
    """LoRA hyperparameters."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Union[str, List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"r (rank) must be >= 1, got {self.r}")
        if self.lora_alpha < 1:
            raise ValueError(
                f"lora_alpha must be >= 1, got {self.lora_alpha}"
            )
        if isinstance(self.target_modules, str):
            if self.target_modules not in _VALID_TARGET_MODULES_STR:
                raise ValueError(
                    f"target_modules string must be one of {_VALID_TARGET_MODULES_STR}, "
                    f"got '{self.target_modules}'"
                )
        elif not isinstance(self.target_modules, list):
            raise ValueError(
                "target_modules must be a list of strings or 'all-linear'"
            )

    def to_peft_dict(self) -> dict:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "use_rslora": self.use_rslora,
        }


# ---------------------------------------------------------------------------
# QLoraConfig
# ---------------------------------------------------------------------------


@dataclass
class QLoraConfig:
    """QLoRA quantization configuration."""
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "Cannot enable both 4-bit and 8-bit quantization simultaneously"
            )

    @staticmethod
    def none() -> "QLoraConfig":
        return QLoraConfig(load_in_4bit=False, load_in_8bit=False)

    def to_bnb_dict(self) -> dict:
        base = {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
        }
        if self.load_in_4bit:
            base.update({
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            })
        return base


# ---------------------------------------------------------------------------
# TrainingPipelineConfig
# ---------------------------------------------------------------------------


@dataclass
class TrainingPipelineConfig:
    """Unified configuration for the full training pipeline."""
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    qlora: QLoraConfig = field(default_factory=QLoraConfig)

    def to_dict(self) -> dict:
        return {
            "finetune": asdict(self.finetune),
            "lora": asdict(self.lora),
            "qlora": asdict(self.qlora),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingPipelineConfig":
        return cls(
            finetune=FineTuneConfig.from_dict(d.get("finetune", {})),
            lora=LoraConfig(**d.get("lora", {})),
            qlora=QLoraConfig(**d.get("qlora", {})),
        )

    def is_qlora(self) -> bool:
        return self.qlora.load_in_4bit or self.qlora.load_in_8bit

    def get_effective_batch_size(self) -> int:
        return self.finetune.batch_size * self.finetune.gradient_accumulation_steps
