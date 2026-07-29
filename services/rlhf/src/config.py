from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    kl_penalty: float = 0.05
    clip_range: float = 0.2
    vf_coef: float = 0.1
    num_epochs: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    target_kl: float = 0.1
    max_grad_norm: float = 1.0
    adaptive_kl: bool = True

@dataclass
class RLHFConfig:
    base_model: str = "Qwen/Qwen3-0.6B"
    reward_model: str = "Qwen/Qwen3-0.6B"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    output_dir: str = "./rlhf_output"
