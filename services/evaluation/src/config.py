"""Evaluation configuration."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class MMLUConfig:
    num_few_shot: int = 5
    subjects: Optional[List[str]] = None  # None = all subjects


@dataclass
class CEvalConfig:
    num_few_shot: int = 5
    subject: Optional[str] = None


@dataclass
class GSM8KConfig:
    num_few_shot: int = 8
    test_only: bool = True


@dataclass
class HumanEvalConfig:
    num_samples: int = 1
    test_only: bool = True


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    batch_size: int = 8
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    benchmarks: List[str] = field(default_factory=lambda: ["mmlu", "ceval", "gsm8k", "humaneval"])

    # Per-benchmark configs
    mmlu: MMLUConfig = field(default_factory=MMLUConfig)
    ceval: CEvalConfig = field(default_factory=CEvalConfig)
    gsm8k: GSM8KConfig = field(default_factory=GSM8KConfig)
    humaneval: HumanEvalConfig = field(default_factory=HumanEvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        valid_keys = {"model_name", "batch_size", "max_tokens", "temperature", "top_p", "top_k", "benchmarks"}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


_VALID_FORMATS = {"markdown", "json", "html"}


@dataclass
class ReportConfig:
    output_dir: str = "./eval_results"
    format: str = "markdown"
    include_plots: bool = True

    def __post_init__(self):
        if self.format not in _VALID_FORMATS:
            raise ValueError(f"format must be one of {_VALID_FORMATS}, got '{self.format}'")
