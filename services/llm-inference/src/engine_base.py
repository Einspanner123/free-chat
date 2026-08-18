"""
Engine abstraction layer for LLM inference.

Defines the contract that all inference engines must satisfy.
Supports pluggable backends (vLLM, HuggingFace, etc.)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator, List, Dict, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_QUANTIZATIONS = {None, "awq", "gptq", "squeezellm"}


def _validate_quantization(value):
    if value not in _VALID_QUANTIZATIONS:
        raise ValueError(
            f"Invalid quantization '{value}'. Must be one of: "
            f"{[q for q in _VALID_QUANTIZATIONS if q is not None]}, or None"
        )


@dataclass
class EngineConfig:
    """Universal engine configuration.

    Every engine backend receives an EngineConfig and extracts
    the parameters it supports.
    """
    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.05
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    quantization: Optional[str] = None
    trust_remote_code: bool = True
    # Speculative decoding (opt-in via draft_model_path)
    draft_model_path: Optional[str] = None
    speculative_gamma: int = 5
    speculative_enabled: bool = True
    # KV-cache eviction (StreamingLLM-style; opt-in via kv_eviction_window)
    kv_eviction_sink: int = 4
    kv_eviction_window: Optional[int] = None  # None / <=0 disables
    # Prefix KV cache (serve-time prefix reuse; opt-in via prefix_cache_enabled)
    prefix_cache_enabled: bool = False
    prefix_cache_capacity: int = 8
    # KV compression (MLA-style latent compression; opt-in via kv_compression)
    kv_compression: str = "none"  # "none" | "mla"
    kv_compression_latent: int = 16
    kv_compression_basis: Optional[str] = None  # path to .pt basis

    def __post_init__(self):
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.speculative_gamma < 1:
            raise ValueError(
                f"speculative_gamma must be >= 1, got {self.speculative_gamma}"
            )
        if self.kv_eviction_sink < 0:
            raise ValueError(
                f"kv_eviction_sink must be >= 0, got {self.kv_eviction_sink}"
            )
        if self.kv_eviction_window is not None and self.kv_eviction_window < 0:
            raise ValueError(
                f"kv_eviction_window must be >= 0, got {self.kv_eviction_window}"
            )
        if (
            self.draft_model_path
            and self.speculative_enabled
            and self._kv_eviction_enabled()
        ):
            raise ValueError(
                "KV-cache eviction (SinkWindowCache) is incompatible with "
                "speculative decoding: the speculative loop rolls back the KV "
                "cache with crop(), which SinkWindowCache forbids. Enable one "
                "or the other."
            )
        if self.prefix_cache_capacity < 1:
            raise ValueError(
                f"prefix_cache_capacity must be >= 1, got {self.prefix_cache_capacity}"
            )
        if self.kv_compression not in ("none", "mla"):
            raise ValueError(
                f"kv_compression must be 'none' or 'mla', got {self.kv_compression}"
            )
        if self.kv_compression_latent < 1:
            raise ValueError(
                f"kv_compression_latent must be >= 1, got {self.kv_compression_latent}"
            )
        if self.kv_compression != "none" and self._kv_eviction_enabled():
            raise ValueError(
                "KV compression (CompressedKVCache) is incompatible with KV "
                "eviction (SinkWindowCache): both rewrite the KV cache. Enable "
                "one or the other."
            )
        _validate_quantization(self.quantization)

    def _kv_eviction_enabled(self) -> bool:
        return self.kv_eviction_window is not None and self.kv_eviction_window > 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        valid_keys = {
            "model_path", "max_tokens", "temperature", "top_p", "top_k",
            "repetition_penalty", "gpu_memory_utilization", "tensor_parallel_size",
            "max_model_len", "quantization", "trust_remote_code",
            "draft_model_path", "speculative_gamma", "speculative_enabled",
            "kv_eviction_sink", "kv_eviction_window",
            "prefix_cache_enabled", "prefix_cache_capacity",
            "kv_compression", "kv_compression_latent", "kv_compression_basis",
        }
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class EngineMetrics:
    """Performance metrics collected during generation."""
    tokens_generated: int = 0
    total_time: float = 0.0
    tokens_per_second: float = 0.0
    first_token_latency: float = 0.0

    def compute_tps(self):
        if self.total_time > 0 and self.tokens_generated > 0:
            self.tokens_per_second = self.tokens_generated / self.total_time
        else:
            self.tokens_per_second = 0.0

    @staticmethod
    def merge(*metrics_list: "EngineMetrics") -> "EngineMetrics":
        if not metrics_list:
            return EngineMetrics()
        total_tokens = sum(m.tokens_generated for m in metrics_list)
        total_time = sum(m.total_time for m in metrics_list)
        first_latency = metrics_list[0].first_token_latency
        merged = EngineMetrics(
            tokens_generated=total_tokens,
            total_time=total_time,
            first_token_latency=first_latency,
        )
        merged.compute_tps()
        return merged

    def __str__(self):
        return (
            f"EngineMetrics(tokens={self.tokens_generated}, "
            f"time={self.total_time:.2f}s, "
            f"tps={self.tokens_per_second:.2f}, "
            f"ttft={self.first_token_latency:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Generation Result
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """A single chunk of generated text (streaming or final)."""
    chunk: str
    is_finished: bool
    generated_tokens: int
    metrics: Optional[EngineMetrics] = None

    def __post_init__(self):
        if self.generated_tokens < 0:
            raise ValueError(
                f"generated_tokens must be >= 0, got {self.generated_tokens}"
            )


# ---------------------------------------------------------------------------
# Prompt Formatting Utilities
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant"}


class PromptFormat:
    """Utilities for formatting messages into model inputs."""

    SUPPORTED_ROLES = _VALID_ROLES

    @staticmethod
    def validate_messages(messages: List[Dict[str, str]]):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Each message must be a dict, got {type(msg)}")
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' field")
            if msg["role"] not in _VALID_ROLES:
                raise ValueError(
                    f"Unsupported role '{msg['role']}'. "
                    f"Must be one of: {sorted(_VALID_ROLES)}"
                )
            if "content" not in msg:
                raise ValueError("Each message must have a 'content' field")

    @staticmethod
    def apply_chat_template(messages: List[Dict[str, str]]) -> Optional[str]:
        """Apply a basic chat template.

        This is a fallback for engines that don't use a tokenizer's
        apply_chat_template. Real engines should delegate to the
        tokenizer's native method.
        """
        if not messages:
            return None
        PromptFormat.validate_messages(messages)
        parts = []
        for m in messages:
            role = m["role"]
            content = str(m.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def count_tokens_in_messages(
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
    ) -> int:
        """Quick token count estimation for messages.

        For production use, delegate to the specific engine's tokenizer.
        This is a rough estimate using 2 chars per token rule.
        """
        if not messages:
            return 0
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        # Add overhead for role markers and template
        overhead = len(messages) * 10
        return max(1, (total_chars + overhead) // 2)


# ---------------------------------------------------------------------------
# Abstract Base Engine
# ---------------------------------------------------------------------------


class BaseEngine(ABC):
    """Abstract base class for all inference engines."""

    def __init__(self, config: EngineConfig):
        self._config = config

    @property
    def config(self) -> EngineConfig:
        return self._config

    # ---- Abstract methods that every engine must implement ----

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """Synchronous generation. Returns the full result at once."""
        ...

    @abstractmethod
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[GenerationResult]:
        """Streaming generation. Yields chunks as they are produced."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...

    @abstractmethod
    def get_metrics(self) -> EngineMetrics:
        """Return cumulative metrics for this engine instance."""
        ...

    @abstractmethod
    def close(self):
        """Release all resources held by the engine."""
        ...

    @abstractmethod
    def info(self) -> dict:
        """Return engine metadata (type, model, capabilities)."""
        ...

    # ---- Context manager support ----

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
