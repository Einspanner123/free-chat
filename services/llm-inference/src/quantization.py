"""
Quantization configuration and registry for LLM inference.

Supports AWQ, GPTQ, SqueezeLLM, and no-quantization (FP16) modes.
Provides conversion between internal config and vLLM/HF formats.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class QuantizationMethod(Enum):
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"

    def __str__(self):
        return self.value


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        method: Quantization method.
        bits: Number of bits (4 or 8).
        group_size: Group size for group-wise quantization (must be multiple of 32).
        desc_act: Whether to use desc_act (GPTQ-specific).
    """
    method: QuantizationMethod
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False

    def __post_init__(self):
        if self.method == QuantizationMethod.NONE:
            self.bits = 16
        elif self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size % 32 != 0:
            raise ValueError(
                f"group_size must be a multiple of 32, got {self.group_size}"
            )

    @staticmethod
    def none() -> "QuantizationConfig":
        """Create a no-quantization config (FP16)."""
        return QuantizationConfig(method=QuantizationMethod.NONE, bits=16)

    def to_vllm_config(self) -> Optional[str]:
        """Convert to vLLM quantization parameter string.

        Returns:
            The quantization string expected by vLLM, or None if no quantization.
        """
        if self.method == QuantizationMethod.NONE:
            return None
        return self.method.value

    def to_hf_config(self) -> Dict:
        """Convert to HuggingFace BitsAndBytesConfig parameters.

        Returns:
            Dict with BitsAndBytes kwargs for HF transformers.
        """
        if self.method == QuantizationMethod.NONE:
            return {}
        return {
            "load_in_4bit": self.bits == 4,
            "load_in_8bit": self.bits == 8,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }


class QuantizationRegistry:
    """Registry of available quantization methods."""

    def __init__(self):
        self._methods = {
            QuantizationMethod.NONE: {"bits": [16], "vllm_supported": False},
            QuantizationMethod.AWQ: {"bits": [4], "vllm_supported": True},
            QuantizationMethod.GPTQ: {"bits": [4, 8], "vllm_supported": True},
            QuantizationMethod.SQUEEZELLM: {"bits": [4], "vllm_supported": True},
        }

    def list_methods(self) -> List[QuantizationMethod]:
        return list(self._methods.keys())

    def is_supported(self, name: str) -> bool:
        try:
            method = QuantizationMethod(name.lower())
            return method in self._methods
        except ValueError:
            return False

    def get_config(self, name: str, bits: int = 4, **kwargs) -> QuantizationConfig:
        try:
            method = QuantizationMethod(name.lower())
        except ValueError:
            raise ValueError(
                f"Unknown quantization method '{name}'. "
                f"Available: {[m.value for m in self._methods]}"
            )
        if method not in self._methods:
            raise ValueError(
                f"Quantization method '{name}' is not in the registry."
            )
        if method == QuantizationMethod.NONE:
            return QuantizationConfig.none()
        return QuantizationConfig(method=method, bits=bits, **kwargs)

    def get_vllm_supported(self) -> List[str]:
        return [
            m.value for m, info in self._methods.items()
            if info["vllm_supported"]
        ]


# Singleton
registry = QuantizationRegistry()
