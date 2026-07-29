"""
Engine factory for creating inference engines.

Supports auto-detection of available backends.
"""

import importlib
from enum import Enum
from typing import Optional, List, Dict, Any

from engine_base import BaseEngine, EngineConfig
from quantization import QuantizationRegistry, QuantizationMethod

quantization_registry = QuantizationRegistry()


class EngineType(Enum):
    VLLM = "vllm"
    HF = "hf"
    AUTO = "auto"

    def __str__(self):
        return self.value


class EngineFactory:
    """Factory for creating inference engine instances."""

    @staticmethod
    def available_engines() -> List[EngineType]:
        return [EngineType.VLLM, EngineType.HF, EngineType.AUTO]

    @staticmethod
    def _is_vllm_available() -> bool:
        try:
            importlib.import_module("vllm")
            return True
        except ImportError:
            return False

    @staticmethod
    def create(
        engine_type: EngineType = EngineType.AUTO,
        model_path: str = "",
        quantization: Optional[str] = None,
        **kwargs,
    ) -> BaseEngine:
        """Create an engine instance.

        Args:
            engine_type: Type of engine to create.
            model_path: Path or HuggingFace model ID.
            quantization: Quantization method (None, 'awq', 'gptq', etc.).
            **kwargs: Additional EngineConfig parameters.

        Returns:
            An initialized engine instance.

        Raises:
            ValueError: If engine type is not supported or quantization is invalid.
        """
        # Validate quantization
        if quantization is not None:
            if not quantization_registry.is_supported(quantization):
                raise ValueError(
                    f"Unknown quantization method '{quantization}'. "
                    f"Supported: {quantization_registry.get_vllm_supported()}"
                )

        # Build config
        config = EngineConfig(model_path=model_path, quantization=quantization, **kwargs)

        # Resolve engine type
        if engine_type == EngineType.AUTO:
            if EngineFactory._is_vllm_available():
                engine_type = EngineType.VLLM
            else:
                engine_type = EngineType.HF

        # Create engine
        if engine_type == EngineType.VLLM:
            return EngineFactory._create_vllm_engine(config)
        elif engine_type == EngineType.HF:
            return EngineFactory._create_hf_engine(config)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

    @staticmethod
    def _create_vllm_engine(config: EngineConfig) -> BaseEngine:
        from vllm_engine import VLLMEngine
        return VLLMEngine(config)

    @staticmethod
    def _create_hf_engine(config: EngineConfig) -> BaseEngine:
        from hf_engine import HFEngine
        return HFEngine(config)
