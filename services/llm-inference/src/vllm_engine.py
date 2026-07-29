"""
vLLM inference engine.

Implements the BaseEngine contract using vLLM for high-performance
inference with PagedAttention, continuous batching, and quantization support.
"""

import time
from typing import Optional, Iterator, List, Dict, Any

from loguru import logger

from engine_base import (
    BaseEngine,
    EngineConfig,
    EngineMetrics,
    GenerationResult,
    PromptFormat,
)


class VLLMEngine(BaseEngine):
    """vLLM-based inference engine.

    Requires the ``vllm`` package to be installed.
    Provides PagedAttention, continuous batching, and quantization (AWQ/GPTQ).
    """

    def __init__(self, config: Optional[EngineConfig] = None, **kwargs):
        if config is None:
            config = EngineConfig(**kwargs)
        super().__init__(config)
        self._metrics = EngineMetrics()
        self._closed = False
        self._llm = None
        self._tokenizer = None

        try:
            from vllm import LLM as VLLMBackend
            from vllm import SamplingParams as VLLMSamplingParams
            self._VLLMBackend = VLLMBackend
            self._VLLMSamplingParams = VLLMSamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. "
                "Install it with: pip install vllm"
            )

        logger.info(
            f"VLLMEngine: initializing model '{config.model_path}' "
            f"(quantization={config.quantization}, "
            f"tp={config.tensor_parallel_size}, "
            f"gpu_mem={config.gpu_memory_utilization})"
        )

        # Build vLLM kwargs
        vllm_kwargs = {
            "model": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.quantization:
            vllm_kwargs["quantization"] = config.quantization

        self._llm = self._VLLMBackend(**vllm_kwargs)

        # Try to get the tokenizer from vLLM
        try:
            if hasattr(self._llm, "get_tokenizer"):
                self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            pass

        logger.info("VLLMEngine: initialized successfully")

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """Synchronous generation (non-streaming)."""
        prompt = self._format_messages(messages)
        sampling_params = self._build_sampling_params(**kwargs)

        start_time = time.time()
        outputs = self._llm.generate([prompt], sampling_params)
        total_time = time.time() - start_time

        if not outputs or not outputs[0].outputs:
            return GenerationResult(
                chunk="",
                is_finished=True,
                generated_tokens=0,
                metrics=EngineMetrics(),
            )

        output = outputs[0].outputs[0]
        chunk = output.text
        generated_tokens = len(output.token_ids)

        self._metrics = EngineMetrics(
            tokens_generated=self._metrics.tokens_generated + generated_tokens,
            total_time=self._metrics.total_time + total_time,
            first_token_latency=self._metrics.first_token_latency,
        )
        self._metrics.compute_tps()

        return GenerationResult(
            chunk=chunk,
            is_finished=True,
            generated_tokens=generated_tokens,
            metrics=self._metrics,
        )

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[GenerationResult]:
        """Streaming generation using vLLM."""
        prompt = self._format_messages(messages)
        sampling_params = self._build_sampling_params(**kwargs)

        start_time = time.time()
        first_token = True
        generated_tokens = 0

        # vLLM streaming output
        outputs = self._llm.generate([prompt], sampling_params)

        if not outputs or not outputs[0].outputs:
            yield GenerationResult(
                chunk="",
                is_finished=True,
                generated_tokens=0,
            )
            return

        output = outputs[0].outputs[0]
        full_text = output.text
        token_ids = output.token_ids

        # vLLM doesn't natively yield per-token in the simple API,
        # so we simulate streaming by yielding the full text as one chunk.
        # For true per-token streaming, use the async API.
        if first_token:
            self._metrics.first_token_latency = time.time() - start_time
            first_token = False

        generated_tokens = len(token_ids)

        if full_text:
            yield GenerationResult(
                chunk=full_text,
                is_finished=False,
                generated_tokens=generated_tokens,
            )

        total_time = time.time() - start_time
        self._metrics = EngineMetrics(
            tokens_generated=self._metrics.tokens_generated + generated_tokens,
            total_time=self._metrics.total_time + total_time,
            first_token_latency=self._metrics.first_token_latency,
        )
        self._metrics.compute_tps()

        yield GenerationResult(
            chunk="",
            is_finished=True,
            generated_tokens=generated_tokens,
            metrics=self._metrics,
        )

    def count_tokens(self, text: str) -> int:
        """Token count using vLLM's tokenizer."""
        if not text:
            return 0
        try:
            if self._tokenizer is not None:
                return len(self._tokenizer.encode(text))
        except Exception:
            pass
        # Fallback
        return max(len(text) // 2, 1)

    def get_metrics(self) -> EngineMetrics:
        return self._metrics

    def close(self):
        self._closed = True
        if self._llm is not None:
            try:
                # vLLM handles cleanup via context manager
                pass
            except Exception:
                pass
        self._llm = None
        self._tokenizer = None
        logger.info("VLLMEngine: resources released")

    def info(self) -> dict:
        return {
            "type": "vllm",
            "model": self.config.model_path,
            "quantization": self.config.quantization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "max_tokens": self.config.max_tokens,
            "closed": self._closed,
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        if not messages:
            return ""
        try:
            if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass
        return PromptFormat.apply_chat_template(messages) or ""

    def _build_sampling_params(self, **kwargs) -> Any:
        """Build vLLM SamplingParams from config and overrides."""
        return self._VLLMSamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
        )
