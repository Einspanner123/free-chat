"""
HuggingFace Transformers inference engine.

Implements the BaseEngine contract using raw HF transformers.
Serves as the fallback baseline when vLLM is not available.
"""

import json
import time
from threading import Thread, Lock
from typing import Optional, Iterator, List, Dict, Any

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from engine_base import (
    BaseEngine,
    EngineConfig,
    EngineMetrics,
    GenerationResult,
    PromptFormat,
)
from quantization import QuantizationConfig, QuantizationMethod


class HFEngine(BaseEngine):
    """HuggingFace Transformers-based inference engine."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[EngineConfig] = None,
    ):
        if config is None and model_path is None:
            raise ValueError("Either model_path or config must be provided")
        if config is None:
            config = EngineConfig(model_path=model_path)
        elif model_path is not None:
            # Both provided, config takes precedence
            config = EngineConfig(model_path=model_path, **{k: v for k, v in config.to_dict().items() if k != 'model_path'})
        super().__init__(config)

        self._lock = Lock()
        self._metrics = EngineMetrics()
        self._closed = False
        self._last_input_tokens = 0

        # Device detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"HFEngine: loading model '{config.model_path}' on {self.device}")

        # Quantization
        model_kwargs = {}
        if config.quantization and config.quantization != "none":
            q_config = QuantizationConfig(
                method=QuantizationMethod(config.quantization),
                bits=4,
            )
            model_kwargs["quantization_config"] = q_config.to_hf_config()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype="auto",
            **model_kwargs,
        ).to(self.device)

        logger.info(f"HFEngine: model loaded successfully on {self.device}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """Synchronous generation (non-streaming)."""
        text = self._format_messages(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with self._lock:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                repetition_penalty=kwargs.get(
                    "repetition_penalty", self.config.repetition_penalty
                ),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                do_sample=True,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        chunk = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_count = len(new_tokens)

        self._metrics = EngineMetrics(
            tokens_generated=self._metrics.tokens_generated + generated_count,
            total_time=self._metrics.total_time,
            first_token_latency=self._metrics.first_token_latency,
        )

        return GenerationResult(
            chunk=chunk,
            is_finished=True,
            generated_tokens=generated_count,
        )

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[GenerationResult]:
        """Streaming generation."""
        text = self._format_messages(messages)

        self._last_input_tokens = self.count_tokens(text)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            do_sample=True,
        )

        start_time = time.time()
        first_token = True
        generated_tokens = 0

        def _safe_generate():
            with self._lock:
                self.model.generate(**gen_kwargs)

        thread = Thread(target=_safe_generate)
        thread.start()

        for chunk in streamer:
            if chunk:
                if first_token:
                    self._metrics.first_token_latency = time.time() - start_time
                    first_token = False
                generated_tokens += self.count_tokens(chunk)
                yield GenerationResult(
                    chunk=chunk,
                    is_finished=False,
                    generated_tokens=generated_tokens,
                )

        # Done
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
        """Token count using the loaded tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def get_metrics(self) -> EngineMetrics:
        return self._metrics

    def close(self):
        self._closed = True
        # Release GPU memory
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HFEngine: resources released")

    def info(self) -> dict:
        return {
            "type": "hf",
            "model": self.config.model_path,
            "device": str(self.device),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "quantization": self.config.quantization,
            "closed": self._closed,
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into model input text."""
        if not messages:
            return ""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Chat template failed, using fallback: {e}")
            return PromptFormat.apply_chat_template(messages) or ""
