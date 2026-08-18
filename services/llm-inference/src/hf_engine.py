"""
HuggingFace Transformers inference engine.

Implements the BaseEngine contract using raw HF transformers.
Serves as the fallback baseline when vLLM is not available.

Prefix KV cache (serve-time prefix reuse) is supported and gated behind
``prefix_cache_enabled`` (default off). When enabled it reuses the prefilled
KV of shared prompt prefixes (system prompt / RAG context) to skip re-prefill,
matching the 1.68–2.97× prefill speedup measured in
``research/inference_optimization/run_kv_cache_speedup.py``. Prefix reuse and
KV eviction are alternatives (the prefix path does not use the eviction cache);
prefix reuse is disabled while speculative decoding is active.
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
from optimization.prefix_cache import PrefixCache


def _clone_dynamic_cache(cache):
    """Return a deep copy of a HF ``DynamicCache`` (or compatible) KV cache.

    ``model.generate`` mutates the ``past_key_values`` it is given in place, so a
    cached KV must be cloned before being reused, otherwise a later shorter
    prefix request would resume from a longer (corrupted) cache.
    """
    cloned = type(cache)()
    for i, layer in enumerate(cache.layers):
        if layer.keys is None:
            continue
        cloned.update(layer.keys.clone(), layer.values.clone(), layer_idx=i)
    return cloned


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

        # Draft model for speculative decoding (loaded lazily, see
        # _ensure_draft_loaded). Kept on the engine so the decoder is cheap.
        self._draft_model = None
        self._draft_tokenizer = None

        # Serve-time prefix KV cache (opt-in). Holds prefilled KV for shared
        # prompt prefixes so repeated prefixes skip re-prefill. Disabled while
        # speculative decoding is active (it owns past_key_values rollback).
        self._prefix_cache = None
        if config.prefix_cache_enabled and not config.draft_model_path:
            self._prefix_cache = PrefixCache(capacity=config.prefix_cache_capacity)

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
        if self.config.speculative_enabled and self.config.draft_model_path:
            return self._generate_speculative(messages, kwargs)
        text = self._format_messages(messages)
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if self._prefix_cache is not None:
            return self._generate_with_prefix(input_ids, kwargs)
        # Default path: single prefill, no prefix reuse.
        with self._lock:
            output_ids = self.model.generate(
                input_ids=input_ids,
                past_key_values=self._select_kv_cache(),
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
        input_len = input_ids.shape[1]
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

    def _generate_with_prefix(
        self,
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> GenerationResult:
        """Synchronous generation reusing a matched prefix KV cache.

        On a cache miss the full prompt is prefilled once to obtain its KV
        (stored for future reuse); on a hit the matched prefix KV is cloned and
        only the suffix is re-prefilled. ``model.generate`` mutates the cache in
        place, so we always clone before reuse and never store the mutated one.
        """
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        prompt_list = input_ids[0].tolist()
        matched, cached = (0, None)
        if self._prefix_cache is not None:
            matched, cached = self._prefix_cache.lookup(prompt_list)

        past = None
        start = 0
        if cached is not None and matched > 0:
            # hit: clone the pristine prefix KV so the stored one stays intact
            past = _clone_dynamic_cache(cached)
            start = matched
        else:
            # miss: prefill the full prompt to obtain its KV, then store it
            with self._lock:
                out = self.model(input_ids=input_ids, use_cache=True)
            cached = out.past_key_values
            if self._prefix_cache is not None:
                self._prefix_cache.store(prompt_list, cached)
            past = _clone_dynamic_cache(cached)

        suffix = input_ids[:, start:]
        with self._lock:
            output_ids = self.model.generate(
                input_ids=suffix,
                past_key_values=past,
                max_new_tokens=max_tokens,
                temperature=kwargs.get("temperature", self.config.temperature),
                repetition_penalty=kwargs.get(
                    "repetition_penalty", self.config.repetition_penalty
                ),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                do_sample=True,
            )

        # Persist the pristine prefix KV (still valid; `past` was the clone that
        # got mutated) under the full prompt so identical/longer prompts reuse it.
        if self._prefix_cache is not None and cached is not None:
            self._prefix_cache.store(prompt_list, cached)

        suffix_len = input_ids.shape[1] - start
        new_tokens = output_ids[0][suffix_len:]
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
        if self.config.speculative_enabled and self.config.draft_model_path:
            yield from self._stream_speculative(messages, kwargs)
            return
        text = self._format_messages(messages)
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if self._prefix_cache is not None:
            yield from self._stream_with_prefix(input_ids, kwargs)
            return

        self._last_input_tokens = self.count_tokens(text)
        inputs = input_ids
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            past_key_values=self._select_kv_cache(),
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

    def _stream_with_prefix(
        self,
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> Iterator[GenerationResult]:
        """Streaming generation reusing a matched prefix KV cache (see _generate_with_prefix)."""
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        prompt_list = input_ids[0].tolist()
        matched, cached = (0, None)
        if self._prefix_cache is not None:
            matched, cached = self._prefix_cache.lookup(prompt_list)

        past = None
        start = 0
        if cached is not None and matched > 0:
            past = _clone_dynamic_cache(cached)
            start = matched
        else:
            with self._lock:
                out = self.model(input_ids=input_ids, use_cache=True)
            cached = out.past_key_values
            if self._prefix_cache is not None:
                self._prefix_cache.store(prompt_list, cached)
            past = _clone_dynamic_cache(cached)

        suffix = input_ids[:, start:]
        self._last_input_tokens = suffix.shape[1]
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(
            input_ids=suffix,
            past_key_values=past,
            streamer=streamer,
            max_new_tokens=max_tokens,
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

        total_time = time.time() - start_time
        if self._prefix_cache is not None and cached is not None:
            self._prefix_cache.store(prompt_list, cached)
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

    # ------------------------------------------------------------------
    # KV-cache eviction (opt-in via kv_eviction_window)
    # ------------------------------------------------------------------

    def _build_eviction_cache(self):
        """Return a fresh SinkWindowCache if KV eviction is enabled, else None.

        Lazy import so this module stays importable under the mocked-torch /
        mocked-transformers test files (same pattern as speculative decoding).
        A fresh cache per call — never shared across concurrent requests.
        """
        if not self.config._kv_eviction_enabled():
            return None
        from optimization.kv_eviction import SinkWindowCache

        return SinkWindowCache(
            sink_size=self.config.kv_eviction_sink,
            window_size=self.config.kv_eviction_window,
        )

    def _select_kv_cache(self):
        """Return the KV cache to inject: MLA compression if enabled, else eviction."""
        if self.config.kv_compression == "mla":
            return self._build_compression_cache()
        return self._build_eviction_cache()

    def _build_compression_cache(self):
        """Return a fresh CompressedKVCache (MLA latent KV) if enabled, else None.

        Uses a calibrated PCA basis from KV_COMPRESSION_BASIS (.pt) when given;
        otherwise a random orthonormal projection (still compresses, quality
        measured by research/inference_optimization/run_kv_compression_quality.py).
        """
        if self.config.kv_compression != "mla":
            return None
        from optimization.kv_compression import (
            CompressedKVCache,
            load_basis,
            random_basis,
        )

        n_layers = self.model.config.num_hidden_layers
        head_dim = getattr(self.model.config, "head_dim", None) or (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        latent = self.config.kv_compression_latent
        if self.config.kv_compression_basis:
            basis_k, basis_v = load_basis(self.config.kv_compression_basis)
        else:
            basis_k, basis_v = random_basis(n_layers, head_dim, latent, device=str(self.device))
        return CompressedKVCache(basis_k=basis_k, basis_v=basis_v, latent_dim=latent)

    # ------------------------------------------------------------------
    # Speculative decoding (opt-in via draft_model_path)
    # ------------------------------------------------------------------

    def _ensure_draft_loaded(self):
        """Load the draft model once; no-op if already loaded or disabled.

        Called under ``self._lock`` so concurrent first requests serialize.
        """
        if self._draft_model is not None or not self.config.draft_model_path:
            return
        self._draft_tokenizer = AutoTokenizer.from_pretrained(
            self.config.draft_model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.draft_model_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype="auto",
        ).to(self.device)
        self._draft_model.eval()
        logger.info(
            f"HFEngine: draft model '{self.config.draft_model_path}' "
            f"loaded on {self.device}"
        )

    def _build_speculative_decoder(self, kwargs):
        """Lazy-import and build the SpeculativeDecoder for this request."""
        # Lazy import: this module must not import torch/transformers at top
        # level is fine, but the decoder module stays importable under the
        # mocked-torch test files.
        from optimization.speculative_decoding import SpeculativeDecoder

        self._ensure_draft_loaded()
        return SpeculativeDecoder(
            draft_model=self._draft_model,
            draft_tokenizer=self._draft_tokenizer,
            target_model=self.model,
            target_tokenizer=self.tokenizer,
            gamma=self.config.speculative_gamma,
            device=self.device,
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
        )

    def _log_speculative(self, stats, generated_count):
        logger.info(
            f"HFEngine speculative: {generated_count} tokens, "
            f"acceptance_rate={stats.acceptance_rate:.3f}, "
            f"target_fwd={stats.n_target_forwards} "
            f"draft_fwd={stats.n_draft_forwards}, "
            f"E[tokens/verify]="
            f"{stats.expected_tokens_per_verify(self.config.speculative_gamma):.2f}"
        )

    def _generate_speculative(
        self,
        messages: List[Dict[str, str]],
        kwargs: dict,
    ) -> GenerationResult:
        """Non-streaming generation via the real draft-verify loop."""
        prompt = self._format_messages(messages)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self.device
        )
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        start_time = time.time()
        with self._lock:
            decoder = self._build_speculative_decoder(kwargs)
            output_ids, stats = decoder.generate(
                prompt_ids,
                max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time

        chunk = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_count = len(output_ids)

        self._metrics = EngineMetrics(
            tokens_generated=self._metrics.tokens_generated + generated_count,
            total_time=self._metrics.total_time + elapsed,
            first_token_latency=self._metrics.first_token_latency,
        )
        self._metrics.compute_tps()
        self._log_speculative(stats, generated_count)

        return GenerationResult(
            chunk=chunk,
            is_finished=True,
            generated_tokens=generated_count,
            metrics=self._metrics,
        )

    def _stream_speculative(
        self,
        messages: List[Dict[str, str]],
        kwargs: dict,
    ) -> Iterator[GenerationResult]:
        """Streaming generation via the real draft-verify loop.

        Yields the delta of the full decoded buffer each round (prefix-stable
        for byte-level BPE), so no partial-UTF8 or token-merge artifacts leak.
        """
        prompt = self._format_messages(messages)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self.device
        )
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        start_time = time.time()
        first_token = True
        buffer: List[int] = []
        prev_text = ""
        generated_tokens = 0
        stats = None

        with self._lock:
            decoder = self._build_speculative_decoder(kwargs)
            for new_ids, stats in decoder.stream_tokens(
                prompt_ids,
                max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            ):
                buffer.extend(new_ids)
                text = self.tokenizer.decode(buffer, skip_special_tokens=True)
                delta = text[len(prev_text):]
                prev_text = text
                if delta:
                    if first_token:
                        self._metrics.first_token_latency = time.time() - start_time
                        first_token = False
                    generated_tokens = len(buffer)
                    yield GenerationResult(
                        chunk=delta,
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
        if stats is not None:
            self._log_speculative(stats, generated_tokens)

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
        if self._draft_model is not None:
            del self._draft_model
            self._draft_model = None
        if self._draft_tokenizer is not None:
            del self._draft_tokenizer
            self._draft_tokenizer = None
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
            "draft_model": self.config.draft_model_path,
            "speculative_enabled": self.config.speculative_enabled,
            "kv_eviction_window": self.config.kv_eviction_window,
            "kv_eviction_sink": self.config.kv_eviction_sink,
            "prefix_cache_enabled": self.config.prefix_cache_enabled,
            "prefix_cache_capacity": self.config.prefix_cache_capacity,
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
