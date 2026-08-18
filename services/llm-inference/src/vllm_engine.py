"""
vLLM inference engine.

Implements the BaseEngine contract using vLLM's AsyncLLM for
high-performance streaming inference with PagedAttention, continuous
batching, and quantization support (AWQ/GPTQ).

vLLM >= 0.26 is required. The sync LLM API does not support streaming,
so this engine drives the async engine on a dedicated event-loop thread
and bridges the async generator to the synchronous iterator contract
via a thread-safe queue.

When ``prefix_cache_enabled`` is set, vLLM's native automatic prefix
caching is enabled (``enable_prefix_caching=True``) — reusing KV for shared
prompt prefixes, the production-grade equivalent of the HF path in
``hf_engine.py``.
"""

import asyncio
import os
import queue
import threading
import time
import uuid
from typing import Optional, Iterator, List, Dict, Any

from loguru import logger

# flashinfer's JIT-compiled sampler requires nvcc / a CUDA toolkit, which
# is often absent on driver-only GPU hosts. Fall back to vLLM's built-in
# sampler (no JIT). Must be set before vllm is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from engine_base import (
    BaseEngine,
    EngineConfig,
    EngineMetrics,
    GenerationResult,
    PromptFormat,
)


class VLLMEngine(BaseEngine):
    """vLLM-based inference engine (AsyncLLM backend).

    Provides true token-level streaming, PagedAttention, continuous
    batching, and quantization (AWQ/GPTQ).
    """

    def __init__(self, config: Optional[EngineConfig] = None, **kwargs):
        if config is None:
            config = EngineConfig(**kwargs)
        super().__init__(config)
        self._metrics = EngineMetrics()
        self._closed = False
        self._llm: Any = None
        self._tokenizer = None

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLM
            from vllm import SamplingParams as VLLMSamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. "
                "Install it with: pip install vllm"
            )

        self._AsyncEngineArgs = AsyncEngineArgs
        self._AsyncLLM = AsyncLLM
        self._VLLMSamplingParams = VLLMSamplingParams

        logger.info(
            f"VLLMEngine: initializing model '{config.model_path}' "
            f"(quantization={config.quantization}, "
            f"tp={config.tensor_parallel_size}, "
            f"gpu_mem={config.gpu_memory_utilization})"
        )

        # Dedicated event-loop thread drives the async engine. The
        # gRPC server is synchronous, so all async work is submitted
        # to this loop and bridged back through queues.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="vllm-engine-loop",
            daemon=True,
        )
        self._loop_thread.start()

        # Create the AsyncLLM inside the event loop so its output
        # handler task is started on the correct loop.
        init_future = asyncio.run_coroutine_threadsafe(
            self._init_async_engine(), self._loop
        )
        try:
            init_future.result(timeout=1800)  # model load can be slow
        except Exception as e:
            self._stop_loop()
            raise RuntimeError(f"VLLMEngine initialization failed: {e}") from e

        logger.info("VLLMEngine: initialized successfully")

    async def _init_async_engine(self):
        kwargs = dict(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=self.config.trust_remote_code,
            quantization=self.config.quantization or None,
            disable_log_stats=True,
        )
        # Native vLLM speculative decoding (draft_model method). spec_tokens
        # maps to num_speculative_tokens and is required for a plain HF draft.
        if self.config.speculative_enabled and self.config.draft_model_path:
            kwargs["spec_model"] = self.config.draft_model_path
            kwargs["spec_tokens"] = self.config.speculative_gamma
            logger.info(
                f"VLLMEngine: speculative decoding ON: "
                f"draft={self.config.draft_model_path} "
                f"gamma={self.config.speculative_gamma}"
            )
        # Native vLLM automatic prefix caching: reuse KV for shared prompt
        # prefixes (system prompt / RAG context) to skip re-prefill.
        if self.config.prefix_cache_enabled:
            kwargs["enable_prefix_caching"] = True
            logger.info("VLLMEngine: automatic prefix caching ON")
        engine_args = self._AsyncEngineArgs(**kwargs)
        self._llm = self._AsyncLLM.from_engine_args(engine_args)
        try:
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            self._tokenizer = None

    def _stop_loop(self):
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=5)
            except Exception:
                pass

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """Synchronous generation (non-streaming)."""
        chunks = []
        result: Optional[GenerationResult] = None
        for r in self.stream_generate(messages, **kwargs):
            if r.is_finished:
                result = r
            else:
                chunks.append(r.chunk)
        full_text = "".join(chunks)
        return GenerationResult(
            chunk=full_text,
            is_finished=True,
            generated_tokens=result.generated_tokens if result else 0,
            metrics=result.metrics if result else self._metrics,
        )

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[GenerationResult]:
        """True token-level streaming generation via AsyncLLM."""
        prompt = self._format_messages(messages)
        sampling_params = self._build_sampling_params(**kwargs)
        request_id = f"req-{uuid.uuid4().hex[:12]}"

        bridge: "queue.Queue[Any]" = queue.Queue()
        sentinel = object()

        async def _run():
            try:
                async for output in self._llm.generate(
                    prompt, sampling_params, request_id
                ):
                    bridge.put(output)
            except Exception as e:  # pragma: no cover - error path
                bridge.put(e)
            finally:
                bridge.put(sentinel)

        # Submit the async streaming task; results arrive via bridge.
        task = asyncio.run_coroutine_threadsafe(_run(), self._loop)

        start_time = time.time()
        first_token = True
        generated_tokens = 0

        try:
            while True:
                item = bridge.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                outputs = item.outputs
                if not outputs:
                    continue
                text = outputs[0].text
                token_ids = outputs[0].token_ids or []
                if text:
                    if first_token:
                        self._metrics.first_token_latency = (
                            time.time() - start_time
                        )
                        first_token = False
                    generated_tokens = len(token_ids)
                    yield GenerationResult(
                        chunk=text,
                        is_finished=False,
                        generated_tokens=generated_tokens,
                    )
        finally:
            # Make sure the async task is not leaked if the consumer
            # stops early (e.g. client disconnect).
            if not task.done():
                task.cancel()

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
        self._llm = None
        self._tokenizer = None
        self._stop_loop()
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
            "draft_model": self.config.draft_model_path,
            "speculative_enabled": self.config.speculative_enabled,
            "prefix_cache_enabled": self.config.prefix_cache_enabled,
            "prefix_cache_capacity": self.config.prefix_cache_capacity,
            "closed": self._closed,
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        if not messages:
            return ""
        try:
            if self._tokenizer is not None and hasattr(
                self._tokenizer, "apply_chat_template"
            ):
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
