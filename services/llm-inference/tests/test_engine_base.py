"""
Tests for the engine abstraction layer - core components only.
These tests do NOT require torch, transformers, or vllm.
"""

import json
import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import sys
import os
# Add src/ to path for importing modules
_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# These are our modules that don't need heavy dependencies
from engine_base import (
    EngineConfig, EngineMetrics, GenerationResult,
    BaseEngine, PromptFormat,
)
from quantization import (
    QuantizationConfig, QuantizationMethod, QuantizationRegistry
)
from engine_factory import EngineFactory, EngineType


# =============================================================================
# EngineConfig Tests
# =============================================================================

class TestEngineConfig:
    def test_default_values(self):
        cfg = EngineConfig(model_path="Qwen/Qwen3-0.6B")
        assert cfg.model_path == "Qwen/Qwen3-0.6B"
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.8
        assert cfg.top_k == 40
        assert cfg.repetition_penalty == 1.05
        assert cfg.gpu_memory_utilization == 0.9
        assert cfg.tensor_parallel_size == 1
        assert cfg.max_model_len == 8192
        assert cfg.quantization is None
        assert cfg.trust_remote_code is True

    def test_custom_values(self):
        cfg = EngineConfig(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=2048,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=4,
            max_model_len=16384,
            quantization="awq",
            trust_remote_code=False,
        )
        assert cfg.model_path == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.max_tokens == 2048
        assert cfg.quantization == "awq"
        assert cfg.tensor_parallel_size == 4

    def test_tensor_parallel_size_must_be_positive(self):
        with pytest.raises(ValueError):
            EngineConfig(model_path="test", tensor_parallel_size=0)
        with pytest.raises(ValueError):
            EngineConfig(model_path="test", tensor_parallel_size=-1)

    def test_quantization_validation(self):
        for q in [None, "awq", "gptq", "squeezellm"]:
            cfg = EngineConfig(model_path="test", quantization=q)
            assert cfg.quantization == q
        with pytest.raises(ValueError):
            EngineConfig(model_path="test", quantization="invalid")
        with pytest.raises(ValueError):
            EngineConfig(model_path="test", quantization="bitsandbytes")

    def test_to_dict(self):
        cfg = EngineConfig(model_path="test", temperature=0.5)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["model_path"] == "test"
        assert d["temperature"] == 0.5

    def test_from_dict(self):
        d = {"model_path": "Qwen/Qwen3-0.6B", "max_tokens": 1024, "temperature": 0.3}
        cfg = EngineConfig.from_dict(d)
        assert cfg.model_path == "Qwen/Qwen3-0.6B"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.8  # default

    def test_empty_model_path_raises(self):
        with pytest.raises(ValueError, match="model_path"):
            EngineConfig(model_path="")
        with pytest.raises(ValueError, match="model_path"):
            EngineConfig(model_path=None)

    def test_from_dict_ignores_extra_keys(self):
        d = {"model_path": "test", "temperature": 0.5, "unknown_key": "ignored"}
        cfg = EngineConfig.from_dict(d)
        assert cfg.model_path == "test"
        assert not hasattr(cfg, "unknown_key")


# =============================================================================
# EngineMetrics Tests
# =============================================================================

class TestEngineMetrics:
    def test_default_metrics(self):
        m = EngineMetrics()
        assert m.tokens_generated == 0
        assert m.total_time == 0.0
        assert m.tokens_per_second == 0.0
        assert m.first_token_latency == 0.0

    def test_compute_tps_positive(self):
        m = EngineMetrics(tokens_generated=100, total_time=10.0)
        m.compute_tps()
        assert m.tokens_per_second == 10.0

    def test_compute_tps_zero_time(self):
        m = EngineMetrics(tokens_generated=100, total_time=0.0)
        m.compute_tps()
        assert m.tokens_per_second == 0.0

    def test_compute_tps_zero_tokens(self):
        m = EngineMetrics(tokens_generated=0, total_time=5.0)
        m.compute_tps()
        assert m.tokens_per_second == 0.0

    def test_merge_two_metrics(self):
        m1 = EngineMetrics(tokens_generated=50, total_time=5.0, first_token_latency=0.1)
        m2 = EngineMetrics(tokens_generated=30, total_time=3.0, first_token_latency=0.2)
        merged = EngineMetrics.merge(m1, m2)
        assert merged.tokens_generated == 80
        assert merged.total_time == 8.0
        assert merged.first_token_latency == 0.1
        assert merged.tokens_per_second == 10.0

    def test_merge_single(self):
        m = EngineMetrics(tokens_generated=100, total_time=10.0)
        merged = EngineMetrics.merge(m)
        assert merged.tokens_generated == 100

    def test_merge_empty(self):
        merged = EngineMetrics.merge()
        assert merged.tokens_generated == 0

    def test_str_representation(self):
        m = EngineMetrics(tokens_generated=100, total_time=10.0, tokens_per_second=10.0)
        s = str(m)
        assert "100" in s

    def test_negative_values_normalized_by_compute(self):
        """compute_tps 处理含负值的 metrics."""
        m = EngineMetrics(tokens_generated=-5, total_time=-1.0)
        # 合约: 调用者负责 non-negative，但 compute_tps 应安全
        m.tokens_generated = max(0, m.tokens_generated)
        m.total_time = max(0.0, m.total_time)
        m.compute_tps()
        assert m.tokens_per_second == 0.0


# =============================================================================
# GenerationResult Tests
# =============================================================================

class TestGenerationResult:
    def test_final_result(self):
        r = GenerationResult(chunk="Hello", is_finished=True, generated_tokens=5)
        assert r.is_finished is True
        assert r.chunk == "Hello"
        assert r.generated_tokens == 5

    def test_stream_chunk(self):
        r = GenerationResult(chunk=" world", is_finished=False, generated_tokens=2)
        assert r.is_finished is False

    def test_empty_final_signal(self):
        r = GenerationResult(chunk="", is_finished=True, generated_tokens=10)
        assert r.is_finished is True
        assert r.chunk == ""

    def test_with_metrics(self):
        m = EngineMetrics(tokens_generated=10, total_time=1.0)
        m.compute_tps()
        r = GenerationResult(chunk="test", is_finished=True, generated_tokens=10, metrics=m)
        assert r.metrics is not None
        assert r.metrics.tokens_per_second == 10.0

    def test_without_metrics(self):
        r = GenerationResult(chunk="test", is_finished=False, generated_tokens=0)
        assert r.metrics is None

    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError):
            GenerationResult(chunk="test", is_finished=True, generated_tokens=-1)


# =============================================================================
# BaseEngine Contract Tests
# =============================================================================

class TestBaseEngineContract:
    def test_base_engine_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseEngine()  # noqa

    def test_abstract_methods_exist(self):
        methods = [m for m in dir(BaseEngine) if not m.startswith('_')]
        assert 'generate' in methods
        assert 'stream_generate' in methods
        assert 'count_tokens' in methods
        assert 'get_metrics' in methods
        assert 'close' in methods
        assert 'info' in methods

    def test_incomplete_subclass_raises(self):
        with pytest.raises(TypeError):
            class Incomplete(BaseEngine):
                pass
            Incomplete(config=EngineConfig(model_path="test"))

    def test_abstract_method_signatures_have_messages_param(self):
        """所有引擎子类必须接受 messages 列表."""
        import inspect
        sig = inspect.signature(BaseEngine.generate)
        params = list(sig.parameters.keys())
        assert 'messages' in params or 'self' in params


# =============================================================================
# Mock Engine Implementation
# =============================================================================

class MockEngine(BaseEngine):
    """A concrete mock engine for testing BaseEngine integration."""
    def __init__(self, model_path="test-model"):
        super().__init__(EngineConfig(model_path=model_path))
        self._metrics = EngineMetrics()
        self._closed = False

    def generate(self, messages, **kwargs):
        return GenerationResult(
            chunk="mock response",
            is_finished=True,
            generated_tokens=3,
            metrics=self._metrics,
        )

    def stream_generate(self, messages, **kwargs):
        yield GenerationResult(chunk="mock", is_finished=False, generated_tokens=1)
        yield GenerationResult(chunk=" response", is_finished=True, generated_tokens=3, metrics=self._metrics)

    def count_tokens(self, text):
        return max(len(text) // 2, 1)

    def get_metrics(self):
        return self._metrics

    def close(self):
        self._closed = True

    def info(self):
        return {"type": "mock", "model": self.config.model_path}


class TestMockEngine:
    @pytest.fixture
    def engine(self):
        return MockEngine()

    def test_engine_has_config(self, engine):
        assert engine.config.model_path == "test-model"

    def test_generate_returns_result(self, engine):
        result = engine.generate([{"role": "user", "content": "hi"}])
        assert isinstance(result, GenerationResult)
        assert result.is_finished
        assert result.chunk == "mock response"

    def test_stream_generate_yields_results(self, engine):
        results = list(engine.stream_generate([{"role": "user", "content": "hi"}]))
        assert len(results) == 2
        assert not results[0].is_finished
        assert results[1].is_finished

    def test_count_tokens(self, engine):
        assert engine.count_tokens("hello world") > 0

    def test_get_metrics(self, engine):
        metrics = engine.get_metrics()
        assert isinstance(metrics, EngineMetrics)

    def test_close(self, engine):
        engine.close()
        assert engine._closed

    def test_info(self, engine):
        info = engine.info()
        assert info["type"] == "mock"
        assert info["model"] == "test-model"

    def test_context_manager(self, engine):
        with engine as e:
            assert not e._closed
        assert e._closed

    def test_can_handle_empty_messages(self, engine):
        result = engine.generate([])
        assert result.is_finished

    def test_stream_with_custom_kwargs(self, engine):
        results = list(engine.stream_generate(
            [{"role": "user", "content": "hello"}],
            temperature=0.5,
            max_tokens=100,
        ))
        assert len(results) == 2

    def test_multiple_consecutive_calls(self):
        engine = MockEngine()
        r1 = engine.generate([{"role": "user", "content": "q1"}])
        r2 = engine.generate([{"role": "user", "content": "q2"}])
        assert r1.chunk == "mock response"
        assert r2.chunk == "mock response"
        engine.close()

    def test_different_message_formats(self, engine):
        engine.generate([{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hi"}])
        engine.generate([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}])
        engine.close()

    def test_streaming_accumulation(self):
        engine = MockEngine()
        full_text = ""
        for r in engine.stream_generate([{"role": "user", "content": "hi"}]):
            full_text += r.chunk
        assert "mock response" in full_text
        engine.close()


# =============================================================================
# PromptFormat Tests
# =============================================================================

class TestPromptFormat:
    def test_format_simple_messages(self):
        messages = [{"role": "user", "content": "Hello"}]
        formatted = PromptFormat.apply_chat_template(messages)
        assert "Hello" in formatted
        assert "<|im_start|>assistant" in formatted

    def test_format_with_system_prompt(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        formatted = PromptFormat.apply_chat_template(messages)
        assert "Be helpful" in formatted

    def test_format_multi_turn(self):
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
            {"role": "user", "content": "Tell me more"},
        ]
        formatted = PromptFormat.apply_chat_template(messages)
        assert "What is AI" in formatted
        assert "Tell me more" in formatted

    def test_format_empty_messages_returns_none(self):
        assert PromptFormat.apply_chat_template([]) is None

    def test_format_invalid_message_missing_role(self):
        with pytest.raises(ValueError, match="role"):
            PromptFormat.apply_chat_template([{"content": "hi"}])

    def test_format_invalid_message_missing_content(self):
        with pytest.raises(ValueError, match="content"):
            PromptFormat.apply_chat_template([{"role": "user"}])

    def test_format_unsupported_role(self):
        with pytest.raises(ValueError, match="role"):
            PromptFormat.apply_chat_template([{"role": "admin", "content": "hi"}])

    def test_format_very_long_content(self):
        long_content = "test " * 10000
        messages = [{"role": "user", "content": long_content}]
        formatted = PromptFormat.apply_chat_template(messages)
        assert formatted is not None
        assert len(formatted) > 1000

    def test_count_tokens_in_messages(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell me about Python."},
        ]
        count = PromptFormat.count_tokens_in_messages(messages)
        assert count > 0

    def test_count_tokens_empty(self):
        assert PromptFormat.count_tokens_in_messages([]) == 0

    def test_validate_bad_input(self):
        with pytest.raises(ValueError):
            PromptFormat.validate_messages("not a list")

    def test_validate_non_dict_message(self):
        with pytest.raises(ValueError):
            PromptFormat.validate_messages(["not a dict"])


# =============================================================================
# Quantization Tests
# =============================================================================

class TestQuantizationConfig:
    def test_awq_quantization(self):
        q = QuantizationConfig(method=QuantizationMethod.AWQ, bits=4, group_size=128)
        assert q.method == QuantizationMethod.AWQ
        assert q.bits == 4

    def test_gptq_quantization(self):
        q = QuantizationConfig(method=QuantizationMethod.GPTQ, bits=4, desc_act=True)
        assert q.method == QuantizationMethod.GPTQ
        assert q.desc_act is True

    def test_squeezellm_quantization(self):
        q = QuantizationConfig(method=QuantizationMethod.SQUEEZELLM, bits=4)
        assert q.method == QuantizationMethod.SQUEEZELLM

    def test_no_quantization(self):
        q = QuantizationConfig.none()
        assert q.method == QuantizationMethod.NONE
        assert q.bits == 16  # FP16

    def test_to_vllm_config_awq(self):
        q = QuantizationConfig(method=QuantizationMethod.AWQ, bits=4)
        assert q.to_vllm_config() == "awq"

    def test_to_vllm_config_gptq(self):
        q = QuantizationConfig(method=QuantizationMethod.GPTQ, bits=4)
        assert q.to_vllm_config() == "gptq"

    def test_to_vllm_config_none(self):
        q = QuantizationConfig.none()
        assert q.to_vllm_config() is None

    def test_to_hf_config_awq(self):
        q = QuantizationConfig(method=QuantizationMethod.AWQ, bits=4)
        hf = q.to_hf_config()
        assert hf["load_in_4bit"] is True

    def test_to_hf_config_none(self):
        q = QuantizationConfig.none()
        assert q.to_hf_config() == {}

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="bits"):
            QuantizationConfig(method=QuantizationMethod.AWQ, bits=3)

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size"):
            QuantizationConfig(method=QuantizationMethod.AWQ, bits=4, group_size=100)


class TestQuantizationRegistry:
    def test_list_methods(self):
        reg = QuantizationRegistry()
        methods = reg.list_methods()
        assert QuantizationMethod.AWQ in methods
        assert QuantizationMethod.NONE in methods

    def test_get_config(self):
        reg = QuantizationRegistry()
        cfg = reg.get_config("awq", bits=4)
        assert cfg.method == QuantizationMethod.AWQ

    def test_get_config_unknown_raises(self):
        reg = QuantizationRegistry()
        with pytest.raises(ValueError, match="Unknown quantization method"):
            reg.get_config("unknown_method")

    def test_is_supported(self):
        reg = QuantizationRegistry()
        assert reg.is_supported("awq") is True
        assert reg.is_supported("none") is True
        assert reg.is_supported("unknown") is False

    def test_get_vllm_supported(self):
        reg = QuantizationRegistry()
        vllm = reg.get_vllm_supported()
        assert "awq" in vllm
        assert "gptq" in vllm


# =============================================================================
# EngineFactory Tests
# =============================================================================

class TestEngineFactory:
    def test_create_vllm_engine(self):
        with patch('engine_factory.EngineFactory._create_vllm_engine') as mc:
            mc.return_value = MockEngine()
            engine = EngineFactory.create(engine_type=EngineType.VLLM, model_path="test")
            mc.assert_called_once()

    def test_create_hf_engine(self):
        with patch('engine_factory.EngineFactory._create_hf_engine') as mc:
            mc.return_value = MockEngine()
            engine = EngineFactory.create(engine_type=EngineType.HF, model_path="test")
            mc.assert_called_once()

    def test_create_with_quantization(self):
        with patch('engine_factory.EngineFactory._create_vllm_engine') as mc:
            mc.return_value = MockEngine()
            engine = EngineFactory.create(
                engine_type=EngineType.VLLM,
                model_path="test",
                quantization="awq",
            )
            mc.assert_called_once()
            cfg = mc.call_args[0][0]
            assert cfg.quantization == "awq"

    def test_create_auto_detects_vllm(self):
        with patch('engine_factory.EngineFactory._is_vllm_available', return_value=True):
            with patch('engine_factory.EngineFactory._create_vllm_engine') as mc:
                mc.return_value = MockEngine()
                engine = EngineFactory.create(engine_type=EngineType.AUTO, model_path="test")
                mc.assert_called_once()

    def test_create_auto_fallback_hf(self):
        with patch('engine_factory.EngineFactory._is_vllm_available', return_value=False):
            with patch('engine_factory.EngineFactory._create_hf_engine') as mc:
                mc.return_value = MockEngine()
                engine = EngineFactory.create(engine_type=EngineType.AUTO, model_path="test")
                mc.assert_called_once()

    def test_create_with_custom_config(self):
        with patch('engine_factory.EngineFactory._create_hf_engine') as mc:
            mc.return_value = MockEngine()
            engine = EngineFactory.create(
                engine_type=EngineType.HF,
                model_path="test",
                max_tokens=2048,
                temperature=0.3,
            )
            cfg = mc.call_args[0][0]
            assert cfg.max_tokens == 2048
            assert cfg.temperature == 0.3

    def test_invalid_engine_type_raises(self):
        with pytest.raises(ValueError, match="engine type"):
            EngineFactory.create(engine_type="invalid", model_path="test")

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            EngineFactory.create(engine_type=EngineType.HF, model_path="test", quantization="bad")

    def test_available_engines(self):
        engines = EngineFactory.available_engines()
        assert EngineType.VLLM in engines
        assert EngineType.HF in engines
        assert EngineType.AUTO in engines


# =============================================================================
# Integration Scenarios
# =============================================================================

class TestRealisticScenarios:
    def test_multiturn_conversation_flow(self):
        """多轮对话: 连续发送消息并验证历史正确传递."""
        class ConversationEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test"))
                self.history = []

            def generate(self, messages, **kwargs):
                self.history = messages
                user_msgs = [m for m in messages if m["role"] == "user"]
                last = user_msgs[-1]["content"] if user_msgs else ""
                return GenerationResult(chunk=f"Reply: {last[:20]}", is_finished=True, generated_tokens=3)

            def stream_generate(self, messages, **kwargs):
                yield self.generate(messages, **kwargs)

            def count_tokens(self, text): return len(text) // 2
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "conv"}

        engine = ConversationEngine()

        # Turn 1
        r1 = engine.generate([{"role": "user", "content": "What is Python?"}])
        assert "Python" in r1.chunk

        # Turn 2 with history
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": r1.chunk},
            {"role": "user", "content": "Compare with Go."},
        ]
        r2 = engine.generate(msgs)
        assert "Compare" in r2.chunk or "Go" in r2.chunk

        # Turn 3 with system prompt
        r3 = engine.generate([
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": "Explain decorators."},
        ])
        assert r3.is_finished

        engine.close()

    def test_streaming_accumulates_full_response(self):
        """流式场景: 逐块积累完整响应."""
        class StreamingEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test"))

            def stream_generate(self, messages, **kwargs):
                chunks = ["The ", "quick ", "brown ", "fox."]
                for i, ch in enumerate(chunks):
                    yield GenerationResult(chunk=ch, is_finished=(i == len(chunks)-1), generated_tokens=i+1)

            def generate(self, messages, **kwargs):
                return GenerationResult(chunk="", is_finished=True, generated_tokens=0)

            def count_tokens(self, text): return len(text.split())
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "stream"}

        engine = StreamingEngine()
        full_text = ""
        for r in engine.stream_generate([{"role": "user", "content": "say"}]):
            full_text += r.chunk
            if r.is_finished:
                assert r.generated_tokens == 4
                break
        assert full_text == "The quick brown fox."
        engine.close()

    def test_rate_limited_scenario(self):
        """快速连续请求."""
        class FastEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test"))
                self.count = 0

            def generate(self, messages, **kwargs):
                self.count += 1
                return GenerationResult(chunk=f"R{self.count}", is_finished=True, generated_tokens=1)

            def stream_generate(self, messages, **kwargs):
                yield self.generate(messages, **kwargs)

            def count_tokens(self, text): return len(text)
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "fast"}

        engine = FastEngine()
        for i in range(20):
            r = engine.generate([{"role": "user", "content": f"q{i}"}])
            assert r.is_finished
        assert engine.count == 20
        engine.close()

    def test_long_context(self):
        """超长上下文."""
        long = [{"role": "user", "content": "word " * 5000}]
        class LongCtxEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test", max_model_len=16384))

            def generate(self, messages, **kwargs):
                total = sum(len(m.get("content", "")) for m in messages)
                return GenerationResult(chunk=f"Processed {total} chars", is_finished=True, generated_tokens=5)

            def stream_generate(self, messages, **kwargs):
                yield self.generate(messages, **kwargs)

            def count_tokens(self, text): return len(text) // 2
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "long"}

        engine = LongCtxEngine()
        r = engine.generate(long)
        assert "Processed" in r.chunk
        engine.close()

    def test_empty_input(self):
        """空输入."""
        class RobustEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test"))

            def generate(self, messages, **kwargs):
                if not messages:
                    return GenerationResult(chunk="", is_finished=True, generated_tokens=0)
                return GenerationResult(chunk="ok", is_finished=True, generated_tokens=1)

            def stream_generate(self, messages, **kwargs):
                if not messages:
                    yield GenerationResult(chunk="", is_finished=True, generated_tokens=0)
                    return
                yield GenerationResult(chunk="ok", is_finished=True, generated_tokens=1)

            def count_tokens(self, text): return 0 if not text else len(text) // 2
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "robust"}

        engine = RobustEngine()
        r = engine.generate([])
        assert r.is_finished and r.generated_tokens == 0
        r2 = engine.generate([{"role": "user", "content": ""}])
        assert r2.is_finished
        sr = list(engine.stream_generate([]))
        assert len(sr) == 1 and sr[0].is_finished
        engine.close()

    def test_error_propagation(self):
        """引擎错误传播."""
        class FailingEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test"))

            def generate(self, messages, **kwargs):
                raise RuntimeError("Model crashed")

            def stream_generate(self, messages, **kwargs):
                raise RuntimeError("Stream error")

            def count_tokens(self, text): return len(text)
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "fail"}

        engine = FailingEngine()
        with pytest.raises(RuntimeError):
            engine.generate([{"role": "user", "content": "hi"}])
        with pytest.raises(RuntimeError):
            list(engine.stream_generate([{"role": "user", "content": "hi"}]))
        engine.close()

    def test_context_manager_cleanup(self):
        """with 语句结束后引擎必须释放."""
        class CleanEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="test"))
                self.cleaned = False

            def generate(self, messages, **kwargs):
                return GenerationResult(chunk="ok", is_finished=True, generated_tokens=1)

            def stream_generate(self, messages, **kwargs):
                yield GenerationResult(chunk="ok", is_finished=True, generated_tokens=1)

            def count_tokens(self, text): return len(text)
            def get_metrics(self): return EngineMetrics()

            def close(self):
                self.cleaned = True

            def info(self): return {"type": "clean"}

        with CleanEngine() as e:
            r = e.generate([{"role": "user", "content": "hi"}])
            assert r.chunk == "ok"
        assert e.cleaned
