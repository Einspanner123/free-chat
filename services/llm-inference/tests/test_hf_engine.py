"""
Tests for HFEngine that mock torch and transformers.
"""

import sys
import os
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add src/ to path
_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# ---------------------------------------------------------------------------
# Mock torch and transformers BEFORE any src imports
# ---------------------------------------------------------------------------

class _MockTensor:
    def __init__(self, shape=None):
        self.shape = shape or (1, 15)
    def to(self, device):
        return self
    def __len__(self):
        """Return the last dimension size, like real tensor indexing."""
        return self.shape[1] if len(self.shape) > 1 else self.shape[0]
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Slicing 1D tensor: (15,)[5:] → (10,)
            start = key.start or 0
            stop = key.stop or (self.shape[0] if len(self.shape) == 1 else self.shape[1])
            new_len = max(0, stop - start)
            if len(self.shape) == 1:
                return _MockTensor((new_len,))
            return _MockTensor((self.shape[0], new_len))
        if isinstance(key, int):
            # Indexing 2D tensor: (1,15)[0] → (15,)
            if len(self.shape) == 2:
                return _MockTensor((self.shape[1],))
            return _MockTensor((self.shape[0] - 1,))
        return self
    @property
    def numel(self):
        return self.shape[0] * self.shape[1] if len(self.shape) > 1 else self.shape[0]


class _MockDevice:
    def __init__(self, name="cpu"):
        self.type = name
    def __str__(self):
        return self.type
    def __repr__(self):
        return self.type


class _MockGenerate:
    def __call__(self, **kwargs):
        if 'streamer' in kwargs:
            # Streaming mode: streamer handles it
            return _MockTensor((1, 15))
        return _MockTensor((1, 15))


class _MockAutoModel:
    dtype = "float16"
    @staticmethod
    def from_pretrained(*args, **kwargs):
        m = _MockAutoModel()
        m.generate = _MockGenerate()
        m.device = _MockDevice("cpu")
        return m
    def to(self, device):
        return self


class _MockTokenizer:
    chat_template = "test_template"
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _MockTokenizer()
    
    def __call__(self, text, **kwargs):
        """Support tokenizer(text, return_tensors='pt') pattern."""
        return _MockTokenizedOutput()
    
    def encode(self, text, **kwargs):
        if not text:
            return []
        return [101, 102, 103, 104, 105]
    
    def decode(self, tokens, **kwargs):
        return "mocked response"
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not messages:
            return ""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


class _MockTokenizedOutput(dict):
    """Mock object returned by tokenizer.__call__().
    Must be dict-like to support **unpacking.
    """
    def __init__(self):
        super().__init__()
        self['input_ids'] = _MockTensor((1, 5))
        self['attention_mask'] = _MockTensor((1, 5))
        self.input_ids = _MockTensor((1, 5))
    
    def to(self, device):
        return self


class _MockStreamer:
    def __init__(self, **kwargs):
        self._chunks = iter(["Hello ", "world", "!"])
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._chunks)


class _MockTorchCuda:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def empty_cache():
        pass


class _MockTorch:
    Tensor = _MockTensor
    device = _MockDevice
    float16 = "float16"
    bfloat16 = "bfloat16"
    cuda = _MockTorchCuda
    @staticmethod
    def device(name):
        return _MockDevice(name)
    @staticmethod
    def no_grad():
        class _CM:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _CM()


class _MockTransformers:
    AutoModelForCausalLM = _MockAutoModel
    AutoTokenizer = _MockTokenizer
    TextIteratorStreamer = _MockStreamer


# Register mocks
torch_mock = types.ModuleType("torch")
torch_mock.Tensor = _MockTensor
torch_mock.device = _MockDevice
torch_mock.float16 = "float16"
torch_mock.bfloat16 = "bfloat16"
torch_mock.cuda = _MockTorchCuda
torch_mock.no_grad = _MockTorch.no_grad

transformers_mock = types.ModuleType("transformers")
transformers_mock.AutoModelForCausalLM = _MockAutoModel
transformers_mock.AutoTokenizer = _MockTokenizer
transformers_mock.TextIteratorStreamer = _MockStreamer

sys.modules['torch'] = torch_mock
sys.modules['transformers'] = transformers_mock

# Also mock the threading imports used by HFEngine
import threading
original_thread = threading.Thread

# Now import the source modules
from hf_engine import HFEngine
from engine_base import EngineConfig, GenerationResult, EngineMetrics, BaseEngine


class TestHFEngineInit:
    """HFEngine initialization tests."""

    def test_init_basic(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        assert engine.config.model_path == "test-model"
        assert str(engine.device) == "cpu"
        assert not engine._closed
        engine.close()

    def test_init_has_config(self):
        config = EngineConfig(model_path="custom-model", max_tokens=1024)
        engine = HFEngine(config=config, device="cpu")
        assert engine.config.max_tokens == 1024
        engine.close()

    def test_info_returns_metadata(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        info = engine.info()
        assert info["type"] == "hf"
        assert info["model"] == "test-model"
        assert info["device"] == "cpu"
        assert "max_tokens" in info
        engine.close()

    def test_close_releases_resources(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        engine.close()
        assert engine._closed


class TestHFEngineTokenCounting:
    """HFEngine token counting tests."""

    def test_count_tokens_positive(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        count = engine.count_tokens("hello world")
        assert count == 5  # mock returns 5 tokens
        engine.close()

    def test_count_tokens_empty(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        assert engine.count_tokens("") == 0
        engine.close()

    def test_count_tokens_unicode(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        count = engine.count_tokens("你好世界")
        assert count == 5  # mock returns 5 tokens regardless
        engine.close()


class TestHFEngineGenerate:
    """HFEngine generate (non-streaming) tests."""

    @pytest.fixture
    def engine(self):
        return HFEngine(model_path="test-model", device="cpu")

    def test_generate_returns_result(self, engine):
        result = engine.generate([{"role": "user", "content": "Hello"}])
        assert isinstance(result, GenerationResult)
        assert result.is_finished
        assert result.generated_tokens > 0

    def test_generate_with_system_prompt(self, engine):
        result = engine.generate([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell me about Python."},
        ])
        assert result.is_finished

    def test_generate_multi_turn(self, engine):
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
            {"role": "user", "content": "Tell me more"},
        ]
        result = engine.generate(messages)
        assert result.is_finished

    def test_generate_empty_messages(self, engine):
        result = engine.generate([])
        assert result.is_finished

    def test_generate_with_custom_kwargs(self, engine):
        result = engine.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert result.is_finished

    def test_generate_updates_metrics(self, engine):
        engine.generate([{"role": "user", "content": "hi"}])
        metrics = engine.get_metrics()
        assert metrics.tokens_generated > 0

    def test_generate_multiple_calls_accumulate_metrics(self, engine):
        engine.generate([{"role": "user", "content": "hi"}])
        engine.generate([{"role": "user", "content": "hello again"}])
        metrics = engine.get_metrics()
        assert metrics.tokens_generated > 0

    def test_engine_works_as_context_manager(self):
        with HFEngine(model_path="test-model", device="cpu") as engine:
            result = engine.generate([{"role": "user", "content": "test"}])
            assert result.is_finished
        assert engine._closed


class TestHFEngineStreamGenerate:
    """HFEngine streaming generation tests."""

    @pytest.fixture
    def engine(self):
        return HFEngine(model_path="test-model", device="cpu")

    def test_stream_generate_yields_results(self, engine):
        results = list(engine.stream_generate([{"role": "user", "content": "Hello"}]))
        assert len(results) >= 2  # at least one chunk + final
        assert results[-1].is_finished

    def test_stream_generate_content(self, engine):
        results = list(engine.stream_generate([{"role": "user", "content": "say something"}]))
        full_text = "".join(r.chunk for r in results)
        assert len(full_text) > 0

    def test_stream_generate_multiple_chunks(self, engine):
        results = list(engine.stream_generate([{"role": "user", "content": "tell me a story"}]))
        non_final = [r for r in results if not r.is_finished]
        assert len(non_final) > 0

    def test_stream_generate_updates_metrics(self, engine):
        list(engine.stream_generate([{"role": "user", "content": "hi"}]))
        metrics = engine.get_metrics()
        assert metrics.tokens_generated > 0

    def test_stream_generate_final_has_metrics(self, engine):
        results = list(engine.stream_generate([{"role": "user", "content": "hi"}]))
        final = results[-1]
        assert final.is_finished
        assert final.metrics is not None or final.generated_tokens > 0

    def test_stream_generate_multiple_calls(self, engine):
        r1 = list(engine.stream_generate([{"role": "user", "content": "first"}]))
        r2 = list(engine.stream_generate([{"role": "user", "content": "second"}]))
        assert r1[-1].is_finished
        assert r2[-1].is_finished

    def test_stream_generate_empty_messages(self, engine):
        results = list(engine.stream_generate([]))
        assert len(results) >= 1
        assert results[-1].is_finished


class TestHFEngineEdgeCases:
    """HFEngine edge cases."""

    def test_very_long_input(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        long_text = "Hello " * 1000
        result = engine.generate([{"role": "user", "content": long_text}])
        assert result.is_finished
        engine.close()

    def test_special_characters(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        result = engine.generate([{"role": "user", "content": "!@#$%^&*()_+{}[]|\":;<>,./?"}])
        assert result.is_finished
        engine.close()

    def test_code_in_message(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        code = "```python\ndef hello():\n    print('world')\n```"
        result = engine.generate([{"role": "user", "content": code}])
        assert result.is_finished
        engine.close()

    def test_json_in_message(self):
        engine = HFEngine(model_path="test-model", device="cpu")
        json_content = '{"key": "value", "nested": {"a": 1}}'
        result = engine.generate([{"role": "user", "content": json_content}])
        assert result.is_finished
        engine.close()

    def test_generate_handles_role_validation(self):
        """引擎应能处理各种合法 role."""
        engine = HFEngine(model_path="test-model", device="cpu")
        for role in ["system", "user", "assistant"]:
            result = engine.generate([{"role": role, "content": "test"}])
            assert result.is_finished
        engine.close()
