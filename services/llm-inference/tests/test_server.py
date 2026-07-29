"""
Tests for the gRPC inference server.

Uses mocked engine to test the server's request handling,
message parsing, and response formatting without real GPU.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Path setup: add src/ to sys.path for imports
# ---------------------------------------------------------------------------
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Mock torch and transformers BEFORE importing server modules
# ---------------------------------------------------------------------------

class _MockTensor:
    def __init__(self, shape=None):
        self.shape = shape or (1, 15)
    def to(self, device): return self
    def __len__(self): return self.shape[1] if len(self.shape) > 1 else self.shape[0]
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or (self.shape[0] if len(self.shape) == 1 else self.shape[1])
            new_len = max(0, stop - start)
            if len(self.shape) == 1:
                return _MockTensor((new_len,))
            return _MockTensor((self.shape[0], new_len))
        if isinstance(key, int):
            if len(self.shape) == 2:
                return _MockTensor((self.shape[1],))
            return _MockTensor((self.shape[0] - 1,))
        return self

class _MockDevice:
    def __init__(self, name="cpu"): self.type = name
    def __str__(self): return self.type

class _MockGenerate:
    def __call__(self, **kwargs):
        return _MockTensor((1, 15))

class _MockAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs): return _MockAutoModel()
    generate = _MockGenerate()
    def to(self, device): return self

class _MockTokenizer:
    chat_template = "test"
    @staticmethod
    def from_pretrained(*args, **kwargs): return _MockTokenizer()
    def __call__(self, text, **kwargs): return _MockTokenizedOutput()
    def encode(self, text, **kwargs):
        if not text: return []
        return [101, 102, 103, 104, 105]
    def decode(self, tokens, **kwargs): return "mocked response"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not messages: return ""
        parts = [f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>" for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

class _MockTokenizedOutput(dict):
    def __init__(self):
        super().__init__()
        self['input_ids'] = _MockTensor((1, 5))
        self['attention_mask'] = _MockTensor((1, 5))
        self.input_ids = _MockTensor((1, 5))
    def to(self, device): return self

class _MockStreamer:
    def __init__(self, **kwargs):
        self._chunks = iter(["Hello", " world", "!"])
    def __iter__(self): return self
    def __next__(self): return next(self._chunks)

class _MockTorchCuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def empty_cache(): pass

torch_mock = types.ModuleType("torch")
torch_mock.Tensor = _MockTensor
torch_mock.device = _MockDevice
torch_mock.float16 = "float16"
torch_mock.bfloat16 = "bfloat16"
torch_mock.cuda = _MockTorchCuda
torch_mock.no_grad = lambda: _MockTensor()
sys.modules['torch'] = torch_mock

transformers_mock = types.ModuleType("transformers")
transformers_mock.AutoModelForCausalLM = _MockAutoModel
transformers_mock.AutoTokenizer = _MockTokenizer
transformers_mock.TextIteratorStreamer = _MockStreamer
sys.modules['transformers'] = transformers_mock

# ---------------------------------------------------------------------------
# Now import the modules under test
# ---------------------------------------------------------------------------
import grpc
from concurrent import futures

from engine_base import BaseEngine, EngineConfig, GenerationResult, EngineMetrics
from engine_factory import EngineFactory, EngineType


# =============================================================================
# Mock Engine for Server Tests
# =============================================================================

class ServerTestEngine(BaseEngine):
    """A deterministic mock engine for server integration tests."""
    def __init__(self, model_path="test-model", **kwargs):
        super().__init__(EngineConfig(model_path=model_path))
        self._closed = False
        self._call_count = 0

    def generate(self, messages, **kwargs):
        self._call_count += 1
        last_user = "".join(m.get("content", "") for m in messages if m["role"] == "user")
        return GenerationResult(
            chunk=f"Response to: {last_user[:30]}",
            is_finished=True,
            generated_tokens=5,
        )

    def stream_generate(self, messages, **kwargs):
        self._call_count += 1
        last_user = "".join(m.get("content", "") for m in messages if m["role"] == "user")
        content = f"Response to: {last_user[:30]}"
        for i, char in enumerate(content):
            yield GenerationResult(
                chunk=char,
                is_finished=False,
                generated_tokens=i + 1,
            )
        yield GenerationResult(
            chunk="",
            is_finished=True,
            generated_tokens=len(content),
            metrics=EngineMetrics(tokens_generated=len(content), total_time=0.1),
        )

    def count_tokens(self, text): return len(text) // 2
    def get_metrics(self): return EngineMetrics()
    def close(self): self._closed = True
    def info(self): return {"type": "mock", "model": "test"}


# Patch config values
import config
config.config.engineType = "hf"
config.config.modelName = "test-model"
config.config.maxTokens = 512
config.config.temperature = 0.7
config.config.quantization = None

from server import InferencerServiceServicer
import llm_inference_pb2 as pb2


class TestServerInitialization:
    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch('server.EngineFactory.create', return_value=ServerTestEngine(model_path="test")):
            yield

    def test_server_init_creates_engine(self):
        servicer = InferencerServiceServicer()
        assert servicer._engine is not None
        info = servicer._engine.info()
        assert info["type"] == "mock"

    def test_server_engine_can_generate(self):
        servicer = InferencerServiceServicer()
        result = servicer._engine.generate([{"role": "user", "content": "hi"}])
        assert result.is_finished
        assert "Response" in result.chunk


class TestServerStreamInference:
    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch('server.EngineFactory.create', return_value=ServerTestEngine(model_path="test")):
            yield

    @pytest.fixture
    def servicer(self):
        return InferencerServiceServicer()

    def test_simple_message(self, servicer):
        """Send plain text message, verify streaming response."""
        def request_iter():
            yield pb2.InferenceRequest(session_id="s1", message="Hello")

        responses = list(servicer.StreamInference(request_iter(), None))
        assert len(responses) >= 2
        assert responses[-1].is_finished
        full_text = "".join(r.chunk for r in responses if r.chunk)
        assert len(full_text) > 0

    def test_json_message_list(self, servicer):
        """Send JSON message list with system prompt."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Tell me about Go."},
        ]
        def request_iter():
            yield pb2.InferenceRequest(
                session_id="s2",
                message=json.dumps(messages),
            )

        responses = list(servicer.StreamInference(request_iter(), None))
        assert responses[-1].is_finished
        full_text = "".join(r.chunk for r in responses if r.chunk)
        assert len(full_text) > 0

    def test_multiple_request_chunks(self, servicer):
        """Multiple gRPC request chunks merged into one message."""
        def request_iter():
            yield pb2.InferenceRequest(session_id="s3", message="Hello")
            yield pb2.InferenceRequest(session_id="s3", message=" World")

        responses = list(servicer.StreamInference(request_iter(), None))
        assert responses[-1].is_finished
        full_text = "".join(r.chunk for r in responses if r.chunk)
        assert len(full_text) > 0

    def test_empty_message(self, servicer):
        """Empty message returns end signal."""
        def request_iter():
            yield pb2.InferenceRequest(session_id="s4", message="")

        responses = list(servicer.StreamInference(request_iter(), None))
        assert len(responses) >= 1
        assert responses[-1].is_finished

    def test_generated_tokens_count(self, servicer):
        """generated_tokens accumulates correctly."""
        def request_iter():
            yield pb2.InferenceRequest(session_id="s5", message="test")

        responses = list(servicer.StreamInference(request_iter(), None))
        final = responses[-1]
        assert final.generated_tokens > 0


class TestServerErrorHandling:
    @pytest.fixture(autouse=True)
    def _patch_normal_engine(self):
        # Don't patch for the error test, it patches specifically
        yield

    def test_engine_error_returns_graceful_response(self):
        """Engine errors are caught and returned gracefully."""
        class FailingEngine(BaseEngine):
            def __init__(self):
                super().__init__(EngineConfig(model_path="fail"))
            def stream_generate(self, messages, **kwargs):
                raise RuntimeError("Inference crashed")
            def generate(self, messages, **kwargs):
                raise RuntimeError("Inference crashed")
            def count_tokens(self, text): return 0
            def get_metrics(self): return EngineMetrics()
            def close(self): pass
            def info(self): return {"type": "fail"}

        with patch('server.EngineFactory.create', return_value=FailingEngine()):
            servicer = InferencerServiceServicer()
            def request_iter():
                yield pb2.InferenceRequest(session_id="err", message="hello")

            responses = list(servicer.StreamInference(request_iter(), None))
            assert len(responses) >= 1
            final = responses[-1]
            assert final.is_finished

    def test_malformed_json_handled_gracefully(self):
        """Malformed JSON doesn't crash the server."""
        with patch('server.EngineFactory.create', return_value=ServerTestEngine(model_path="test")):
            servicer = InferencerServiceServicer()
            def request_iter():
                yield pb2.InferenceRequest(session_id="s6", message="{bad json!!!}")

            responses = list(servicer.StreamInference(request_iter(), None))
            assert responses[-1].is_finished

    def test_very_long_message(self):
        """Very long messages don't cause crashes."""
        with patch('server.EngineFactory.create', return_value=ServerTestEngine(model_path="test")):
            servicer = InferencerServiceServicer()
            long_msg = "A" * 100000
            def request_iter():
                yield pb2.InferenceRequest(session_id="s7", message=long_msg)

            responses = list(servicer.StreamInference(request_iter(), None))
            assert responses[-1].is_finished


class TestServerConcurrentRequests:
    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch('server.EngineFactory.create', return_value=ServerTestEngine(model_path="test")):
            yield

    def test_concurrent_requests(self):
        """Multiple concurrent requests handled correctly."""
        servicer = InferencerServiceServicer()

        def make_request(msg):
            def iter():
                yield pb2.InferenceRequest(session_id="c", message=msg)
            return list(servicer.StreamInference(iter(), None))

        import threading
        results = [None] * 5
        threads = []

        def run(idx):
            results[idx] = make_request(f"message-{idx}")

        for i in range(5):
            t = threading.Thread(target=run, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for r in results:
            assert r is not None
            assert r[-1].is_finished

    def test_sequential_requests_maintain_state(self):
        """Sequential requests maintain engine state."""
        servicer = InferencerServiceServicer()
        for i in range(3):
            def request_iter(msg=f"req-{i}"):
                yield pb2.InferenceRequest(session_id="seq", message=msg)
            responses = list(servicer.StreamInference(request_iter(), None))
            assert responses[-1].is_finished
