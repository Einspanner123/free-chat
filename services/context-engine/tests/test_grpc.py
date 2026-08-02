"""
Tests for context-engine gRPC server.
Uses an in-process server; no network dependency.
"""

import os
import sys
from concurrent import futures

# Clear proxy env vars (local gRPC connections must not go through proxy)
for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(k, None)

import grpc
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from contextengine_pb2 import BuildContextRequest
from contextengine_pb2_grpc import (
    ContextEngineServiceStub,
    add_ContextEngineServiceServicer_to_server,
)
from grpc_server import ContextEngineServicer


@pytest.fixture
def stub():
    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)
        def decode(self, tokens, skip_special_tokens=True):
            return "".join(tokens)

    import grpc_server
    grpc_server._tokenizer = CharTokenizer()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_ContextEngineServiceServicer_to_server(ContextEngineServicer(), server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield ContextEngineServiceStub(channel)
    server.stop(0)
    grpc_server._tokenizer = None


class TestContextEngineGRPC:
    def test_build_truncation(self, stub):
        resp = stub.BuildContext(BuildContextRequest(
            text="A" * 200, strategy="truncation", budget=50, query=""
        ))
        assert resp.tokens <= 55
        assert resp.strategy == "truncation"

    def test_build_topic(self, stub):
        resp = stub.BuildContext(BuildContextRequest(
            text="Paragraph 1: apple apple\nParagraph 2: banana",
            strategy="project_topic", budget=30, query="apple",
        ))
        assert "apple" in resp.context

    def test_build_attention_sink(self, stub):
        resp = stub.BuildContext(BuildContextRequest(
            text="apple content here apple",
            strategy="attention_sink", budget=30, query="apple",
        ))
        assert resp.context.startswith("\n\n")  # sink token

    def test_build_bm25(self, stub):
        resp = stub.BuildContext(BuildContextRequest(
            text="Paragraph 1: Apple makes phones.\nParagraph 2: Banana.",
            strategy="bm25_top1", budget=50, query="Apple makes phones",
        ))
        assert "Apple makes" in resp.context

    def test_invalid_strategy(self, stub):
        with pytest.raises(grpc.RpcError):
            stub.BuildContext(BuildContextRequest(
                text="hello", strategy="invalid", budget=10, query="",
            ))

    def test_empty_text(self, stub):
        resp = stub.BuildContext(BuildContextRequest(
            text="", strategy="truncation", budget=10, query="",
        ))
        assert resp.context == ""

    def test_compression_ratio_present(self, stub):
        resp = stub.BuildContext(BuildContextRequest(
            text="A" * 200, strategy="truncation", budget=50, query="",
        ))
        assert resp.compression_ratio > 0
