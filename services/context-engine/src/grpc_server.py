"""
gRPC server for context-engine.

Exposes context optimization as a remote service, so any client
(Go chat-service, other microservices) can build optimized contexts
via gRPC.

Usage: python -m src.grpc_server [--port 8089]
"""

import argparse
import os
import sys
from concurrent import futures

import grpc

# Allow running as script or module
sys.path.insert(0, os.path.dirname(__file__))

from contextengine_pb2 import BuildContextRequest, BuildContextResponse
from contextengine_pb2_grpc import (
    ContextEngineServiceServicer,
    add_ContextEngineServiceServicer_to_server,
)
from pipeline import ContextPipeline, PipelineConfig


class ContextEngineServicer(ContextEngineServiceServicer):
    """gRPC implementation of ContextEngineService."""

    def BuildContext(self, request: BuildContextRequest, context) -> BuildContextResponse:
        """Build an optimized context under a token budget."""
        # Validate strategy
        config = PipelineConfig(
            strategy=request.strategy or "truncation",
            budget=request.budget or 1024,
        )
        pipe = ContextPipeline(config)
        try:
            result = pipe.build_with_metadata(
                request.text,
                _get_tokenizer(),
                query=request.query or "",
            )
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return BuildContextResponse()

        return BuildContextResponse(
            context=result["context"],
            strategy=result["strategy"],
            tokens=result["tokens"],
            compression_ratio=result["compression_ratio"],
        )


_tokenizer = None


def _get_tokenizer():
    """Lazy-load a tokenizer for budget accounting."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-0.6B"),
            trust_remote_code=True,
        )
    return _tokenizer


def serve(port: int = 8089):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ContextEngineServiceServicer_to_server(ContextEngineServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"ContextEngine gRPC server listening on :{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()
    serve(args.port)
