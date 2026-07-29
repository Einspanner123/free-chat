"""
gRPC server for LLM inference.

Uses the engine abstraction layer to support multiple backends
(vLLM, HuggingFace) with optional quantization (AWQ, GPTQ).
"""

import json
import os
import signal
import socket
import sys
import urllib
from concurrent import futures
from typing import Iterator

import time

import grpc
import llm_inference_pb2 as pb2
import llm_inference_pb2_grpc as pb2_grpc
from config import config
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from loguru import logger

from engine_factory import EngineFactory, EngineType


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def register_consul(service_id, name, address, port, consul_addr):
    url = f"http://{consul_addr}/v1/agent/service/register"
    payload = {
        "ID": service_id,
        "Name": name,
        "Tags": [name, "api", "v1"],
        "Address": address,
        "Port": port,
        "Check": {
            "GRPC": f"{address}:{port}",
            "GRPCUseTLS": False,
            "Interval": "10s",
            "Timeout": "3s",
            "DeregisterCriticalServiceAfter": "1m",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="PUT", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as _:
            logger.info(f"{name} registration successful")
            return True
    except Exception as e:
        logger.error(f"{name} registration failed: {e}")
        return False


def deregister_consul(service_id, consul_addr):
    url = f"http://{consul_addr}/v1/agent/service/deregister/{service_id}"
    req = urllib.request.Request(url, method="PUT")
    try:
        urllib.request.urlopen(req, timeout=5).close()
        logger.info("Consul deregistration successful")
        return True
    except Exception as e:
        logger.error(f"Consul deregistration failed: {e}")
        return False


class InferencerServiceServicer(pb2_grpc.InferencerServiceServicer):
    def __init__(self):
        logger.info(
            f"Initializing engine: type={config.engineType}, "
            f"model={config.modelName}, "
            f"quantization={config.quantization}"
        )

        # Resolve engine type
        engine_type = EngineType.AUTO
        if config.engineType.lower() == "vllm":
            engine_type = EngineType.VLLM
        elif config.engineType.lower() == "hf":
            engine_type = EngineType.HF

        # Create engine via factory
        self._engine = EngineFactory.create(
            engine_type=engine_type,
            model_path=config.modelName,
            quantization=config.quantization,
            max_tokens=config.maxTokens,
            temperature=config.temperature,
            top_p=config.topP,
            top_k=config.topK,
            repetition_penalty=config.repetitionPenalty,
            gpu_memory_utilization=config.gpuMemoryUtilization,
            tensor_parallel_size=config.tensorParallelSize,
            max_model_len=config.maxModelLen,
        )

        engine_info = self._engine.info()
        logger.info(f"Engine started: {engine_info}")

    def StreamInference(
        self, request_iterator: Iterator[pb2.InferenceRequest], context
    ) -> Iterator[pb2.InferenceResponse]:
        logger.info("Received streaming inference request")

        try:
            session_id = None
            messages: str = ""
            for request in request_iterator:
                session_id = request.session_id
                if request.message:
                    messages += str(request.message)

            logger.info(
                f"Processing: session_id={session_id}, "
                f"message_length={len(messages)}"
            )

            # Parse messages from JSON or create simple user message
            try:
                parsed_messages = json.loads(messages)
                if not isinstance(parsed_messages, list):
                    parsed_messages = [{"role": "user", "content": messages}]
            except (json.JSONDecodeError, TypeError):
                # Add default system prompt if no system message
                parsed_messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": messages},
                ]

            gen_tokens = 0
            start_time = time.time()

            try:
                for result in self._engine.stream_generate(parsed_messages):
                    gen_tokens = result.generated_tokens
                    yield pb2.InferenceResponse(
                        chunk=result.chunk,
                        is_finished=False,
                        error="",
                        generated_tokens=gen_tokens,
                    )

                duration = time.time() - start_time
                tps = gen_tokens / duration if duration > 0 else 0
                logger.info(
                    f"Generation complete: tokens={gen_tokens}, "
                    f"time={duration:.2f}s, tps={tps:.2f}"
                )

                # Send final signal
                yield pb2.InferenceResponse(
                    chunk="",
                    is_finished=True,
                    error="",
                    generated_tokens=gen_tokens,
                )

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield pb2.InferenceResponse(
                    chunk="",
                    is_finished=True,
                    error=str(e),
                    generated_tokens=gen_tokens,
                )

        except Exception as e:
            logger.error(f"Request processing error: {str(e)}")
            yield pb2.InferenceResponse(
                chunk="",
                is_finished=True,
                error=str(e),
                generated_tokens=0,
            )


def serve():
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.maxWorkers))
    pb2_grpc.add_InferencerServiceServicer_to_server(
        InferencerServiceServicer(), server
    )

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    service_name = os.getenv("SERVER_NAME", config.serverName)
    local_ip = get_local_ip()
    service_id = f"{service_name}-{local_ip}-{config.grpcPort}"
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)

    consul_addr = os.getenv("CONSUL_ADDRESS", "localhost:8500")
    register_consul(service_id, service_name, local_ip, config.grpcPort, consul_addr)

    server_address = f"[::]:{config.grpcPort}"
    server.add_insecure_port(server_address)

    logger.info(f"gRPC server starting on {server_address}")
    server.start()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received, stopping gRPC server...")
        deregister_consul(service_id, consul_addr)
        self_engine = getattr(server, '_engine', None)
        done_event = server.stop(grace=5)
        done_event.wait(5)
        logger.info("gRPC server stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
