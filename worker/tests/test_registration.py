from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import httpx
import pytest
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_scheduler.grpc_server import WorkerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_worker.capacity import EngineCapacity
from freechat_worker.registration import (
    RegistrationConfig,
    RegistrationLoop,
    artifact_identity,
    capabilities,
)


def capacity() -> EngineCapacity:
    return EngineCapacity(
        num_blocks=100,
        block_size_tokens=16,
        block_bytes=128,
        allocated_bytes=12800,
        max_context_tokens=1024,
        gpu_name="fixture",
        gpu_uuid="GPU-fixture",
        total_vram_bytes=100000,
        compute_capability="8.6",
    )


def config(port: int) -> RegistrationConfig:
    return RegistrationConfig(
        scheduler_target=f"127.0.0.1:{port}",
        node_id="node",
        endpoint="http://127.0.0.1:8000",
        execution_endpoint="127.0.0.1:50052",
        interval=0.02,
    )


def caps(settings: RegistrationConfig) -> Any:
    model = SimpleNamespace(
        served_model_name="qwen",
        hf_text_config=SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=2,
        ),
        is_moe=False,
        dtype="torch.bfloat16",
        quantization=None,
    )
    return capabilities(settings, capacity(), model, worker_id="gpu", generation=1, revision="sha")


def test_artifact_identity_tracks_bytes_not_absolute_path(tmp_path: Path) -> None:
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"fixture weights")
        (directory / "tokenizer.json").write_text("{}")
    original = artifact_identity(tmp_path / "a")
    assert original == artifact_identity(tmp_path / "b")
    (tmp_path / "b/tokenizer.json").write_text('{"changed":true}')
    assert original != artifact_identity(tmp_path / "b")
    with pytest.raises(ValueError, match="weight artifacts"):
        artifact_identity(tmp_path)


def test_capability_uses_physical_identity_and_measured_geometry() -> None:
    result = caps(config(50051))
    assert result.gpu_ids == ("GPU-fixture",)
    assert result.models[0].attention == "gqa"
    assert result.models[0].kv_admission_bytes_per_token_per_rank == 8
    assert result.models[0].kv_block_size_tokens == 16
    assert result.execution_endpoint == "127.0.0.1:50052"


@pytest.mark.parametrize("field", ["scheduler_target", "endpoint", "execution_endpoint"])
def test_unverified_remote_control_transport_is_rejected(field: str) -> None:
    values = config(50051).model_dump()
    values[field] = "http://remote:8000" if field == "endpoint" else "remote:50051"
    with pytest.raises(ValueError):
        RegistrationConfig.model_validate(values)


async def test_authenticated_registration_heartbeat_reconnect_and_drain() -> None:
    registry = InMemoryWorkerRegistry()
    server = grpc.aio.server()
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry, token="t" * 32),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    settings = config(port)
    observation_ok = True

    async def response(request: httpx.Request) -> httpx.Response:
        if not observation_ok:
            return httpx.Response(503)
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.headers["x-freechat-worker-token"] == "t" * 32
        return httpx.Response(
            200,
            text=(
                'vllm:num_requests_running{model_name="qwen",engine="0"} 2\n'
                'vllm:num_requests_waiting{model_name="qwen",engine="0"} 1\n'
                'vllm:kv_cache_usage_perc{model_name="qwen",engine="0"} 0.5\n'
            ),
        )

    async def memory(method: str, timeout: int) -> list[int]:
        assert method == "freechat_free_memory"
        return [4000]

    loop = RegistrationLoop(
        settings,
        caps(settings),
        capacity(),
        "engine",
        "t" * 32,
        SimpleNamespace(collective_rpc=memory),
        transport=httpx.MockTransport(response),
    )
    task = None
    try:
        async with grpc.aio.insecure_channel(settings.scheduler_target) as channel:
            stub = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Register(control_pb2.WorkerRegistration())
            assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        task = asyncio.create_task(loop.run())
        await asyncio.wait_for(loop.ready.wait(), 3)
        telemetry = registry.snapshot()[1][0].telemetry
        assert telemetry.free_vram_bytes == 4000
        assert telemetry.active_requests == 2 and telemetry.queue_depth == 1
        assert telemetry.kv_admission_available_bytes_per_rank == 99 * 128
        assert telemetry.admission_accounting == "scheduler_exclusive_gross"
        assert registry.remove("gpu", 1)
        async with asyncio.timeout(3):
            while not registry.snapshot()[1]:
                await asyncio.sleep(0.01)
        observation_ok = False
        async with asyncio.timeout(3):
            while registry.snapshot()[1][0].telemetry.healthy:
                await asyncio.sleep(0.01)
        assert not loop.ready.is_set()
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await server.stop(None)
    assert registry.snapshot()[1][0].telemetry.draining
