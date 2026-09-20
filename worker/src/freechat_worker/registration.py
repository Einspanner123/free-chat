"""Measured single-rank registration and heartbeats for the managed service."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import grpc
import httpx
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import ModelCapability, WorkerCapabilities, WorkerTelemetry
from pydantic import BaseModel, ConfigDict, Field

from freechat_worker.capacity import EngineCapacity
from freechat_worker.telemetry import TelemetryCollector

LOGGER = logging.getLogger(__name__)


class RegistrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scheduler_target: str = Field(pattern=r"^127\.0\.0\.1:[1-9][0-9]{0,4}$")
    node_id: str = Field(min_length=1)
    endpoint: str = Field(pattern=r"^http://127\.0\.0\.1:[1-9][0-9]{0,4}$")
    execution_endpoint: str = Field(pattern=r"^127\.0\.0\.1:[1-9][0-9]{0,4}$")
    interval: float = Field(default=5, gt=0, le=10)


def artifact_identity(directory: Path) -> str:
    """Content-derived identity for the actual local model/tokenizer artifacts."""
    if not directory.is_dir():
        raise ValueError("managed registration requires a resolved local model directory")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
    )
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("model weight artifacts missing")
    if not any(path.name.startswith("tokenizer") for path in files):
        raise ValueError("tokenizer artifacts missing")
    records = []
    for path in files:
        before = path.stat()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("model artifacts changed during identity calculation")
        records.append((path.name, digest))
    return (
        "sha256:" + hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    )


def capabilities(
    config: RegistrationConfig,
    capacity: EngineCapacity,
    model: Any,
    *,
    worker_id: str,
    generation: int,
    revision: str,
) -> WorkerCapabilities:
    name = model.served_model_name
    if not isinstance(name, str) or not name:
        raise ValueError("managed registration requires the normalized native model name")
    if not capacity.gpu_uuid:
        raise ValueError("physical GPU identity unavailable")
    text_config = model.hf_text_config
    attention = (
        "mla"
        if getattr(text_config, "kv_lora_rank", None) is not None
        else (
            "gqa"
            if text_config.num_attention_heads
            != getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
            else "mha"
        )
    )
    return WorkerCapabilities(
        worker_id=worker_id,
        generation=generation,
        endpoint=config.endpoint,
        execution_endpoint=config.execution_endpoint,
        node_id=config.node_id,
        gpu_id=capacity.gpu_uuid,
        gpu_ids=(capacity.gpu_uuid,),
        gpu_name=capacity.gpu_name,
        compute_capability=capacity.compute_capability,
        total_vram_bytes=capacity.total_vram_bytes,
        p2p_domain=config.node_id,
        network_domain="node-local",
        models=(
            ModelCapability(
                model_id=name,
                revision=revision,
                tokenizer_revision=revision,
                architecture="moe" if model.is_moe else "dense",
                is_moe=model.is_moe,
                attention=attention,
                max_context_tokens=capacity.max_context_tokens,
                dtype=str(model.dtype).removeprefix("torch."),
                quantization=model.quantization,
                kv_bytes_per_token=capacity.bytes_per_token,
                kv_admission_bytes_per_token_per_rank=capacity.bytes_per_token,
                kv_block_size_tokens=capacity.block_size_tokens,
                supports_kv_offload=False,
            ),
        ),
    )


class RegistrationLoop:
    def __init__(
        self,
        config: RegistrationConfig,
        caps: WorkerCapabilities,
        capacity: EngineCapacity,
        engine_instance_id: str,
        token: str,
        engine: Any,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if len(token) < 32:
            raise ValueError("worker token requires at least 32 characters")
        self.config, self.caps, self.capacity, self.engine = config, caps, capacity, engine
        self.instance = engine_instance_id
        self._transport = transport
        self.metadata = (("authorization", f"Bearer {token}"),)
        self.headers = {"x-freechat-worker-token": token}
        self.collector = TelemetryCollector(caps, engine_instance_id)
        self.ready = asyncio.Event()

    async def publish(self, stub: Any, telemetry: WorkerTelemetry) -> None:
        async def messages() -> AsyncIterator[control_pb2.WorkerHeartbeat]:
            yield control_pb2.WorkerHeartbeat(
                context=control_pb2.RequestContext(
                    request_id=telemetry.observed_at.isoformat(),
                    tenant_id="system",
                ),
                worker_id=self.caps.worker_id,
                generation=self.caps.generation,
                telemetry_json=telemetry.model_dump_json(),
            )

        # Exhaust the one-message RPC: do not leave a stream/channel owner behind.
        replies = [
            reply
            async for reply in stub.Heartbeat(
                messages(),
                metadata=self.metadata,
                timeout=5,
            )
        ]
        if len(replies) != 1 or replies[0].status != "heartbeat_accepted":
            raise ValueError("heartbeat_not_accepted")

    async def observe(self, client: httpx.AsyncClient) -> WorkerTelemetry:
        health = await client.get(self.config.endpoint + "/health")
        health.raise_for_status()
        response = await client.get(self.config.endpoint + "/metrics", headers=self.headers)
        response.raise_for_status()
        ranks = await self.engine.collective_rpc("freechat_free_memory", timeout=5)
        if len(ranks) != 1 or not isinstance(ranks[0], int) or ranks[0] < 0:
            raise ValueError("invalid_free_memory_observation")
        telemetry = self.collector.collect(
            response.text,
            free_vram_bytes=ranks[0],
            observed_at=datetime.now(UTC),
        )
        return telemetry.model_copy(
            update={
                "kv_admission_available_bytes_per_rank": self.capacity.usable_bytes,
                "admission_accounting": "scheduler_exclusive_gross",
            }
        )

    async def run(self) -> None:
        async with grpc.aio.insecure_channel(self.config.scheduler_target) as channel:
            stub = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
            async with httpx.AsyncClient(
                timeout=5, trust_env=False, transport=self._transport
            ) as client:
                last: WorkerTelemetry | None = None
                try:
                    while True:
                        try:
                            # Retrying same-generation registration preserves current telemetry;
                            # a restored or empty registry can be repopulated after disconnection.
                            result = await stub.Register(
                                control_pb2.WorkerRegistration(
                                    context=control_pb2.RequestContext(
                                        request_id=self.instance,
                                        idempotency_key=self.instance,
                                        tenant_id="system",
                                    ),
                                    worker_id=self.caps.worker_id,
                                    generation=self.caps.generation,
                                    endpoint=self.caps.endpoint,
                                    capabilities_json=self.caps.model_dump_json(),
                                ),
                                metadata=self.metadata,
                                timeout=5,
                            )
                            if result.status != "registered":
                                raise ValueError("registration_not_accepted")
                            last = await self.observe(client)
                            await self.publish(stub, last)
                            if not self.ready.is_set():
                                LOGGER.info(
                                    "worker registered worker=%s generation=%s engine=%s",
                                    self.caps.worker_id,
                                    self.caps.generation,
                                    self.instance,
                                )
                            self.ready.set()
                        except Exception as error:
                            self.ready.clear()
                            LOGGER.warning("worker heartbeat failed: %s", type(error).__name__)
                            # Publish fail-closed health if control remains reachable.
                            if last is not None:
                                with suppress(Exception):
                                    await self.publish(
                                        stub,
                                        last.model_copy(
                                            update={
                                                "healthy": False,
                                                "observed_at": datetime.now(UTC),
                                            }
                                        ),
                                    )
                        await asyncio.sleep(self.config.interval)
                finally:
                    self.ready.clear()
                    if last is not None:
                        try:
                            await self.publish(
                                stub,
                                last.model_copy(
                                    update={
                                        "draining": True,
                                        "healthy": False,
                                        "observed_at": datetime.now(UTC),
                                    }
                                ),
                            )
                        except Exception:
                            LOGGER.warning("final draining heartbeat unavailable")
