"""Authenticated allocator observations for isolated, non-dispatching probe routes."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import grpc
import httpx
from freechat.control.v1 import control_pb2_grpc
from freechat_contracts import (
    AgentHints,
    RequestProfile,
    RouteDecision,
    WorkerCapabilities,
    WorkerTelemetry,
)
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionStatus
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler.group_runtime import LocalRuntimeEndpoint
from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.preparation import NativePreparer
from freechat_scheduler.registry import InMemoryWorkerRegistry, WorkerSnapshot
from freechat_scheduler.request_execution import LocalGrpcExecutionDriver, RequestExecutionConfig
from freechat_scheduler.scheduler import Scheduler
from freechat_worker.capacity import EngineCapacity
from freechat_worker.telemetry import TelemetryCollector, heartbeat
from pydantic import BaseModel, ConfigDict, SecretStr


class RuntimeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    generation: int
    engine_instance_id: str
    observed_at: datetime
    capabilities: WorkerCapabilities
    capacity: EngineCapacity

    def validate_scope(self, caps: WorkerCapabilities, engine: str) -> None:
        if (
            self.observed_at.tzinfo is None
            or not 0 <= (datetime.now(UTC) - self.observed_at).total_seconds() <= 10
        ):
            raise ValueError("runtime_observation_not_fresh")
        if (
            self.worker_id != caps.worker_id
            or self.generation != caps.generation
            or self.engine_instance_id != engine
            or self.capabilities != caps
        ):
            raise ValueError("runtime_identity_mismatch")
        if len(caps.models) != 1:
            raise ValueError("runtime_requires_single_model")
        model, capacity = caps.models[0], self.capacity
        if (
            model.tensor_parallel_size != 1
            or model.pipeline_parallel_size != 1
            or capacity.gpu_uuid != caps.gpu_id
            or not capacity.gpu_uuid
            or capacity.gpu_name != caps.gpu_name
            or capacity.compute_capability != caps.compute_capability
            or capacity.total_vram_bytes != caps.total_vram_bytes
            or capacity.bytes_per_token != model.kv_admission_bytes_per_token_per_rank
            or capacity.block_size_tokens != model.kv_block_size_tokens
            or capacity.max_context_tokens != model.max_context_tokens
        ):
            raise ValueError("runtime_geometry_mismatch")

    def budgeted(self, telemetry: WorkerTelemetry) -> WorkerTelemetry:
        self.validate_scope(self.capabilities, self.engine_instance_id)
        if (
            telemetry.worker_id != self.worker_id
            or telemetry.generation != self.generation
            or telemetry.engine_instance_id != self.engine_instance_id
            or telemetry.active_requests
            or telemetry.queue_depth
        ):
            raise ValueError("runtime_requires_matching_idle_observation")
        return telemetry.model_copy(
            update={
                "kv_admission_available_bytes_per_rank": self.capacity.usable_bytes,
                # This is an isolated observation, not ownership of a live scheduler pool.
                "admission_accounting": "unmanaged_observation",
            }
        )


async def read_runtime(
    client: httpx.AsyncClient,
    caps: WorkerCapabilities,
    engine: str,
) -> RuntimeObservation:
    response = await client.get(caps.endpoint.rstrip("/") + "/freechat/runtime")
    response.raise_for_status()
    try:
        observation = RuntimeObservation.model_validate_json(response.content)
    except ValueError as error:
        raise ValueError("runtime_observation_invalid") from error
    observation.validate_scope(caps, engine)
    return observation


async def route_once(
    caps: WorkerCapabilities,
    telemetry: WorkerTelemetry,
    request: RequestProfile,
) -> RouteDecision:
    """A fresh private ledger per case; never dispatch or fabricate a release receipt."""
    registry = InMemoryWorkerRegistry()
    await registry.register(caps, telemetry)
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(Scheduler(registry), LeaseBook()),
        server,
    )
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry),
        server,
    )
    client = None
    try:
        port = server.add_insecure_port("127.0.0.1:0")
        if not port:
            raise RuntimeError("probe_control_port_unavailable")
        await server.start()
        target = f"127.0.0.1:{port}"
        client = GrpcSchedulerClient(target)
        async with grpc.aio.insecure_channel(target) as channel:
            stub = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
            if await heartbeat(stub, telemetry) != "heartbeat_accepted":
                raise ValueError("probe_heartbeat_not_accepted")
            return await client.route(request)
    finally:
        if client is not None:
            await client.aclose()
        await server.stop(None)


class ProbeSession:
    """Serial operator probe, not another scheduler or a concurrent workload runner."""

    def __init__(self, caps: WorkerCapabilities, engine: str, token: str) -> None:
        if len(token) < 32:
            raise ValueError("FREECHAT_WORKER_TOKEN requires at least 32 characters")
        self.caps, self.engine = caps, engine
        self.initial: RuntimeObservation | None = None
        self.preparer = NativePreparer(token)
        if caps.execution_endpoint is None:
            raise ValueError("runtime_execution_endpoint_required")
        self.driver = LocalGrpcExecutionDriver(
            RequestExecutionConfig(
                mode="local-contract",
                endpoints={
                    caps.worker_id: LocalRuntimeEndpoint(
                        address=caps.execution_endpoint,
                        token=SecretStr(token),
                    )
                },
            )
        )

    async def observe(self, client: httpx.AsyncClient) -> RuntimeObservation:
        current = await read_runtime(client, self.caps, self.engine)
        if self.initial is not None and self.initial.capacity != current.capacity:
            raise ValueError("runtime_capacity_changed")
        self.initial = current
        return current

    async def infer(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        current = await self.observe(client)
        metrics = await client.get(self.caps.endpoint.rstrip("/") + "/metrics")
        metrics.raise_for_status()
        observed = current.budgeted(
            TelemetryCollector(self.caps, self.engine).collect(
                metrics.text,
                free_vram_bytes=0,
                observed_at=datetime.now(UTC),
            )
        )
        request = RequestProfile(
            tenant_id="calibration-probe",
            local_node_id=self.caps.node_id,
            model_id=self.caps.models[0].model_id,
            input_tokens=1,
            output_tokens=body["max_tokens"],
            native_protocol="/v1/chat/completions",
            native_request_json=json.dumps(body),
            hints=AgentHints(harness_id="calibration", task_id="probe", agent_id="probe"),
        )
        prepared = await self.preparer.prepare(
            request,
            (WorkerSnapshot(self.caps, observed),),
        )
        preparation = prepared.get(self.caps.worker_id)
        if preparation is None:
            raise ValueError("probe_preparation_rejected")
        budget, capacity = preparation.budget, current.capacity
        total = budget.input_tokens + budget.output_tokens
        required = (
            (total + capacity.block_size_tokens - 1)
            // capacity.block_size_tokens
            * capacity.block_bytes
        )
        if (
            budget.expires_at <= time.time()
            or budget.output_tokens != body["max_tokens"]
            or total > capacity.max_context_tokens
            or required > capacity.usable_bytes
        ):
            raise ValueError("probe_preparation_exceeds_scope")
        command = ExecutionCommand(
            tenant_id=request.tenant_id,
            request_id=request.request_id,
            decision_id=str(uuid4()),
            worker_id=self.caps.worker_id,
            worker_generation=self.caps.generation,
            engine_instance_id=self.engine,
            action=ExecutionAction.QUERY,
        )
        headers = {
            "x-freechat-internal-tenant": command.tenant_id,
            "x-freechat-internal-request-id": command.request_id,
            "x-freechat-internal-decision-id": command.decision_id,
            "x-freechat-internal-worker-generation": str(command.worker_generation),
            "x-freechat-internal-engine-instance-id": command.engine_instance_id,
            "x-freechat-internal-preparation": preparation.preparation_id,
            "x-freechat-internal-reserved-kv-bytes": str(required),
        }
        try:
            response = await client.post(
                self.caps.endpoint.rstrip("/") + "/v1/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            async with asyncio.timeout(15):
                while True:
                    receipt = await self.driver.observe(command)
                    if receipt.releasable:
                        if receipt.status != ExecutionStatus.COMPLETED:
                            raise ValueError("probe_execution_not_completed")
                        break
                    await asyncio.sleep(0.05)
            await self.observe(client)
            return dict(response.json())
        except BaseException:
            # Stop this probe on uncertainty; an abort attempt is not a release proof.
            with suppress(Exception):
                await self.driver.observe(
                    command.model_copy(update={"action": ExecutionAction.ABORT})
                )
            raise


async def observe_free_vram(caps: WorkerCapabilities, host: str | None) -> int:
    """Diagnostic free VRAM only; never the allocator admission budget."""
    gpu_id = caps.gpu_id
    with suppress(ValueError):
        # PyTorch reports a bare UUID; nvidia-smi accepts its GPU-prefixed form.
        gpu_id = f"GPU-{UUID(gpu_id)}"
    command = [] if host is None else ["ssh", host]
    process = await asyncio.create_subprocess_exec(
        *command,
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    if process.returncode:
        raise RuntimeError("GPU observation failed")
    value = int(stdout.decode().strip()) * 1024**2
    if not 0 <= value <= caps.total_vram_bytes:
        raise ValueError("GPU free memory observation out of bounds")
    return value
