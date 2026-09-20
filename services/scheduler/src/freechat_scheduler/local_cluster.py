"""Local loopback gRPC validation of synthetic 3x4 GPU resource groups.

No SSH, CUDA, inference endpoint, external etcd or cluster mutation is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import Any

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import ModelCapability, WorkerCapabilities, WorkerTelemetry
from freechat_control_store import InMemoryStore

from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.parallel import DeviceLink
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.resource_groups import (
    Device,
    GroupController,
    GroupSpec,
    GroupState,
    Inventory,
)
from freechat_scheduler.scheduler import RoutingStrategy, Scheduler


async def validate_layout(tp: int) -> dict[str, Any]:
    devices = tuple(
        Device(
            gpu_id=f"sim-node{node}/gpu{gpu}",
            node_id=f"sim-node{node}",
            memory_bytes=80 * 1024**3,
            compute_capability="9.0",
        )
        for node in range(3)
        for gpu in range(4)
    )
    links = tuple(
        DeviceLink(f"sim-node{node}/gpu{a}", f"sim-node{node}/gpu{b}", True, True, True)
        for node in range(3)
        for a, b in combinations(range(4), 2)
    )
    controller = GroupController(InMemoryStore(), Inventory(generation=1, devices=devices), links)
    registry = InMemoryWorkerRegistry()
    view = (datetime.now(UTC), await controller.snapshot())
    scheduler = Scheduler(
        registry, strategy=RoutingStrategy.ROUND_ROBIN, group_snapshot=lambda: view
    )
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(scheduler, LeaseBook()), server
    )
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    worker = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
    router = control_pb2_grpc.SchedulerServiceStub(channel)  # type: ignore[no-untyped-call]
    expected = set()
    routed = []
    try:
        for node in range(3):
            for start in range(0, 4, tp):
                group_id = f"group-{node}-{start}"
                spec = GroupSpec(
                    group_id=group_id,
                    worker_id=group_id,
                    gpu_ids=tuple(f"sim-node{node}/gpu{gpu}" for gpu in range(start, start + tp)),
                    model_id="synthetic-model",
                    model_revision="synthetic-revision",
                    tensor_parallel_size=tp,
                    memory_per_gpu_bytes=1024,
                    evidence_reference="SYNTHETIC-NOT-HARDWARE-EVIDENCE",
                    evidence_expires_at=datetime.now(UTC) + timedelta(hours=1),
                    communication_fraction=0.1,
                    measured_speedup=1.2,
                )
                lease = await controller.reserve(spec, "local-validator")
                await controller.transition(
                    group_id, lease.generation, "local-validator", GroupState.STARTING
                )
                model = ModelCapability(
                    model_id=spec.model_id,
                    revision=spec.model_revision,
                    tokenizer_revision="synthetic",
                    architecture="dense",
                    attention="gqa",
                    max_context_tokens=4096,
                    dtype="bfloat16",
                    tensor_parallel_size=tp,
                    kv_admission_bytes_per_token_per_rank=16_384,
                    kv_block_size_tokens=16,
                )
                caps = WorkerCapabilities(
                    worker_id=group_id,
                    generation=lease.generation,
                    endpoint="http://127.0.0.1:9/non-serving-fixture",
                    node_id=f"sim-node{node}",
                    gpu_id=str(start),
                    gpu_name="SIMULATED H100",
                    compute_capability="9.0",
                    total_vram_bytes=80 * 1024**3,
                    p2p_domain=f"sim-node{node}",
                    network_domain="unknown",
                    models=(model,),
                    gpu_ids=spec.gpu_ids,
                    resource_group_id=group_id,
                    resource_group_generation=lease.generation,
                )
                await worker.Register(
                    control_pb2.WorkerRegistration(
                        worker_id=group_id,
                        generation=lease.generation,
                        endpoint=caps.endpoint,
                        capabilities_json=caps.model_dump_json(),
                    )
                )
                telemetry = WorkerTelemetry(
                    worker_id=group_id,
                    generation=lease.generation,
                    free_vram_bytes=1024**3,
                    kv_admission_available_bytes_per_rank=1024**3,
                    engine_instance_id=group_id,
                )

                async def heartbeat(
                    item: WorkerTelemetry = telemetry,
                ) -> AsyncIterator[control_pb2.WorkerHeartbeat]:
                    yield control_pb2.WorkerHeartbeat(
                        worker_id=item.worker_id,
                        generation=item.generation,
                        telemetry_json=item.model_dump_json(),
                    )

                ack = await worker.Heartbeat(heartbeat()).read()
                assert ack.status == "heartbeat_accepted"
                await controller.transition(
                    group_id,
                    lease.generation,
                    "local-validator",
                    GroupState.READY,
                    engine_instance_id=group_id,
                    acknowledgement="simulated-ready",
                )
                expected.add(group_id)
        view = (datetime.now(UTC), await controller.snapshot())
        for index in range(len(expected) * 2):
            route = await router.Route(
                control_pb2.RouteRequest(
                    context=control_pb2.RequestContext(
                        request_id=f"r{index}", tenant_id="local-fixture"
                    ),
                    hints=control_pb2.AgentHints(
                        harness_id="local-fixture",
                        task_id="task",
                        agent_id="agent",
                        source="explicit",
                        confidence=1,
                        allow_remote_worker=True,
                    ),
                    model="synthetic-model",
                    input_tokens=32,
                    output_tokens=16,
                )
            )
            routed.append(route.worker_id)
            assert route.worker_generation == view[1].groups[route.worker_id].generation
            await router.Release(
                control_pb2.LeaseRequest(
                    context=control_pb2.RequestContext(tenant_id="local-fixture"),
                    decision_id=route.decision_id,
                    worker_id=route.worker_id,
                    worker_generation=route.worker_generation,
                )
            )
        assert set(routed) == expected
        return {
            "tp": tp,
            "resource_groups": len(expected),
            "gpu_count": 12,
            "routed_requests": len(routed),
            "unique_workers": sorted(set(routed)),
            "evidence_level": "LOCAL_SIMULATION_ONLY",
            "hardware_verified": False,
        }
    finally:
        await channel.close()
        await server.stop(None)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = {
        "scope": "local loopback, synthetic inventory, no model execution",
        "layouts": [await validate_layout(tp) for tp in (1, 2, 4)],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())
