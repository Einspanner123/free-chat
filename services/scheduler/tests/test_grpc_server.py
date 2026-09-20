import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import grpc
import pytest
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import (
    AgentHints,
    ModelCapability,
    RequestProfile,
    RouteDecision,
    WorkerCapabilities,
    WorkerTelemetry,
)
from freechat_contracts.preparation import PreparedAdmission
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_ledger import RequestState
from freechat_scheduler.scheduler import Scheduler


class RecordingScheduler(Scheduler):
    def __init__(self, registry: InMemoryWorkerRegistry) -> None:
        super().__init__(registry)
        self.received: list[RequestProfile] = []

    def route(
        self,
        request: RequestProfile,
        *,
        reserved: dict[str, tuple[int, int]] | None = None,
        prepared: dict[str, PreparedAdmission] | None = None,
    ) -> RouteDecision:
        self.received.append(request)
        return super().route(request, reserved=reserved, prepared=prepared)


def test_worker_registration_and_route_over_grpc() -> None:
    async def scenario() -> None:
        registry = InMemoryWorkerRegistry()
        scheduler = RecordingScheduler(registry)
        ledger = LeaseBook()
        server = grpc.aio.server()
        control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
            SchedulerGrpcService(scheduler, ledger),
            server,
        )
        control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
            WorkerGrpcService(registry),
            server,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        try:
            worker_stub = control_pb2_grpc.WorkerControlServiceStub(  # type: ignore[no-untyped-call]
                channel
            )
            scheduler_stub = control_pb2_grpc.SchedulerServiceStub(  # type: ignore[no-untyped-call]
                channel
            )
            capabilities = WorkerCapabilities(
                worker_id="ross-a6000",
                generation=1,
                endpoint="http://ross-worker:8000",
                node_id="ross",
                gpu_id="0",
                gpu_name="NVIDIA RTX A6000",
                compute_capability="8.6",
                total_vram_bytes=48 * 1024**3,
                p2p_domain="ross-0",
                network_domain="weak-lan",
                models=(
                    ModelCapability(
                        model_id="Qwen/Qwen2.5-0.5B-Instruct",
                        revision="model-sha",
                        tokenizer_revision="tokenizer-sha",
                        architecture="dense",
                        attention="gqa",
                        max_context_tokens=32_768,
                        dtype="bfloat16",
                        supports_kv_offload=True,
                        kv_bytes_per_token=16_384,
                        kv_admission_bytes_per_token_per_rank=16_384,
                        kv_block_size_tokens=16,
                    ),
                ),
            )
            registered = await worker_stub.Register(
                control_pb2.WorkerRegistration(
                    context=control_pb2.RequestContext(
                        request_id="register",
                        idempotency_key="register-1",
                        tenant_id="system",
                    ),
                    worker_id=capabilities.worker_id,
                    generation=capabilities.generation,
                    endpoint=capabilities.endpoint,
                    capabilities_json=capabilities.model_dump_json(),
                )
            )
            assert registered.status == "registered"

            async def initial_heartbeat() -> AsyncIterator[control_pb2.WorkerHeartbeat]:
                yield control_pb2.WorkerHeartbeat(
                    worker_id=capabilities.worker_id,
                    generation=capabilities.generation,
                    telemetry_json=WorkerTelemetry(
                        worker_id=capabilities.worker_id,
                        generation=capabilities.generation,
                        free_vram_bytes=0,
                        kv_admission_available_bytes_per_rank=1024**3,
                    ).model_dump_json(),
                )

            assert (
                await worker_stub.Heartbeat(initial_heartbeat()).read()
            ).status == "heartbeat_accepted"
            route = await scheduler_stub.Route(
                control_pb2.RouteRequest(
                    context=control_pb2.RequestContext(
                        request_id="request",
                        idempotency_key="request-1",
                        tenant_id="tenant-a",
                    ),
                    hints=control_pb2.AgentHints(
                        harness_id="langgraph",
                        task_id="task",
                        agent_id="agent",
                        lifecycle="active",
                        prefix_scope="agent",
                        reuse_class="growing_history",
                        confidence=1,
                        source="explicit",
                        allow_preemption=True,
                        allow_kv_offload=True,
                        allow_remote_worker=True,
                        expected_reuse_probability=1.0,
                    ),
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    input_tokens=128,
                    output_tokens=32,
                )
            )
            assert route.worker_id == "ross-a6000"
            assert route.worker_generation == 1
            assert route.cost.estimate_available is False
            assert route.strategy == "least-load"
            assert route.requested_strategy == "lifecycle-aware"
            assert route.fallback_reason == "candidate_cost_unavailable"
            assert route.HasField("kv_transfer")
            assert route.kv_transfer.reason
            client = GrpcSchedulerClient(f"127.0.0.1:{port}")
            try:
                now = datetime.now(UTC)
                hints = AgentHints(
                    harness_id="test",
                    task_id="task",
                    session_id="session",
                    agent_id="agent",
                    call_id="call",
                    expected_reuse_probability=0.7,
                    expected_resume_ms=5000,
                    metadata={
                        "reuse_forecast_status": "caller_supplied",
                        "reuse_forecast_task_id": "task",
                        "reuse_forecast_session_id": "session",
                        "reuse_forecast_agent_id": "agent",
                        "reuse_forecast_branch_id": "main",
                        "reuse_forecast_call_id": "call",
                        "reuse_forecast_evidence_reference": "fixture",
                        "reuse_forecast_observed_at": (now - timedelta(minutes=2)).isoformat(),
                        "reuse_forecast_expires_at": (now - timedelta(minutes=1)).isoformat(),
                    },
                )
                profile = RequestProfile(
                    tenant_id="tenant-a",
                    model_id=capabilities.models[0].model_id,
                    input_tokens=128,
                    output_tokens=32,
                    hints=hints,
                    local_node_id="ross",
                )
                result = await client.route(profile)
                assert result.kv_transfer.reason == "reuse_forecast_not_current"
                assert scheduler.received[-1].hints.metadata == hints.metadata
                assert scheduler.received[-1].local_node_id == "ross"
                assert scheduler.received[-1].hints.expected_resume_ms == 5000
                assert result.reserved_kv_bytes_per_rank == 160 * 16_384
                assert (await client.route(profile)).decision_id == result.decision_id
                await client.renew(profile, result)
                await client.cancel(profile, result)
                assert (await ledger.snapshot()).reservations[result.decision_id].state is (
                    RequestState.CANCEL_REQUESTED
                )
                with pytest.raises(grpc.aio.AioRpcError) as cross_tenant:
                    await client.release(
                        profile.model_copy(update={"tenant_id": "tenant-b"}), result
                    )
                assert cross_tenant.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                await client.release(profile, result)
                await client.release(profile, result)
                state = await ledger.snapshot()
                assert state.reservations[result.decision_id].state is RequestState.CANCEL_REQUESTED
                assert len(state.reservations[result.decision_id].renewals) == 1
                assert (
                    len([e for e in state.pending.values() if e.aggregate_id == result.decision_id])
                    == 3
                )
                # Explicit zero and absence have different meanings for an active call.
                for horizon in (0, None):
                    next_hints = hints.model_copy(deep=True)
                    next_hints.expected_resume_ms = horizon
                    if horizon is None:
                        next_hints.metadata = {}
                    next_profile = profile.model_copy(
                        update={"request_id": f"horizon-{horizon}", "hints": next_hints}
                    )
                    next_result = await client.route(next_profile)
                    assert scheduler.received[-1].hints.expected_resume_ms == horizon
                    assert scheduler.received[-1].hints.metadata == next_hints.metadata
                    await client.release(next_profile, next_result)
            finally:
                await client.aclose()
            advanced = capabilities.model_copy(update={"generation": 2})
            await worker_stub.Register(
                control_pb2.WorkerRegistration(
                    worker_id=advanced.worker_id,
                    generation=2,
                    endpoint=advanced.endpoint,
                    capabilities_json=advanced.model_dump_json(),
                )
            )
            with pytest.raises(grpc.aio.AioRpcError) as stale:
                await worker_stub.Register(
                    control_pb2.WorkerRegistration(
                        worker_id=capabilities.worker_id,
                        generation=1,
                        endpoint=capabilities.endpoint,
                        capabilities_json=capabilities.model_dump_json(),
                    )
                )
            assert stale.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            assert registry.snapshot()[1][0].capabilities.generation == 2
        finally:
            await channel.close()
            await server.stop(grace=None)

    asyncio.run(scenario())
