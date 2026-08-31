from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import (
    AgentHints,
    HintSource,
    Lifecycle,
    PrefixScope,
    RequestProfile,
    ReuseClass,
    RouteDecision,
    WorkerCapabilities,
    WorkerTelemetry,
)

from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.scheduler import NoEligibleWorker, Scheduler


@dataclass(slots=True)
class LeaseBook:
    decisions: dict[str, RouteDecision] = field(default_factory=dict)

    def remember(self, decision: RouteDecision) -> None:
        self.decisions[decision.decision_id] = decision

    def require(self, decision_id: str, worker_id: str, generation: int) -> RouteDecision:
        decision = self.decisions.get(decision_id)
        if decision is None:
            raise KeyError("decision_not_found")
        if decision.worker_id != worker_id or decision.worker_generation != generation:
            raise ValueError("lease_fencing_mismatch")
        return decision

    def release(self, decision_id: str, worker_id: str, generation: int) -> RouteDecision:
        decision = self.require(decision_id, worker_id, generation)
        del self.decisions[decision_id]
        return decision


class SchedulerGrpcService(control_pb2_grpc.SchedulerServiceServicer):
    def __init__(self, scheduler: Scheduler, leases: LeaseBook) -> None:
        self._scheduler = scheduler
        self._leases = leases

    async def Route(
        self,
        request: control_pb2.RouteRequest,
        context: Any,
    ) -> control_pb2.RouteDecision:
        try:
            decision = self._scheduler.route(_request_profile(request))
        except (ValueError, NoEligibleWorker) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
        self._leases.remember(decision)
        return _decision_message(decision)

    async def RenewLease(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            self._leases.require(request.decision_id, request.worker_id, request.worker_generation)
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return control_pb2.Operation(operation_id=request.decision_id, status="renewed")

    async def Release(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            self._leases.release(request.decision_id, request.worker_id, request.worker_generation)
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return control_pb2.Operation(operation_id=request.decision_id, status="released")

    async def Cancel(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            self._leases.release(request.decision_id, request.worker_id, request.worker_generation)
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return control_pb2.Operation(operation_id=request.decision_id, status="cancelled")

    async def ExplainDecision(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.RouteDecision:
        try:
            decision = self._leases.require(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
        return _decision_message(decision)


class WorkerGrpcService(control_pb2_grpc.WorkerControlServiceServicer):
    def __init__(self, registry: InMemoryWorkerRegistry) -> None:
        self._registry = registry

    async def Register(
        self,
        request: control_pb2.WorkerRegistration,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            capabilities = WorkerCapabilities.model_validate_json(request.capabilities_json)
            if (
                capabilities.worker_id != request.worker_id
                or capabilities.generation != request.generation
                or capabilities.endpoint != request.endpoint
            ):
                raise ValueError("registration envelope does not match capabilities")
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
        telemetry = WorkerTelemetry(
            worker_id=capabilities.worker_id,
            generation=capabilities.generation,
            free_vram_bytes=capabilities.total_vram_bytes,
        )
        self._registry.upsert(capabilities, telemetry)
        self._registry.topology_generation += 1
        return control_pb2.Operation(
            operation_id=request.context.idempotency_key,
            status="registered",
        )

    async def Heartbeat(
        self,
        request_iterator: AsyncIterable[control_pb2.WorkerHeartbeat],
        context: Any,
    ) -> AsyncIterator[control_pb2.Operation]:
        async for request in request_iterator:
            try:
                telemetry = WorkerTelemetry.model_validate_json(request.telemetry_json)
                snapshots = self._registry.snapshot()[1]
                capabilities = next(
                    item.capabilities
                    for item in snapshots
                    if item.capabilities.worker_id == request.worker_id
                )
                self._registry.upsert(capabilities, telemetry)
            except (ValueError, StopIteration) as error:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            yield control_pb2.Operation(
                operation_id=request.context.idempotency_key,
                status="heartbeat_accepted",
            )


def _request_profile(request: control_pb2.RouteRequest) -> RequestProfile:
    hints = request.hints
    lifecycle = Lifecycle(hints.lifecycle or Lifecycle.ACTIVE)
    return RequestProfile(
        request_id=request.context.request_id,
        tenant_id=request.context.tenant_id,
        model_id=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        cache_key=request.cache_key or None,
        hints=AgentHints(
            harness_id=hints.harness_id or "openai-compatible",
            harness_version=hints.harness_version or None,
            task_id=hints.task_id,
            session_id=hints.session_id or None,
            agent_id=hints.agent_id,
            parent_agent_id=hints.parent_agent_id or None,
            branch_id=hints.branch_id or "main",
            parent_branch_id=hints.parent_branch_id or None,
            turn_id=hints.turn_id or None,
            call_id=hints.call_id,
            lifecycle=lifecycle,
            prefix_scope=PrefixScope(hints.prefix_scope or PrefixScope.PRIVATE),
            reuse_class=ReuseClass(hints.reuse_class or ReuseClass.UNKNOWN),
            expected_reuse_probability=hints.expected_reuse_probability,
            expected_resume_ms=(
                hints.expected_resume_ms
                if lifecycle in {Lifecycle.TOOL_WAIT, Lifecycle.RESUME}
                else None
            ),
            ttl_ms=hints.ttl_ms or 300_000,
            priority=hints.priority,
            deadline_ms=hints.deadline_ms or None,
            expected_output_tokens=hints.expected_output_tokens or None,
            privacy_domain=hints.privacy_domain or "default",
            allow_preemption=hints.allow_preemption,
            allow_kv_offload=hints.allow_kv_offload,
            allow_remote_worker=hints.allow_remote_worker,
            confidence=hints.confidence,
            source=HintSource(hints.source or HintSource.EXPLICIT),
        ),
    )


def _decision_message(decision: RouteDecision) -> control_pb2.RouteDecision:
    cost = decision.selected
    return control_pb2.RouteDecision(
        decision_id=decision.decision_id,
        worker_id=decision.worker_id,
        endpoint=decision.endpoint,
        worker_generation=decision.worker_generation,
        cost=control_pb2.CostBreakdown(
            queue_ms=cost.queue_ms,
            prefill_ms=cost.prefill_ms,
            decode_ms=cost.decode_ms,
            cache_ms=cost.cache_ms,
            network_ms=cost.network_ms,
            cold_start_ms=cost.cold_start_ms,
            deadline_risk=cost.deadline_risk,
            eviction_externality=cost.eviction_externality,
            affinity_credit_ms=cost.affinity_credit_ms,
            total=cost.total_ms,
        ),
        rejected_candidates=[
            f"{worker_id}:{','.join(reasons)}"
            for worker_id, reasons in sorted(decision.rejected.items())
        ],
        lease_ttl_ms=decision.lease_ttl_ms,
        topology_generation=decision.topology_generation,
    )


async def serve(address: str) -> None:
    registry = InMemoryWorkerRegistry()
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(Scheduler(registry), LeaseBook()),
        server,
    )
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry),
        server,
    )
    server.add_insecure_port(address)
    await server.start()
    await server.wait_for_termination()


def run() -> None:
    asyncio.run(serve(os.environ.get("FREECHAT_SCHEDULER_LISTEN", "0.0.0.0:50051")))
