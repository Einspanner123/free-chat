from __future__ import annotations

import asyncio
import hashlib
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
from freechat_control_store import EtcdHttpStore, InMemoryStore, KeyValueStore
from freechat_trace_replay import EventEnvelope
from freechat_trace_replay.bus import DurableLifecycleEmitter, connect_lifecycle_stream

from freechat_scheduler.registry import InMemoryWorkerRegistry, PersistentWorkerRegistry
from freechat_scheduler.scheduler import NoEligibleWorker, Scheduler


@dataclass(slots=True)
class LeaseBook:
    store: KeyValueStore = field(default_factory=InMemoryStore)
    key_prefix: str = "/freechat/leases/"

    async def remember(self, decision: RouteDecision) -> None:
        await self.store.put(self._key(decision.decision_id), decision.model_dump_json().encode())

    async def require(
        self,
        decision_id: str,
        worker_id: str,
        generation: int,
    ) -> RouteDecision:
        item = await self.store.get(self._key(decision_id))
        if item is None:
            raise KeyError("decision_not_found")
        decision = RouteDecision.model_validate_json(item.value)
        if decision.worker_id != worker_id or decision.worker_generation != generation:
            raise ValueError("lease_fencing_mismatch")
        return decision

    async def release(
        self,
        decision_id: str,
        worker_id: str,
        generation: int,
    ) -> RouteDecision:
        key = self._key(decision_id)
        item = await self.store.get(key)
        if item is None:
            raise KeyError("decision_not_found")
        decision = RouteDecision.model_validate_json(item.value)
        if decision.worker_id != worker_id or decision.worker_generation != generation:
            raise ValueError("lease_fencing_mismatch")
        await self.store.compare_and_delete(key, item.revision)
        return decision

    def _key(self, decision_id: str) -> str:
        return f"{self.key_prefix}{decision_id}"


class SchedulerGrpcService(control_pb2_grpc.SchedulerServiceServicer):
    def __init__(
        self,
        scheduler: Scheduler,
        leases: LeaseBook,
        emitter: DurableLifecycleEmitter | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._leases = leases
        self._emitter = emitter

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
        await self._leases.remember(decision)
        await self._emit(
            event_type="route.decided",
            idempotency_key=request.context.idempotency_key,
            tenant_id=request.context.tenant_id,
            aggregate_id=request.context.request_id,
            aggregate_generation=decision.worker_generation,
            harness_id=request.hints.harness_id or "openai-compatible",
            payload={
                "decision_id": decision.decision_id,
                "worker_id": decision.worker_id,
                "worker_generation": decision.worker_generation,
                "topology_generation": decision.topology_generation,
            },
        )
        return _decision_message(decision)

    async def RenewLease(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            await self._leases.require(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return control_pb2.Operation(operation_id=request.decision_id, status="renewed")

    async def Release(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            decision = await self._leases.release(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        await self._emit_lease_event("lease.released", request, decision)
        return control_pb2.Operation(operation_id=request.decision_id, status="released")

    async def Cancel(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.Operation:
        try:
            decision = await self._leases.release(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        await self._emit_lease_event("lease.cancelled", request, decision)
        return control_pb2.Operation(operation_id=request.decision_id, status="cancelled")

    async def ExplainDecision(
        self,
        request: control_pb2.LeaseRequest,
        context: Any,
    ) -> control_pb2.RouteDecision:
        try:
            decision = await self._leases.require(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
        return _decision_message(decision)

    async def _emit_lease_event(
        self,
        event_type: str,
        request: control_pb2.LeaseRequest,
        decision: RouteDecision,
    ) -> None:
        await self._emit(
            event_type=event_type,
            idempotency_key=request.context.idempotency_key,
            tenant_id=request.context.tenant_id,
            aggregate_id=decision.request_id,
            aggregate_generation=decision.worker_generation,
            harness_id="gateway",
            payload={"decision_id": decision.decision_id, "worker_id": decision.worker_id},
        )

    async def _emit(
        self,
        *,
        event_type: str,
        idempotency_key: str,
        tenant_id: str,
        aggregate_id: str,
        aggregate_generation: int,
        harness_id: str,
        payload: dict[str, Any],
    ) -> None:
        if self._emitter is None:
            return
        event_id = hashlib.sha256(f"{event_type}:{idempotency_key}".encode()).hexdigest()
        await self._emitter.emit(
            EventEnvelope(
                event_id=event_id,
                event_type=event_type,
                tenant_id=tenant_id,
                aggregate_id=aggregate_id,
                aggregate_generation=aggregate_generation,
                payload=payload,
            ),
            harness_id,
        )


class WorkerGrpcService(control_pb2_grpc.WorkerControlServiceServicer):
    def __init__(
        self,
        registry: InMemoryWorkerRegistry,
        emitter: DurableLifecycleEmitter | None = None,
    ) -> None:
        self._registry = registry
        self._emitter = emitter

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
        await self._registry.register(capabilities, telemetry)
        if self._emitter is not None:
            event_id = hashlib.sha256(
                f"worker.registered:{request.context.idempotency_key}".encode()
            ).hexdigest()
            await self._emitter.emit(
                EventEnvelope(
                    event_id=event_id,
                    event_type="worker.registered",
                    tenant_id="system",
                    aggregate_id=capabilities.worker_id,
                    aggregate_generation=capabilities.generation,
                    payload={
                        "worker_id": capabilities.worker_id,
                        "generation": capabilities.generation,
                        "node_id": capabilities.node_id,
                    },
                ),
                "worker",
            )
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
                if telemetry.worker_id != request.worker_id:
                    raise ValueError("heartbeat envelope does not match telemetry")
                await self._registry.heartbeat(telemetry)
            except ValueError as error:
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
    endpoint = os.environ.get("ETCD_ENDPOINT")
    store: KeyValueStore
    etcd: EtcdHttpStore | None = None
    if endpoint:
        etcd = EtcdHttpStore(endpoint)
        store = etcd
        persistent_registry = PersistentWorkerRegistry(store)
        await persistent_registry.restore()
        registry: InMemoryWorkerRegistry = persistent_registry
    else:
        store = InMemoryStore()
        registry = InMemoryWorkerRegistry()
    nats_client = None
    emitter = None
    nats_url = os.environ.get("NATS_URL")
    if nats_url:
        nats_client, publisher = await connect_lifecycle_stream(nats_url)
        emitter = DurableLifecycleEmitter(store, publisher)
        await emitter.replay()
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(Scheduler(registry), LeaseBook(store), emitter),
        server,
    )
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry, emitter),
        server,
    )
    server.add_insecure_port(address)
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        if etcd is not None:
            await etcd.close()
        if nats_client is not None:
            await nats_client.drain()


def run() -> None:
    asyncio.run(serve(os.environ.get("FREECHAT_SCHEDULER_LISTEN", "0.0.0.0:50051")))
