from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import suppress
from pathlib import Path
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

from freechat_scheduler.group_reconciler import GroupRuntimeConfig, configured_reconciler
from freechat_scheduler.preparation import NativePreparer
from freechat_scheduler.registry import InMemoryWorkerRegistry, PersistentWorkerRegistry
from freechat_scheduler.request_execution import (
    LocalGrpcExecutionDriver,
    RequestExecutionConfig,
    RequestExecutionReconciler,
)
from freechat_scheduler.request_ledger import RequestLedger
from freechat_scheduler.scheduler import NoEligibleWorker, RoutingStrategy, Scheduler

LOGGER = logging.getLogger(__name__)
LeaseBook = RequestLedger


class SchedulerGrpcService(control_pb2_grpc.SchedulerServiceServicer):
    def __init__(
        self,
        scheduler: Scheduler,
        leases: LeaseBook,
        emitter: DurableLifecycleEmitter | None = None,
        execution: RequestExecutionReconciler | None = None,
        preparer: NativePreparer | None = None,
    ) -> None:
        self._scheduler, self._leases, self._emitter = scheduler, leases, emitter
        self._flush_lock = asyncio.Lock()
        self._execution = execution
        self._preparer = preparer

    async def Route(
        self, request: control_pb2.RouteRequest, context: Any
    ) -> control_pb2.RouteDecision:
        try:
            profile = _request_profile(request)
            prepared = None
            if self._preparer is not None:
                prepared = await self._preparer.prepare(
                    profile, self._scheduler.preparation_candidates(profile)
                )
            elif profile.native_protocol is not None:
                raise ValueError("native_preparer_not_configured")
            decision = await self._leases.reserve(
                profile,
                request.context.idempotency_key or profile.request_id,
                lambda reserved: self._scheduler.route(
                    profile, reserved=reserved, prepared=prepared
                ),
            )
        except (ValueError, NoEligibleWorker) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
        return _decision_message(decision)

    async def RenewLease(
        self, request: control_pb2.LeaseRequest, context: Any
    ) -> control_pb2.Operation:
        try:
            await self._leases.renew(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
                request.context.tenant_id,
                request.context.idempotency_key,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return control_pb2.Operation(operation_id=request.decision_id, status="renewed")

    async def Release(
        self, request: control_pb2.LeaseRequest, context: Any
    ) -> control_pb2.Operation:
        try:
            await self._leases.release(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
                request.context.tenant_id,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        state = (await self._leases.snapshot()).reservations[request.decision_id].state
        return control_pb2.Operation(operation_id=request.decision_id, status=state)

    async def Cancel(
        self, request: control_pb2.LeaseRequest, context: Any
    ) -> control_pb2.Operation:
        try:
            await self._leases.release(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
                request.context.tenant_id,
                cancelled=True,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return control_pb2.Operation(operation_id=request.decision_id, status="cancel_requested")

    async def ExplainDecision(
        self, request: control_pb2.LeaseRequest, context: Any
    ) -> control_pb2.RouteDecision:
        try:
            decision = await self._leases.require(
                request.decision_id,
                request.worker_id,
                request.worker_generation,
                request.context.tenant_id,
            )
        except (KeyError, ValueError) as error:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
        return _decision_message(decision)

    async def flush_events(self) -> None:
        if self._emitter is None:
            return
        async with self._flush_lock:
            for event in (await self._leases.snapshot()).pending.values():
                await self._emitter.emit(event, str(event.payload["harness_id"]))
                await self._leases.acknowledge_event(event.event_id)

    async def maintain(self) -> None:
        while True:
            for action in (self.flush_events, self._leases.expire):
                try:
                    await action()
                except Exception:
                    LOGGER.exception("request ledger maintenance failed; pending intents retained")
            if self._execution is not None:
                try:
                    errors = await self._execution.tick()
                    if errors:
                        LOGGER.warning("request execution observations unavailable: %s", errors)
                except Exception:
                    LOGGER.exception("request execution reconciliation failed; capacity retained")
            await asyncio.sleep(1)


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
        try:
            await self._registry.register(capabilities, telemetry)
        except ValueError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            raise AssertionError("context.abort must terminate the RPC") from error
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
                if (
                    telemetry.worker_id != request.worker_id
                    or telemetry.generation != request.generation
                ):
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
        native_protocol=request.native_protocol or None,
        native_request_json=request.native_request_json or None,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        cache_key=request.cache_key or None,
        local_node_id=request.local_node_id if request.HasField("local_node_id") else None,
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
                if (
                    hints.expected_resume_ms
                    or lifecycle in {Lifecycle.TOOL_WAIT, Lifecycle.RESUME}
                    or hints.metadata.get("reuse_forecast_status") == "caller_supplied"
                )
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
            metadata=dict(hints.metadata),
        ),
    )


def _decision_message(decision: RouteDecision) -> control_pb2.RouteDecision:
    cost = decision.selected
    return control_pb2.RouteDecision(
        decision_id=decision.decision_id,
        worker_id=decision.worker_id,
        endpoint=decision.endpoint,
        worker_generation=decision.worker_generation,
        engine_instance_id=decision.engine_instance_id,
        preparation_json=(
            "" if decision.preparation is None else decision.preparation.model_dump_json()
        ),
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
            estimate_available=cost.estimate_available,
            calibration_id=cost.calibration_id or "",
            unavailable_reason=cost.unavailable_reason or "",
        ),
        rejected_candidates=[
            f"{worker_id}:{','.join(reasons)}"
            for worker_id, reasons in sorted(decision.rejected.items())
        ],
        lease_ttl_ms=decision.lease_ttl_ms,
        reserved_kv_bytes_per_rank=decision.reserved_kv_bytes_per_rank,
        topology_generation=decision.topology_generation,
        strategy=decision.strategy,
        requested_strategy=decision.requested_strategy or "",
        fallback_reason=decision.fallback_reason or "",
        kv_transfer=control_pb2.PredictiveOffloadDirective(
            applicable=decision.kv_transfer.applicable,
            enabled=decision.kv_transfer.enabled,
            max_offload_tokens=decision.kv_transfer.max_offload_tokens,
            estimated_kv_bytes=decision.kv_transfer.estimated_kv_bytes,
            predicted_reuse_probability=(decision.kv_transfer.predicted_reuse_probability),
            predicted_eviction_probability=(decision.kv_transfer.predicted_eviction_probability),
            estimated_recompute_ms=decision.kv_transfer.estimated_recompute_ms,
            estimated_store_ms=decision.kv_transfer.estimated_store_ms,
            estimated_restore_ms=decision.kv_transfer.estimated_restore_ms,
            expected_net_benefit_ms=decision.kv_transfer.expected_net_benefit_ms,
            reason=decision.kv_transfer.reason,
        ),
    )


async def serve(address: str, *, contract_only: bool = False) -> None:
    preparer = (
        None if contract_only else NativePreparer(os.environ.get("FREECHAT_WORKER_TOKEN", ""))
    )
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
    group_config_path = os.environ.get("FREECHAT_GROUP_RUNTIME_CONFIG")
    reconciler = None
    if group_config_path:
        config = GroupRuntimeConfig.model_validate_json(Path(group_config_path).read_text())
        reconciler = configured_reconciler(config, store, registry)
        await reconciler.tick()
    leases = LeaseBook(store)
    execution = None
    execution_config_path = os.environ.get("FREECHAT_REQUEST_EXECUTION_CONFIG")
    if execution_config_path:
        execution_config = RequestExecutionConfig.model_validate_json(
            Path(execution_config_path).read_text()
        )
        execution = RequestExecutionReconciler(leases, LocalGrpcExecutionDriver(execution_config))
    scheduler_service = SchedulerGrpcService(
        Scheduler(
            registry,
            group_snapshot=None if reconciler is None else reconciler.snapshot,
            strategy=RoutingStrategy(
                os.environ.get("FREECHAT_ROUTING_STRATEGY", "lifecycle-aware")
            ),
        ),
        leases,
        emitter,
        execution,
        preparer,
    )
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        scheduler_service,
        server,
    )
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry, emitter),
        server,
    )
    server.add_insecure_port(address)
    maintenance = None
    group_maintenance = None
    try:
        await server.start()
        maintenance = asyncio.create_task(scheduler_service.maintain())
        if reconciler is not None:
            group_maintenance = asyncio.create_task(reconciler.run())
        await server.wait_for_termination()
    finally:
        if group_maintenance is not None:
            group_maintenance.cancel()
            with suppress(asyncio.CancelledError):
                await group_maintenance
        if maintenance is not None:
            maintenance.cancel()
            with suppress(asyncio.CancelledError):
                await maintenance
        await server.stop(grace=5)
        if etcd is not None:
            await etcd.close()
        if nats_client is not None:
            await nats_client.drain()


def run() -> None:
    asyncio.run(serve(os.environ.get("FREECHAT_SCHEDULER_LISTEN", "0.0.0.0:50051")))
