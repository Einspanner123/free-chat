from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import (
    CandidateCost,
    PredictiveOffloadDirective,
    RequestProfile,
    RouteDecision,
)
from freechat_contracts.cache_lifecycle import CacheLifecycleReceipt, CacheLifecycleUpdate
from freechat_contracts.preparation import PreparedAdmission


class SchedulerClient(Protocol):
    async def route(self, request: RequestProfile) -> RouteDecision: ...

    async def cache_lifecycle(
        self,
        tenant_id: str,
        update: CacheLifecycleUpdate,
    ) -> CacheLifecycleReceipt: ...

    async def renew(self, request: RequestProfile, decision: RouteDecision) -> None: ...

    async def release(self, request: RequestProfile, decision: RouteDecision) -> None: ...

    async def cancel(self, request: RequestProfile, decision: RouteDecision) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticSchedulerClient:
    """CPU contract fixture only; the service entry point requires gRPC."""

    worker_id: str
    endpoint: str
    generation: int = 1

    async def route(self, request: RequestProfile) -> RouteDecision:
        cost = CandidateCost(
            worker_id=self.worker_id,
            queue_ms=0,
            prefill_ms=0,
            decode_ms=0,
            cache_ms=0,
            network_ms=0,
            cold_start_ms=0,
            deadline_risk=0,
            eviction_externality=0,
            affinity_credit_ms=0,
            total_ms=0,
        )
        return RouteDecision(
            decision_id=str(uuid4()),
            request_id=request.request_id,
            worker_id=self.worker_id,
            worker_generation=self.generation,
            endpoint=self.endpoint,
            selected=cost,
            candidates=(cost,),
            rejected={},
            topology_generation=1,
        )

    async def cache_lifecycle(
        self,
        tenant_id: str,
        update: CacheLifecycleUpdate,
    ) -> CacheLifecycleReceipt:
        raise NotImplementedError("static_scheduler_has_no_cache_owner")

    async def aclose(self) -> None:
        return None

    async def release(self, request: RequestProfile, decision: RouteDecision) -> None:
        del request, decision

    async def renew(self, request: RequestProfile, decision: RouteDecision) -> None:
        del request, decision

    async def cancel(self, request: RequestProfile, decision: RouteDecision) -> None:
        del request, decision


class GrpcSchedulerClient:
    def __init__(self, target: str, *, token: str | None = None) -> None:
        self._metadata = () if token is None else (("authorization", f"Bearer {token}"),)
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = control_pb2_grpc.SchedulerServiceStub(  # type: ignore[no-untyped-call]
            self._channel
        )

    async def aclose(self) -> None:
        await self._channel.close()

    async def route(self, request: RequestProfile) -> RouteDecision:
        hints = request.hints
        response = await self._stub.Route(
            control_pb2.RouteRequest(
                context=control_pb2.RequestContext(
                    request_id=request.request_id,
                    idempotency_key=request.request_id,
                    tenant_id=request.tenant_id,
                    schema_version=hints.schema_version,
                ),
                hints=control_pb2.AgentHints(
                    harness_id=hints.harness_id,
                    harness_version=hints.harness_version or "",
                    task_id=hints.task_id,
                    session_id=hints.session_id or "",
                    agent_id=hints.agent_id,
                    parent_agent_id=hints.parent_agent_id or "",
                    branch_id=hints.branch_id,
                    parent_branch_id=hints.parent_branch_id or "",
                    turn_id=hints.turn_id or "",
                    call_id=hints.call_id,
                    lifecycle=hints.lifecycle,
                    prefix_scope=hints.prefix_scope,
                    reuse_class=hints.reuse_class,
                    expected_reuse_probability=hints.expected_reuse_probability,
                    expected_resume_ms=hints.expected_resume_ms or 0,
                    ttl_ms=hints.ttl_ms,
                    priority=hints.priority,
                    deadline_ms=hints.deadline_ms or 0,
                    expected_output_tokens=hints.expected_output_tokens or 0,
                    privacy_domain=hints.privacy_domain,
                    allow_preemption=hints.allow_preemption,
                    allow_kv_offload=hints.allow_kv_offload,
                    allow_remote_worker=hints.allow_remote_worker,
                    confidence=hints.confidence,
                    source=hints.source,
                    metadata=hints.metadata,
                ),
                model=request.model_id,
                native_protocol=request.native_protocol or "",
                native_request_json=request.native_request_json or "",
                input_tokens=request.input_tokens,
                output_tokens=request.output_tokens,
                cache_key=request.cache_key or "",
                local_node_id=request.local_node_id,
            ),
            metadata=self._metadata,
        )
        cost = CandidateCost(
            worker_id=response.worker_id,
            queue_ms=response.cost.queue_ms,
            prefill_ms=response.cost.prefill_ms,
            decode_ms=response.cost.decode_ms,
            cache_ms=response.cost.cache_ms,
            network_ms=response.cost.network_ms,
            cold_start_ms=response.cost.cold_start_ms,
            deadline_risk=response.cost.deadline_risk,
            eviction_externality=response.cost.eviction_externality,
            affinity_credit_ms=response.cost.affinity_credit_ms,
            total_ms=response.cost.total,
            estimate_available=response.cost.estimate_available,
            calibration_id=response.cost.calibration_id or None,
            unavailable_reason=response.cost.unavailable_reason or None,
        )
        return RouteDecision(
            decision_id=response.decision_id,
            request_id=request.request_id,
            worker_id=response.worker_id,
            worker_generation=response.worker_generation,
            engine_instance_id=response.engine_instance_id
            if response.HasField("engine_instance_id")
            else None,
            endpoint=response.endpoint,
            selected=cost,
            candidates=(cost,),
            rejected=_parse_rejections(response.rejected_candidates),
            topology_generation=response.topology_generation,
            strategy=response.strategy or "lifecycle-aware",
            requested_strategy=response.requested_strategy or None,
            fallback_reason=response.fallback_reason or None,
            lease_ttl_ms=response.lease_ttl_ms,
            reserved_kv_bytes_per_rank=response.reserved_kv_bytes_per_rank,
            preparation=(
                PreparedAdmission.model_validate_json(response.preparation_json)
                if response.preparation_json
                else None
            ),
            kv_transfer=PredictiveOffloadDirective(
                applicable=response.kv_transfer.applicable,
                enabled=response.kv_transfer.enabled,
                max_offload_tokens=response.kv_transfer.max_offload_tokens,
                estimated_kv_bytes=response.kv_transfer.estimated_kv_bytes,
                predicted_reuse_probability=(response.kv_transfer.predicted_reuse_probability),
                predicted_eviction_probability=(
                    response.kv_transfer.predicted_eviction_probability
                ),
                estimated_recompute_ms=response.kv_transfer.estimated_recompute_ms,
                estimated_store_ms=response.kv_transfer.estimated_store_ms,
                estimated_restore_ms=response.kv_transfer.estimated_restore_ms,
                expected_net_benefit_ms=response.kv_transfer.expected_net_benefit_ms,
                reason=response.kv_transfer.reason or "not_evaluated",
            ),
        )

    async def cache_lifecycle(
        self,
        tenant_id: str,
        update: CacheLifecycleUpdate,
    ) -> CacheLifecycleReceipt:
        response = await self._stub.UpdateCacheLifecycle(
            control_pb2.CacheLifecycleUpdateRequest(
                context=control_pb2.RequestContext(
                    tenant_id=tenant_id,
                    schema_version=1,
                    request_id=f"{update.decision_id}:cache:{update.sequence}",
                    idempotency_key=f"{update.decision_id}:cache:{update.sequence}",
                ),
                update_json=update.model_dump_json(),
            ),
            metadata=self._metadata,
            timeout=7,
        )
        receipt = CacheLifecycleReceipt.model_validate_json(response.receipt_json)
        if receipt.command.owner.tenant_id != tenant_id or receipt.command.update != update:
            raise ValueError("cache_response_binding_mismatch")
        return receipt

    async def release(self, request: RequestProfile, decision: RouteDecision) -> None:
        response = await self._stub.Release(
            control_pb2.LeaseRequest(
                context=control_pb2.RequestContext(
                    request_id=f"{request.request_id}:release",
                    idempotency_key=f"{request.request_id}:release:{decision.decision_id}",
                    tenant_id=request.tenant_id,
                    schema_version=request.hints.schema_version,
                ),
                decision_id=decision.decision_id,
                worker_id=decision.worker_id,
                worker_generation=decision.worker_generation,
            ),
            metadata=self._metadata,
        )
        if response.status not in {"released", "completion_pending", "cancel_requested", "expired"}:
            raise RuntimeError(f"scheduler lease release failed: {response.status}")

    async def renew(self, request: RequestProfile, decision: RouteDecision) -> None:
        response = await self._stub.RenewLease(
            control_pb2.LeaseRequest(
                context=control_pb2.RequestContext(
                    request_id=f"{request.request_id}:renew",
                    idempotency_key=f"{request.request_id}:renew:{uuid4()}",
                    tenant_id=request.tenant_id,
                    schema_version=request.hints.schema_version,
                ),
                decision_id=decision.decision_id,
                worker_id=decision.worker_id,
                worker_generation=decision.worker_generation,
            ),
            metadata=self._metadata,
        )
        if response.status != "renewed":
            raise RuntimeError(f"scheduler lease renewal failed: {response.status}")

    async def cancel(self, request: RequestProfile, decision: RouteDecision) -> None:
        response = await self._stub.Cancel(
            control_pb2.LeaseRequest(
                context=control_pb2.RequestContext(
                    request_id=request.request_id,
                    idempotency_key=f"{request.request_id}:cancel:{decision.decision_id}",
                    tenant_id=request.tenant_id,
                ),
                decision_id=decision.decision_id,
                worker_id=decision.worker_id,
                worker_generation=decision.worker_generation,
            ),
            metadata=self._metadata,
        )
        if response.status != "cancel_requested":
            raise RuntimeError(f"scheduler cancellation failed: {response.status}")


def _parse_rejections(entries: Any) -> dict[str, tuple[str, ...]]:
    rejected: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        worker_id, separator, reasons = str(entry).partition(":")
        if separator:
            rejected[worker_id] = tuple(item for item in reasons.split(",") if item)
    return rejected
