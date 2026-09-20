from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from freechat_contracts import (
    CandidateCost,
    HintSource,
    Lifecycle,
    PredictiveOffloadDirective,
    RequestProfile,
    RouteDecision,
)
from freechat_contracts.preparation import PreparedAdmission

from freechat_scheduler.calibration import matching_calibration
from freechat_scheduler.forecasts import forecast_rejection
from freechat_scheduler.registry import InMemoryWorkerRegistry, WorkerSnapshot
from freechat_scheduler.resource_groups import GroupLedger, GroupState
from freechat_scheduler.resources import required_kv_bytes_per_rank


class NoEligibleWorker(RuntimeError):
    def __init__(self, rejected: dict[str, tuple[str, ...]]) -> None:
        super().__init__("no worker satisfies the request's hard constraints")
        self.rejected = rejected


class RoutingStrategy(StrEnum):
    ROUND_ROBIN = "round-robin"
    LEAST_LOAD = "least-load"
    PREFIX_AFFINITY = "prefix-affinity"
    COST_AWARE = "cost-aware"
    LIFECYCLE_AWARE = "lifecycle-aware"


@dataclass(frozen=True, slots=True)
class CostWeights:
    queue_request_ms: float = 15.0
    cold_model_ms: float = 20_000.0
    eviction_externality_per_gib_ms: float = 2.0
    deadline_risk_weight: float = 1_000.0
    explicit_affinity_credit_ms: float = 250.0
    inferred_affinity_credit_ms: float = 100.0
    kv_eviction_high_watermark: float = 0.80
    kv_eviction_wait_horizon_ms: int = 60_000
    telemetry_max_age_seconds: float = 30.0


class Scheduler:
    def __init__(
        self,
        registry: InMemoryWorkerRegistry,
        *,
        weights: CostWeights | None = None,
        strategy: RoutingStrategy = RoutingStrategy.LIFECYCLE_AWARE,
        group_snapshot: Callable[[], tuple[datetime, GroupLedger]] | None = None,
    ) -> None:
        self._registry = registry
        self._group_snapshot = group_snapshot
        self._weights = weights or CostWeights()
        self._strategy = strategy
        self._round_robin_index = 0

    def preparation_candidates(self, request: RequestProfile) -> tuple[WorkerSnapshot, ...]:
        """Apply locality/health/model gates before disclosing a prompt to workers."""
        _, workers = self._registry.snapshot()
        probe = request.model_copy(update={"input_tokens": 0, "output_tokens": 1})
        group_view = self._group_snapshot() if self._group_snapshot is not None else None
        return tuple(
            worker for worker in workers if not self._hard_filter(probe, worker, group_view)
        )

    def route(
        self,
        request: RequestProfile,
        *,
        reserved: dict[str, tuple[int, int]] | None = None,
        prepared: dict[str, PreparedAdmission] | None = None,
    ) -> RouteDecision:
        if request.native_protocol is not None and prepared is None:
            raise ValueError("native_preparation_required")
        topology_generation, workers = self._registry.snapshot()
        if reserved:
            workers = tuple(
                WorkerSnapshot(
                    item.capabilities,
                    item.telemetry.model_copy(
                        update={
                            "kv_admission_available_bytes_per_rank": (
                                None
                                if item.telemetry.kv_admission_available_bytes_per_rank is None
                                else max(
                                    0,
                                    item.telemetry.kv_admission_available_bytes_per_rank
                                    - reserved.get(item.capabilities.worker_id, (0, 0))[0],
                                )
                            ),
                            "active_requests": (
                                max(
                                    item.telemetry.active_requests,
                                    reserved.get(item.capabilities.worker_id, (0, 0))[1],
                                )
                                if item.telemetry.admission_accounting
                                == "scheduler_exclusive_gross"
                                else item.telemetry.active_requests
                                + reserved.get(item.capabilities.worker_id, (0, 0))[1]
                            ),
                        }
                    ),
                )
                for item in workers
            )
        group_view = self._group_snapshot() if self._group_snapshot is not None else None
        rejected: dict[str, tuple[str, ...]] = {}
        eligible: list[WorkerSnapshot] = []
        profiles: dict[str, RequestProfile] = {}
        for worker in workers:
            worker_id = worker.capabilities.worker_id
            profile = request
            if prepared is not None:
                item = prepared.get(worker_id)
                if (
                    item is None
                    or item.worker_id != worker_id
                    or item.generation != worker.capabilities.generation
                    or item.engine_instance_id != worker.telemetry.engine_instance_id
                    or item.budget.tenant_id != request.tenant_id
                    or item.budget.protocol != request.native_protocol
                    or item.budget.body_sha256 != request.native_body_sha256
                    or item.budget.expires_at <= time.time()
                ):
                    rejected[worker_id] = ("native_preparation_missing_expired_or_mismatched",)
                    continue
                profile = request.model_copy(
                    update={
                        "input_tokens": item.budget.input_tokens,
                        "output_tokens": item.budget.output_tokens,
                        "estimated_kv_bytes": 0,
                    }
                )
            profiles[worker_id] = profile
            reasons = self._hard_filter(profile, worker, group_view)
            if reasons:
                rejected[worker.capabilities.worker_id] = tuple(reasons)
            else:
                eligible.append(worker)

        if not eligible:
            raise NoEligibleWorker(rejected)

        cost_by_worker = {
            worker.capabilities.worker_id: self._cost(
                profiles[worker.capabilities.worker_id],
                worker,
                include_lifecycle=self._strategy is RoutingStrategy.LIFECYCLE_AWARE,
            )
            for worker in eligible
        }
        ordered_workers = self._order_workers(request, eligible, cost_by_worker)
        costs = tuple(cost_by_worker[item.capabilities.worker_id] for item in ordered_workers)
        selected = costs[0]
        worker_caps = next(
            item.capabilities
            for item in eligible
            if item.capabilities.worker_id == selected.worker_id
        )
        selected_worker = next(
            item for item in eligible if item.capabilities.worker_id == selected.worker_id
        )
        fallback = self._strategy in {
            RoutingStrategy.COST_AWARE,
            RoutingStrategy.LIFECYCLE_AWARE,
        } and any(not item.estimate_available for item in costs)
        return RouteDecision(
            request_id=request.request_id,
            worker_id=worker_caps.worker_id,
            worker_generation=worker_caps.generation,
            engine_instance_id=selected_worker.telemetry.engine_instance_id,
            endpoint=worker_caps.endpoint,
            selected=selected,
            candidates=costs,
            rejected=rejected,
            topology_generation=topology_generation,
            strategy=RoutingStrategy.LEAST_LOAD if fallback else self._strategy,
            requested_strategy=self._strategy,
            fallback_reason="candidate_cost_unavailable" if fallback else None,
            preparation=None if prepared is None else prepared[worker_caps.worker_id],
            kv_transfer=(
                self._predictive_offload(request, selected_worker)
                if prepared is None
                else PredictiveOffloadDirective(reason="managed_native_transfer_not_enabled")
            ),
            reserved_kv_bytes_per_rank=required_kv_bytes_per_rank(
                profiles[worker_caps.worker_id],
                next(model for model in worker_caps.models if model.model_id == request.model_id),
            )
            or 0,
        )

    def _predictive_offload(
        self,
        request: RequestProfile,
        worker: WorkerSnapshot,
    ) -> PredictiveOffloadDirective:
        model = next(
            item for item in worker.capabilities.models if item.model_id == request.model_id
        )
        if not model.supports_kv_offload:
            return PredictiveOffloadDirective(reason="worker_model_lacks_kv_offload")
        if self._strategy is not RoutingStrategy.LIFECYCLE_AWARE:
            return PredictiveOffloadDirective(
                applicable=True,
                reason="strategy_does_not_authorize_offload",
            )
        hints = request.hints
        if not hints.allow_kv_offload:
            return PredictiveOffloadDirective(
                applicable=True,
                reason="offload_forbidden_by_hints",
            )
        if hints.lifecycle in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}:
            return PredictiveOffloadDirective(
                applicable=True,
                reason="terminal_lifecycle",
            )
        forecast_error = forecast_rejection(hints, datetime.now(UTC))
        if forecast_error is not None:
            return PredictiveOffloadDirective(applicable=True, reason=forecast_error)
        estimated_kv_bytes = request.estimated_kv_bytes or (
            request.input_tokens * model.kv_bytes_per_token
        )
        if request.input_tokens == 0 or estimated_kv_bytes == 0:
            return PredictiveOffloadDirective(
                applicable=True,
                reason="missing_kv_size_estimate",
            )

        telemetry = worker.telemetry
        calibration, reason = matching_calibration(request, worker)
        if calibration is None:
            return PredictiveOffloadDirective(
                applicable=True,
                estimated_kv_bytes=estimated_kv_bytes,
                reason=reason,
            )
        if calibration.store_bytes_per_second is None or calibration.load_bytes_per_second is None:
            return PredictiveOffloadDirective(
                applicable=True,
                estimated_kv_bytes=estimated_kv_bytes,
                reason="transfer_calibration_required",
            )
        assert calibration.transfer_bytes_min is not None
        assert calibration.transfer_bytes_max is not None
        if (
            not calibration.transfer_bytes_min
            <= estimated_kv_bytes
            <= calibration.transfer_bytes_max
        ):
            return PredictiveOffloadDirective(applicable=True, reason="transfer_size_out_of_scope")
        reuse_probability = hints.expected_reuse_probability
        if hints.source is HintSource.INFERRED:
            reuse_probability *= hints.confidence
        pressure_risk = 0.0
        if telemetry.kv_cache_capacity_bytes is not None:
            assert telemetry.kv_cache_free_bytes is not None
            cache_used_ratio = 1.0 - (
                telemetry.kv_cache_free_bytes / telemetry.kv_cache_capacity_bytes
            )
            watermark = self._weights.kv_eviction_high_watermark
            pressure_risk = max(
                0.0,
                min(1.0, (cache_used_ratio - watermark) / (1.0 - watermark)),
            )
        wait_risk = 0.0
        if hints.expected_resume_ms is not None:
            wait_risk = min(
                hints.expected_resume_ms / self._weights.kv_eviction_wait_horizon_ms,
                1.0,
            )
        eviction_probability = max(pressure_risk, wait_risk)
        recompute_ms = request.input_tokens / calibration.prefill_tokens_per_second * 1_000
        store_ms = estimated_kv_bytes / calibration.store_bytes_per_second * 1_000
        restore_ms = estimated_kv_bytes / calibration.load_bytes_per_second * 1_000
        expected_avoided_recompute_ms = reuse_probability * eviction_probability * recompute_ms
        expected_transfer_ms = store_ms + (reuse_probability * eviction_probability * restore_ms)
        net_benefit_ms = expected_avoided_recompute_ms - expected_transfer_ms
        enabled = reuse_probability > 0 and eviction_probability > 0 and net_benefit_ms > 0
        reason = "expected_recompute_exceeds_transfer" if enabled else "transfer_cost_not_recovered"
        return PredictiveOffloadDirective(
            applicable=True,
            enabled=enabled,
            max_offload_tokens=request.input_tokens if enabled else 0,
            estimated_kv_bytes=estimated_kv_bytes,
            predicted_reuse_probability=reuse_probability,
            predicted_eviction_probability=eviction_probability,
            estimated_recompute_ms=recompute_ms,
            estimated_store_ms=store_ms,
            estimated_restore_ms=restore_ms,
            expected_net_benefit_ms=net_benefit_ms,
            reason=reason,
        )

    def _order_workers(
        self,
        request: RequestProfile,
        workers: list[WorkerSnapshot],
        costs: dict[str, CandidateCost],
    ) -> list[WorkerSnapshot]:
        if self._strategy is RoutingStrategy.ROUND_ROBIN:
            ordered = sorted(workers, key=lambda item: item.capabilities.worker_id)
            selected_index = self._round_robin_index % len(ordered)
            self._round_robin_index += 1
            return ordered[selected_index:] + ordered[:selected_index]
        missing_cost = self._strategy in {
            RoutingStrategy.COST_AWARE,
            RoutingStrategy.LIFECYCLE_AWARE,
        } and any(not item.estimate_available for item in costs.values())
        if self._strategy is RoutingStrategy.LEAST_LOAD or missing_cost:
            return sorted(
                workers,
                key=lambda item: (
                    item.telemetry.active_requests,
                    item.telemetry.queue_depth,
                    item.capabilities.worker_id,
                ),
            )
        if self._strategy is RoutingStrategy.PREFIX_AFFINITY:
            return sorted(
                workers,
                key=lambda item: (
                    not (
                        request.cache_key is not None
                        and request.cache_key in item.telemetry.cached_prefixes
                    ),
                    item.telemetry.active_requests,
                    item.telemetry.queue_depth,
                    item.capabilities.worker_id,
                ),
            )
        return sorted(
            workers,
            key=lambda item: _sort_key(costs[item.capabilities.worker_id]),
        )

    def _hard_filter(
        self,
        request: RequestProfile,
        worker: WorkerSnapshot,
        group_view: tuple[datetime, GroupLedger] | None = None,
    ) -> list[str]:
        caps = worker.capabilities
        telemetry = worker.telemetry
        reasons: list[str] = []
        observed = telemetry.observed_at
        if observed.tzinfo is None:
            reasons.append("telemetry_timestamp_without_timezone")
        else:
            age = (datetime.now(UTC) - observed).total_seconds()
            if age > self._weights.telemetry_max_age_seconds:
                reasons.append("telemetry_stale")
            if age < -5:
                reasons.append("telemetry_from_future")
        if not telemetry.healthy:
            reasons.append("worker_unhealthy")
        if telemetry.draining:
            reasons.append("worker_draining")
        model = next((item for item in caps.models if item.model_id == request.model_id), None)
        if model is None:
            reasons.append("model_unavailable")
            return reasons
        required_context = request.input_tokens + request.output_tokens
        grouped = (
            caps.resource_group_id is not None
            or max(model.tensor_parallel_size, model.pipeline_parallel_size, len(caps.gpu_ids)) > 1
        )
        if grouped:
            if group_view is None or caps.resource_group_id is None:
                reasons.append("resource_group_snapshot_required")
            else:
                timestamp, ledger = group_view
                if (
                    timestamp.tzinfo is None
                    or not 0 <= (datetime.now(UTC) - timestamp).total_seconds() <= 5
                ):
                    reasons.append("resource_group_snapshot_stale")
                group = ledger.groups.get(caps.resource_group_id)
                if group is None or group.state is not GroupState.READY:
                    reasons.append("resource_group_not_ready")
                elif group.expires_at <= datetime.now(UTC):
                    reasons.append("resource_group_lease_expired")
                elif (
                    caps.resource_group_generation != group.generation
                    or caps.generation != group.generation
                    or caps.worker_id != group.spec.worker_id
                    or caps.node_id != group.node_id
                    or set(caps.gpu_ids) != set(group.spec.gpu_ids)
                    or len(caps.gpu_ids) != len(group.spec.gpu_ids)
                    or telemetry.engine_instance_id != group.engine_instance_id
                    or model.model_id != group.spec.model_id
                    or model.revision != group.spec.model_revision
                    or model.tensor_parallel_size != group.spec.tensor_parallel_size
                    or model.pipeline_parallel_size != 1
                ):
                    reasons.append("resource_group_binding_mismatch")
        if model.max_context_tokens < required_context:
            reasons.append("context_capacity")
        required_bytes = required_kv_bytes_per_rank(request, model)
        available_bytes = telemetry.kv_admission_available_bytes_per_rank
        if required_bytes is None:
            reasons.append("kv_geometry_unknown")
        if available_bytes is None:
            reasons.append("kv_admission_budget_unknown")
        if (
            required_bytes is not None
            and available_bytes is not None
            and available_bytes < required_bytes
        ):
            reasons.append("vram_capacity")
        if request.local_node_id is None and (
            not request.hints.allow_remote_worker or not caps.allow_remote_requests
        ):
            reasons.append("request_locality_unknown")
        remote = request.local_node_id is not None and caps.node_id != request.local_node_id
        if remote and not request.hints.allow_remote_worker:
            reasons.append("remote_worker_forbidden")
        if remote and not caps.allow_remote_requests:
            reasons.append("worker_rejects_remote")
        return reasons

    def _cost(
        self,
        request: RequestProfile,
        worker: WorkerSnapshot,
        *,
        include_lifecycle: bool,
    ) -> CandidateCost:
        caps = worker.capabilities
        telemetry = worker.telemetry
        calibration, reason = matching_calibration(request, worker)
        if calibration is None:
            return CandidateCost(
                worker_id=caps.worker_id,
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
                estimate_available=False,
                unavailable_reason=reason,
            )
        if request.local_node_id is None:
            return CandidateCost(
                worker_id=caps.worker_id,
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
                estimate_available=False,
                unavailable_reason="request_locality_unknown",
            )
        model_loaded = any(item.model_id == request.model_id for item in caps.models)
        queue_ms = telemetry.queue_depth * self._weights.queue_request_ms
        prefill_ms = request.input_tokens / calibration.prefill_tokens_per_second * 1_000
        decode_ms = max(0, request.output_tokens - 1) / calibration.decode_tokens_per_second * 1_000
        cache_hit = request.cache_key is not None and request.cache_key in telemetry.cached_prefixes
        cache_ms = 0.0
        # Absence of a local prefix does not imply an external KV load will occur.
        # Until the residency catalog proves a load, estimate a cold prefill only.
        remote = request.local_node_id is not None and caps.node_id != request.local_node_id
        network_ms = telemetry.network_rtt_ms if remote else 0.0
        cold_start_ms = 0.0 if model_loaded else self._weights.cold_model_ms
        predicted_ms = queue_ms + prefill_ms + decode_ms + cache_ms + network_ms + cold_start_ms
        deadline_risk = 0.0
        if request.hints.deadline_ms is not None and predicted_ms > request.hints.deadline_ms:
            overrun = (predicted_ms - request.hints.deadline_ms) / request.hints.deadline_ms
            deadline_risk = overrun * self._weights.deadline_risk_weight
        externality = 0.0
        if include_lifecycle:
            externality = (
                (
                    required_kv_bytes_per_rank(
                        request,
                        next(item for item in caps.models if item.model_id == request.model_id),
                    )
                    or 0
                )
                / (1024**3)
            ) * self._weights.eviction_externality_per_gib_ms
        affinity_credit_ms = 0.0
        if include_lifecycle and cache_hit:
            base_credit = (
                self._weights.explicit_affinity_credit_ms
                if request.hints.source is HintSource.EXPLICIT
                else self._weights.inferred_affinity_credit_ms * request.hints.confidence
            )
            lifecycle_probability = request.hints.expected_reuse_probability
            if request.hints.lifecycle.value == "resume":
                lifecycle_probability = 1.0
            affinity_credit_ms = min(base_credit * lifecycle_probability, prefill_ms)
        total_ms = max(
            0.0,
            predicted_ms + deadline_risk + externality - affinity_credit_ms,
        )
        return CandidateCost(
            worker_id=caps.worker_id,
            estimate_available=True,
            calibration_id=calibration.calibration_id,
            queue_ms=queue_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            cache_ms=cache_ms,
            network_ms=network_ms,
            cold_start_ms=cold_start_ms,
            deadline_risk=deadline_risk,
            eviction_externality=externality,
            affinity_credit_ms=affinity_credit_ms,
            total_ms=total_ms,
        )


def _sort_key(cost: CandidateCost) -> tuple[float, str]:
    return cost.total_ms, cost.worker_id
