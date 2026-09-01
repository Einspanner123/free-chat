from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from freechat_contracts import (
    CandidateCost,
    HintSource,
    RequestProfile,
    RouteDecision,
)

from freechat_scheduler.registry import InMemoryWorkerRegistry, WorkerSnapshot


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


class Scheduler:
    def __init__(
        self,
        registry: InMemoryWorkerRegistry,
        *,
        weights: CostWeights | None = None,
        strategy: RoutingStrategy = RoutingStrategy.LIFECYCLE_AWARE,
    ) -> None:
        self._registry = registry
        self._weights = weights or CostWeights()
        self._strategy = strategy
        self._round_robin_index = 0

    def route(self, request: RequestProfile) -> RouteDecision:
        topology_generation, workers = self._registry.snapshot()
        rejected: dict[str, tuple[str, ...]] = {}
        eligible: list[WorkerSnapshot] = []
        for worker in workers:
            reasons = self._hard_filter(request, worker)
            if reasons:
                rejected[worker.capabilities.worker_id] = tuple(reasons)
            else:
                eligible.append(worker)

        if not eligible:
            raise NoEligibleWorker(rejected)

        cost_by_worker = {
            worker.capabilities.worker_id: self._cost(
                request,
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
        return RouteDecision(
            request_id=request.request_id,
            worker_id=worker_caps.worker_id,
            worker_generation=worker_caps.generation,
            endpoint=worker_caps.endpoint,
            selected=selected,
            candidates=costs,
            rejected=rejected,
            topology_generation=topology_generation,
            strategy=self._strategy,
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
        if self._strategy is RoutingStrategy.LEAST_LOAD:
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

    @staticmethod
    def _hard_filter(request: RequestProfile, worker: WorkerSnapshot) -> list[str]:
        caps = worker.capabilities
        telemetry = worker.telemetry
        reasons: list[str] = []
        if not telemetry.healthy:
            reasons.append("worker_unhealthy")
        if telemetry.draining:
            reasons.append("worker_draining")
        model = next((item for item in caps.models if item.model_id == request.model_id), None)
        if model is None:
            reasons.append("model_unavailable")
            return reasons
        required_context = request.input_tokens + request.output_tokens
        if model.max_context_tokens < required_context:
            reasons.append("context_capacity")
        if telemetry.free_vram_bytes < request.estimated_kv_bytes:
            reasons.append("vram_capacity")
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
        model_loaded = any(item.model_id == request.model_id for item in caps.models)
        queue_ms = telemetry.queue_depth * self._weights.queue_request_ms
        prefill_ms = request.input_tokens / telemetry.estimated_prefill_tokens_per_second * 1_000
        decode_ms = request.output_tokens / telemetry.estimated_decode_tokens_per_second * 1_000
        cache_hit = request.cache_key is not None and request.cache_key in telemetry.cached_prefixes
        cache_ms = 0.0
        if request.estimated_kv_bytes and not cache_hit:
            cache_ms = request.estimated_kv_bytes / telemetry.cache_load_bytes_per_second * 1_000
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
                request.estimated_kv_bytes / (1024**3)
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
