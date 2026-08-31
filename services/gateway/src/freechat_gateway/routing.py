from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from freechat_contracts import CandidateCost, RequestProfile, RouteDecision


class SchedulerClient(Protocol):
    async def route(self, request: RequestProfile) -> RouteDecision: ...


@dataclass(frozen=True, slots=True)
class StaticSchedulerClient:
    """Compose bootstrap only; production uses generated Scheduler gRPC client."""

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
