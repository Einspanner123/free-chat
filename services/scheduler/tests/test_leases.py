import asyncio

import pytest
from freechat_contracts import AgentHints, CandidateCost, RequestProfile, RouteDecision
from freechat_scheduler.grpc_server import LeaseBook


def decision() -> RouteDecision:
    cost = CandidateCost(
        worker_id="worker",
        queue_ms=0,
        prefill_ms=1,
        decode_ms=2,
        cache_ms=0,
        network_ms=0,
        cold_start_ms=0,
        deadline_risk=0,
        eviction_externality=0,
        affinity_credit_ms=0,
        total_ms=3,
    )
    return RouteDecision(
        request_id="request",
        worker_id="worker",
        worker_generation=4,
        endpoint="http://worker",
        selected=cost,
        candidates=(cost,),
        rejected={},
        topology_generation=2,
        reserved_kv_bytes_per_rank=1024,
    )


def request() -> RequestProfile:
    return RequestProfile(
        request_id="request",
        tenant_id="tenant-a",
        model_id="fixture",
        input_tokens=1,
        output_tokens=1,
        hints=AgentHints(harness_id="test", task_id="task", agent_id="agent"),
    )


def test_lease_generation_fences_stale_release() -> None:
    async def scenario() -> None:
        leases = LeaseBook()
        route = decision()
        await leases.reserve(request(), "request", lambda _: route)
        with pytest.raises(ValueError, match="fencing"):
            await leases.release(route.decision_id, route.worker_id, 3, "tenant-a")
        assert await leases.require(route.decision_id, route.worker_id, 4, "tenant-a") == route

    asyncio.run(scenario())
