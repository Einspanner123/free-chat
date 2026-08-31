import pytest
from freechat_contracts import CandidateCost, RouteDecision
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
    )


def test_lease_generation_fences_stale_release() -> None:
    leases = LeaseBook()
    route = decision()
    leases.remember(route)
    with pytest.raises(ValueError, match="fencing"):
        leases.release(route.decision_id, route.worker_id, 3)
    assert leases.require(route.decision_id, route.worker_id, 4) == route
