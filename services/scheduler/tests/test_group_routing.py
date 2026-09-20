from datetime import UTC, datetime, timedelta

import pytest
from freechat_contracts import (
    AgentHints,
    ModelCapability,
    RequestProfile,
    WorkerCapabilities,
    WorkerTelemetry,
)
from freechat_scheduler.local_cluster import validate_layout
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.resource_groups import GroupLease, GroupLedger, GroupSpec, GroupState
from freechat_scheduler.scheduler import NoEligibleWorker, Scheduler


@pytest.mark.parametrize("tp,groups", [(1, 12), (2, 6), (4, 3)])
async def test_loopback_grpc_register_heartbeat_route_and_release(tp: int, groups: int) -> None:
    result = await validate_layout(tp)
    assert result["resource_groups"] == groups
    assert result["routed_requests"] == 2 * groups
    assert not result["hardware_verified"]


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing_view",
        "stale_view",
        "naive_view",
        "future_view",
        "no_group",
        "not_ready",
        "expired",
        "generation",
        "worker_generation",
        "worker_id",
        "gpu_ids",
        "instance",
        "revision",
        "tp",
        "pp",
        "node",
        "missing_binding",
    ],
)
def test_group_routing_gates(failure: str | None) -> None:
    now = datetime.now(UTC)
    group = GroupSpec(
        group_id="group",
        worker_id="worker",
        gpu_ids=("node/0", "node/1"),
        model_id="model",
        model_revision="revision",
        tensor_parallel_size=2,
        memory_per_gpu_bytes=1024,
    )
    lease = GroupLease(
        spec=group,
        owner_id="owner",
        node_id="node",
        generation=7,
        state=GroupState.READY,
        expires_at=now + timedelta(seconds=60),
        engine_instance_id="engine",
    )
    model = ModelCapability(
        model_id="model",
        revision="revision",
        tokenizer_revision="tokenizer",
        architecture="dense",
        attention="gqa",
        max_context_tokens=4096,
        dtype="bf16",
        tensor_parallel_size=2,
        kv_admission_bytes_per_token_per_rank=16,
        kv_block_size_tokens=16,
    )
    caps = WorkerCapabilities(
        worker_id="worker",
        generation=7,
        endpoint="http://fixture",
        node_id="node",
        gpu_id="0",
        gpu_ids=group.gpu_ids,
        gpu_name="SIMULATED",
        compute_capability="9.0",
        total_vram_bytes=4096,
        p2p_domain="node",
        network_domain="unknown",
        models=(model,),
        resource_group_id="group",
        resource_group_generation=7,
    )
    telemetry = WorkerTelemetry(
        worker_id="worker",
        generation=7,
        free_vram_bytes=4096,
        engine_instance_id="engine",
        kv_admission_available_bytes_per_rank=4096,
    )
    if failure == "stale_view":
        now -= timedelta(seconds=6)
    elif failure == "naive_view":
        now = now.replace(tzinfo=None)
    elif failure == "future_view":
        now += timedelta(seconds=1)
    elif failure == "not_ready":
        lease = lease.model_copy(update={"state": GroupState.DRAINING})
    elif failure == "expired":
        lease = lease.model_copy(update={"expires_at": now})
    elif failure == "generation":
        caps = caps.model_copy(update={"resource_group_generation": 8})
    elif failure == "worker_generation":
        caps = caps.model_copy(update={"generation": 8})
        telemetry = telemetry.model_copy(update={"generation": 8})
    elif failure == "worker_id":
        lease = lease.model_copy(update={"spec": group.model_copy(update={"worker_id": "other"})})
    elif failure == "gpu_ids":
        caps = caps.model_copy(update={"gpu_ids": ("node/0", "node/0")})
    elif failure == "instance":
        telemetry = telemetry.model_copy(update={"engine_instance_id": "other"})
    elif failure in {"revision", "tp", "pp"}:
        updates: dict[str, dict[str, object]] = {
            "revision": {"revision": "other"},
            "tp": {"tensor_parallel_size": 4},
            "pp": {"pipeline_parallel_size": 2},
        }
        caps = caps.model_copy(update={"models": (model.model_copy(update=updates[failure]),)})
    elif failure == "node":
        caps = caps.model_copy(update={"node_id": "other"})
    elif failure == "missing_binding":
        caps = caps.model_copy(update={"resource_group_id": None})
    ledger = GroupLedger(
        inventory_hash="fixture", groups={} if failure == "no_group" else {"group": lease}
    )
    registry = InMemoryWorkerRegistry()
    registry.upsert(caps, telemetry)
    scheduler = Scheduler(
        registry, group_snapshot=None if failure == "missing_view" else lambda: (now, ledger)
    )
    request = RequestProfile(
        tenant_id="tenant",
        model_id="model",
        input_tokens=32,
        output_tokens=16,
        hints=AgentHints(harness_id="test", task_id="task", agent_id="agent"),
    )
    if failure is None:
        assert scheduler.route(request).worker_id == "worker"
    else:
        with pytest.raises(NoEligibleWorker) as error:
            scheduler.route(request)
        assert any(reason.startswith("resource_group") for reason in error.value.rejected["worker"])
