from datetime import UTC, datetime, timedelta

import pytest
from freechat_contracts import (
    AgentHints,
    ModelCapability,
    RequestProfile,
    WorkerCapabilities,
    WorkerTelemetry,
)
from freechat_scheduler import NoEligibleWorker, RoutingStrategy, Scheduler
from freechat_scheduler.registry import InMemoryWorkerRegistry

MODEL = ModelCapability(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    revision="model-sha",
    tokenizer_revision="tokenizer-sha",
    architecture="dense",
    attention="gqa",
    max_context_tokens=32_768,
    dtype="float16",
    supports_kv_offload=True,
    kv_bytes_per_token=16_384,
)


def add_worker(
    registry: InMemoryWorkerRegistry,
    worker_id: str,
    *,
    node: str,
    queue: int = 0,
    active: int = 0,
    cached: frozenset[str] = frozenset(),
    healthy: bool = True,
    free_vram_bytes: int = 20 * 1024**3,
    kv_cache_capacity_bytes: int | None = None,
    kv_cache_free_bytes: int | None = None,
) -> None:
    registry.upsert(
        WorkerCapabilities(
            worker_id=worker_id,
            generation=1,
            endpoint=f"http://{worker_id}:8000",
            node_id=node,
            gpu_id="0",
            gpu_name="RTX A6000",
            compute_capability="8.6",
            total_vram_bytes=48 * 1024**3,
            p2p_domain=node,
            network_domain="lan",
            models=(MODEL,),
        ),
        WorkerTelemetry(
            worker_id=worker_id,
            generation=1,
            queue_depth=queue,
            active_requests=active,
            free_vram_bytes=free_vram_bytes,
            kv_cache_capacity_bytes=kv_cache_capacity_bytes,
            kv_cache_free_bytes=kv_cache_free_bytes,
            cached_prefixes=cached,
            estimated_prefill_tokens_per_second=10_000,
            estimated_decode_tokens_per_second=100,
            network_rtt_ms=30,
            network_bandwidth_bytes_per_second=100_000_000,
            cache_load_bytes_per_second=5_000_000_000,
            cache_store_bytes_per_second=5_000_000_000,
            healthy=healthy,
        ),
    )


def profile(**overrides: object) -> RequestProfile:
    values: dict[str, object] = {
        "tenant_id": "tenant-a",
        "model_id": MODEL.model_id,
        "input_tokens": 8_000,
        "output_tokens": 200,
        "estimated_kv_bytes": 128 * 1024**2,
        "cache_key": "shared-prefix",
        "local_node_id": "ross",
        "hints": AgentHints(harness_id="agents", task_id="task", agent_id="agent"),
    }
    values.update(overrides)
    return RequestProfile(**values)


def test_cache_affinity_can_beat_small_queue_difference() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "warm", node="ross", queue=1, cached=frozenset({"shared-prefix"}))
    add_worker(registry, "cold", node="ross", queue=0)
    hints = AgentHints(
        harness_id="agents",
        task_id="task",
        agent_id="agent",
        expected_reuse_probability=1.0,
    )
    decision = Scheduler(registry).route(profile(hints=hints))
    assert decision.worker_id == "warm"
    assert decision.selected.affinity_credit_ms > 0


def test_round_robin_rotates_only_across_eligible_workers() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker-a", node="ross")
    add_worker(registry, "worker-b", node="ross")
    scheduler = Scheduler(registry, strategy=RoutingStrategy.ROUND_ROBIN)
    assert [scheduler.route(profile()).worker_id for _ in range(3)] == [
        "worker-a",
        "worker-b",
        "worker-a",
    ]


def test_least_load_uses_active_requests_before_queue_depth() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "queued", node="ross", queue=5, active=0)
    add_worker(registry, "active", node="ross", queue=0, active=1)
    decision = Scheduler(registry, strategy=RoutingStrategy.LEAST_LOAD).route(profile())
    assert decision.worker_id == "queued"
    assert decision.strategy == RoutingStrategy.LEAST_LOAD


def test_prefix_affinity_is_a_distinct_reproducible_baseline() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(
        registry,
        "warm",
        node="ross",
        queue=20,
        cached=frozenset({"shared-prefix"}),
    )
    add_worker(registry, "cold", node="ross")
    decision = Scheduler(registry, strategy=RoutingStrategy.PREFIX_AFFINITY).route(profile())
    assert decision.worker_id == "warm"


def test_lifecycle_credit_changes_selection_relative_to_cost_aware() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(
        registry,
        "warm",
        node="ross",
        queue=18,
        cached=frozenset({"shared-prefix"}),
    )
    add_worker(registry, "cold", node="ross")
    hints = AgentHints(
        harness_id="agents",
        task_id="task",
        agent_id="agent",
        expected_reuse_probability=1.0,
    )
    request = profile(hints=hints)
    cost = Scheduler(registry, strategy=RoutingStrategy.COST_AWARE).route(request)
    lifecycle = Scheduler(registry, strategy=RoutingStrategy.LIFECYCLE_AWARE).route(request)
    assert cost.worker_id == "cold"
    assert lifecycle.worker_id == "warm"


def test_predictive_offload_is_enabled_only_when_expected_savings_are_positive() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(
        registry,
        "pressured",
        node="ross",
        kv_cache_capacity_bytes=1024**3,
        kv_cache_free_bytes=32 * 1024**2,
    )
    hints = AgentHints(
        harness_id="agents",
        task_id="task",
        agent_id="agent",
        expected_reuse_probability=0.9,
    )
    decision = Scheduler(registry).route(profile(hints=hints))
    assert decision.kv_transfer.enabled is True
    assert decision.kv_transfer.max_offload_tokens == 8_000
    assert decision.kv_transfer.estimated_kv_bytes == 128 * 1024**2
    assert decision.kv_transfer.expected_net_benefit_ms > 0


def test_predictive_offload_fails_closed_for_non_lifecycle_baseline() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross", free_vram_bytes=2 * 1024**3)
    decision = Scheduler(registry, strategy=RoutingStrategy.LEAST_LOAD).route(profile())
    assert decision.kv_transfer.enabled is False
    assert decision.kv_transfer.max_offload_tokens == 0
    assert decision.kv_transfer.reason == "strategy_does_not_authorize_offload"


def test_predictive_offload_respects_explicit_forbid() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross", free_vram_bytes=2 * 1024**3)
    hints = AgentHints(
        harness_id="agents",
        task_id="task",
        agent_id="agent",
        expected_reuse_probability=1.0,
        allow_kv_offload=False,
    )
    decision = Scheduler(registry).route(profile(hints=hints))
    assert decision.kv_transfer.reason == "offload_forbidden_by_hints"


def test_predictive_offload_requires_reported_store_bandwidth() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(
        registry,
        "worker",
        node="ross",
        kv_cache_capacity_bytes=1024**3,
        kv_cache_free_bytes=32 * 1024**2,
    )
    snapshot = registry.snapshot()[1][0]
    registry.upsert(
        snapshot.capabilities,
        snapshot.telemetry.model_copy(update={"cache_store_bytes_per_second": None}),
    )
    hints = AgentHints(
        harness_id="agents",
        task_id="task",
        agent_id="agent",
        expected_reuse_probability=1.0,
    )
    decision = Scheduler(registry).route(profile(hints=hints))
    assert decision.kv_transfer.enabled is False
    assert decision.kv_transfer.reason == "missing_cache_store_bandwidth"


def test_remote_worker_is_hard_filtered() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "remote", node="workstation")
    hints = AgentHints(
        harness_id="agents",
        task_id="task",
        agent_id="agent",
        allow_remote_worker=False,
    )
    with pytest.raises(NoEligibleWorker) as error:
        Scheduler(registry).route(profile(hints=hints))
    assert error.value.rejected["remote"] == ("remote_worker_forbidden",)


def test_unhealthy_worker_is_explained() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "bad", node="ross", healthy=False)
    with pytest.raises(NoEligibleWorker) as error:
        Scheduler(registry).route(profile())
    assert "worker_unhealthy" in error.value.rejected["bad"]


def test_stale_worker_is_excluded_even_when_marked_healthy() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "stale", node="ross")
    snapshot = registry.snapshot()[1][0]
    registry.upsert(
        snapshot.capabilities,
        snapshot.telemetry.model_copy(update={
            "observed_at": datetime.now(UTC) - timedelta(seconds=31),
        }),
    )
    with pytest.raises(NoEligibleWorker) as error:
        Scheduler(registry).route(profile())
    assert "telemetry_stale" in error.value.rejected["stale"]


@pytest.mark.asyncio
async def test_heartbeat_cannot_replace_newer_or_different_engine_sample() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    snapshot = registry.snapshot()[1][0]
    telemetry = snapshot.telemetry.model_copy(update={"engine_instance_id": "first"})
    await registry.heartbeat(telemetry)
    with pytest.raises(ValueError, match="out_of_order"):
        await registry.heartbeat(telemetry.model_copy(update={
            "observed_at": telemetry.observed_at - timedelta(seconds=1),
        }))
    with pytest.raises(ValueError, match="engine_instance_changed"):
        await registry.heartbeat(telemetry.model_copy(update={"engine_instance_id": "second"}))
    assert registry.snapshot()[1][0].telemetry == telemetry


@pytest.mark.scale
def test_routes_across_sixty_four_workers_deterministically() -> None:
    registry = InMemoryWorkerRegistry()
    for index in range(64):
        add_worker(registry, f"worker-{index:02d}", node=f"node-{index // 4}", queue=index % 4)
    decision = Scheduler(registry).route(profile(local_node_id=None, cache_key=None))
    assert len(decision.candidates) == 64
    assert decision.worker_id == "worker-00"
