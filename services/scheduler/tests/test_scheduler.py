import pytest
from freechat_contracts import (
    AgentHints,
    ModelCapability,
    RequestProfile,
    WorkerCapabilities,
    WorkerTelemetry,
)
from freechat_scheduler import NoEligibleWorker, Scheduler
from freechat_scheduler.registry import InMemoryWorkerRegistry

MODEL = ModelCapability(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    revision="model-sha",
    tokenizer_revision="tokenizer-sha",
    architecture="dense",
    attention="gqa",
    max_context_tokens=32_768,
    dtype="float16",
)


def add_worker(
    registry: InMemoryWorkerRegistry,
    worker_id: str,
    *,
    node: str,
    queue: int = 0,
    cached: frozenset[str] = frozenset(),
    healthy: bool = True,
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
            free_vram_bytes=20 * 1024**3,
            cached_prefixes=cached,
            estimated_prefill_tokens_per_second=10_000,
            estimated_decode_tokens_per_second=100,
            network_rtt_ms=30,
            network_bandwidth_bytes_per_second=100_000_000,
            cache_load_bytes_per_second=5_000_000_000,
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
    decision = Scheduler(registry).route(profile())
    assert decision.worker_id == "warm"
    assert decision.selected.affinity_credit_ms > 0


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


@pytest.mark.scale
def test_routes_across_sixty_four_workers_deterministically() -> None:
    registry = InMemoryWorkerRegistry()
    for index in range(64):
        add_worker(registry, f"worker-{index:02d}", node=f"node-{index // 4}", queue=index % 4)
    decision = Scheduler(registry).route(profile(local_node_id=None, cache_key=None))
    assert len(decision.candidates) == 64
    assert decision.worker_id == "worker-00"
