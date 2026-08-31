from freechat_contracts import (
    AgentHints,
    CacheActionKind,
    Lifecycle,
    ReuseClass,
)
from freechat_kv_controller import BlockRecord, LifecycleAwarePolicy, TierCosts

POLICY = LifecycleAwarePolicy(
    TierCosts(
        hbm_pressure_ms_per_gib=100,
        cpu_load_bandwidth_bytes_per_second=10 * 1024**3,
        nvme_load_bandwidth_bytes_per_second=2 * 1024**3,
        prefill_tokens_per_second=2_000,
    )
)


def block(hints: AgentHints, *, references: int = 1) -> BlockRecord:
    return BlockRecord(
        block_id="block",
        tenant_id="tenant",
        worker_id="worker",
        worker_generation=3,
        cache_generation=9,
        token_count=8_000,
        bytes=2 * 1024**3,
        reference_count=references,
        current_tier="hbm",
        hints=hints,
    )


def test_shared_immutable_prefix_is_retained() -> None:
    hints = AgentHints(
        harness_id="opencode",
        task_id="task",
        agent_id="lead",
        reuse_class=ReuseClass.IMMUTABLE_SHARED,
        expected_reuse_probability=0.9,
    )
    action = POLICY.choose(block(hints, references=4))
    assert action.kind is CacheActionKind.RETAIN


def test_tool_wait_offloads_when_restore_beats_recompute() -> None:
    hints = AgentHints(
        harness_id="langgraph",
        task_id="task",
        agent_id="agent",
        lifecycle=Lifecycle.TOOL_WAIT,
        expected_resume_ms=5_000,
        expected_reuse_probability=1.0,
        reuse_class=ReuseClass.GROWING_HISTORY,
    )
    action = POLICY.choose(block(hints))
    assert action.kind is CacheActionKind.OFFLOAD_CPU


def test_terminal_private_suffix_is_evicted() -> None:
    hints = AgentHints(
        harness_id="openhands",
        task_id="task",
        agent_id="agent",
        lifecycle=Lifecycle.TERMINAL,
        reuse_class=ReuseClass.GROWING_HISTORY,
    )
    action = POLICY.choose(block(hints, references=0))
    assert action.kind is CacheActionKind.EVICT
    assert action.worker_generation == 3
    assert action.cache_generation == 9
