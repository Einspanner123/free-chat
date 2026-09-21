from dataclasses import replace

import pytest
from freechat_contracts import AgentHints, CacheActionKind, HintSource, Lifecycle, ReuseClass
from freechat_kv_controller import LifecycleAwarePolicy, TierCosts
from test_policy import block


@pytest.mark.parametrize(
    "kind", ["ephemeral", "forbidden", "cpu", "nvme", "retain", "evict", "inferred"]
)
def test_policy_cost_and_permission_boundaries(kind: str) -> None:
    hints = AgentHints(
        harness_id="fixture",
        task_id="t",
        agent_id="a",
        lifecycle=Lifecycle.TOOL_WAIT if kind in {"forbidden", "cpu", "nvme"} else Lifecycle.ACTIVE,
        expected_resume_ms=1000 if kind in {"forbidden", "cpu", "nvme"} else None,
        expected_reuse_probability=1 if kind != "evict" else 0,
        allow_kv_offload=kind != "forbidden",
        reuse_class=ReuseClass.EPHEMERAL_REASONING if kind == "ephemeral" else ReuseClass.UNKNOWN,
        source=HintSource.INFERRED if kind == "inferred" else HintSource.EXPLICIT,
        confidence=0 if kind == "inferred" else 1,
    )
    costs = TierCosts(
        hbm_pressure_ms_per_gib=100,
        cpu_load_bandwidth_bytes_per_second=10 * 1024**3,
        nvme_load_bandwidth_bytes_per_second=20 * 1024**3,
        prefill_tokens_per_second=2000,
    )
    if kind == "nvme":
        costs = replace(costs, cpu_load_bandwidth_bytes_per_second=1024**3)
    expected = {
        "ephemeral": CacheActionKind.EVICT,
        "forbidden": CacheActionKind.RETAIN,
        "cpu": CacheActionKind.OFFLOAD_CPU,
        "nvme": CacheActionKind.OFFLOAD_NVME,
        "retain": CacheActionKind.RETAIN,
        "evict": CacheActionKind.EVICT,
        "inferred": CacheActionKind.EVICT,
    }
    action = LifecycleAwarePolicy(costs).choose(block(hints))
    assert action.kind == expected[kind]
    assert action.estimated_cost_ms >= 0
    assert (action.worker_generation, action.cache_generation) == (3, 9)
