import pytest
from freechat_contracts import Lifecycle, PrefixScope, ReuseClass
from freechat_harness_adapters import ADAPTERS, HarnessCall, adapter_for


@pytest.mark.parametrize("harness_id", sorted(ADAPTERS))
def test_all_harnesses_emit_the_same_lifecycle_depth(harness_id: str) -> None:
    adapter = adapter_for(harness_id)
    body = adapter.apply(
        {"model": "Qwen/Qwen2.5-7B-Instruct", "messages": []},
        HarnessCall(
            task_id="task",
            session_id="session",
            agent_id="coder",
            branch_id="child",
            parent_branch_id="main",
            turn_id="42",
            lifecycle=Lifecycle.TOOL_WAIT,
            expected_resume_ms=30_000,
        ),
    )
    hints = body["freechat"]["agent_hints"]
    assert hints["harness_id"] == harness_id
    assert hints["lifecycle"] == "tool_wait"
    assert hints["prefix_scope"] == PrefixScope.BRANCH
    assert hints["reuse_class"] == ReuseClass.GROWING_HISTORY
    assert hints["expected_resume_ms"] == 30_000


def test_unknown_harness_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported harness"):
        adapter_for("unknown")
