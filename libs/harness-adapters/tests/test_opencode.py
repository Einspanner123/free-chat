from __future__ import annotations

import pytest
from freechat_contracts import Lifecycle
from freechat_harness_adapters import OpenCodeLifecycle


def tool_event(*, call_id: str, status: str, session_id: str = "ses_1") -> dict:
    return {
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": f"prt_{call_id}",
                "sessionID": session_id,
                "messageID": "msg_1",
                "type": "tool",
                "callID": call_id,
                "tool": "read",
                "state": {"status": status},
            }
        },
    }


def test_native_tool_updates_preserve_wait_until_parallel_calls_finish() -> None:
    lifecycle = OpenCodeLifecycle(
        session_id="ses_1", task_id="task_1", expected_resume_ms=500
    )

    lifecycle.on_event(tool_event(call_id="call_1", status="running"))
    lifecycle.on_event(tool_event(call_id="call_2", status="pending"))
    lifecycle.on_event(tool_event(call_id="call_1", status="completed"))
    assert lifecycle.current.lifecycle is Lifecycle.TOOL_WAIT
    assert lifecycle.pending_tool_calls == frozenset({"call_2"})

    lifecycle.on_event(tool_event(call_id="call_2", status="error"))
    body = lifecycle.apply({"model": "local"})
    hints = body["freechat"]["agent_hints"]
    assert hints["harness_id"] == "opencode"
    assert hints["task_id"] == "task_1"
    assert hints["session_id"] == "ses_1"
    assert hints["turn_id"] == "msg_1"
    assert hints["call_id"] == "call_2"
    assert hints["lifecycle"] == "resume"
    assert hints["expected_resume_ms"] == 500


def test_replayed_stale_running_update_cannot_reopen_completed_call() -> None:
    lifecycle = OpenCodeLifecycle(session_id="ses_1", expected_resume_ms=500)
    lifecycle.on_event(tool_event(call_id="call_1", status="completed"))
    lifecycle.on_event(tool_event(call_id="call_1", status="running"))
    assert lifecycle.current.lifecycle is Lifecycle.RESUME
    assert lifecycle.pending_tool_calls == frozenset()


def test_cross_session_and_unknown_tool_status_fail_closed() -> None:
    lifecycle = OpenCodeLifecycle(session_id="ses_1")
    with pytest.raises(ValueError, match="different session"):
        lifecycle.on_event(
            tool_event(call_id="call_1", status="running", session_id="ses_other")
        )
    with pytest.raises(ValueError, match="unsupported OpenCode tool status"):
        lifecycle.on_event(tool_event(call_id="call_1", status="mystery"))


def test_terminal_state_is_sticky_during_late_event_replay() -> None:
    lifecycle = OpenCodeLifecycle(session_id="ses_1")
    lifecycle.terminal(call_id="session-finished")
    lifecycle.on_event(tool_event(call_id="late", status="running"))
    assert lifecycle.current.lifecycle is Lifecycle.TERMINAL
    assert lifecycle.pending_tool_calls == frozenset()


def test_global_error_without_session_identity_does_not_cancel_session() -> None:
    lifecycle = OpenCodeLifecycle(session_id="ses_1")
    lifecycle.on_event({"type": "session.error", "properties": {"error": {}}})
    assert lifecycle.current.lifecycle is Lifecycle.SPAWN
