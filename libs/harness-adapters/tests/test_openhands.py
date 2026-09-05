from __future__ import annotations

import pytest
from freechat_contracts import Lifecycle
from freechat_harness_adapters import OpenHandsLifecycle


def event(kind: str, event_id: str, **fields: str) -> dict[str, str]:
    return {"kind": kind, "id": event_id, **fields}


def test_action_observation_pairs_preserve_parallel_tool_wait() -> None:
    lifecycle = OpenHandsLifecycle(
        conversation_id="conv_1", task_id="task_1", expected_resume_ms=750
    )
    lifecycle.on_event(
        event(
            "ActionEvent",
            "action_1",
            tool_call_id="call_1",
            llm_response_id="response_1",
        )
    )
    lifecycle.on_event(
        event(
            "ActionEvent",
            "action_2",
            tool_call_id="call_2",
            llm_response_id="response_1",
        )
    )
    lifecycle.on_event(
        event("ObservationEvent", "observation_1", tool_call_id="call_1")
    )
    assert lifecycle.current.lifecycle is Lifecycle.TOOL_WAIT
    assert lifecycle.pending_tool_calls == frozenset({"call_2"})

    lifecycle.on_event(
        event("AgentErrorEvent", "error_2", tool_call_id="call_2")
    )
    hints = lifecycle.apply({})["freechat"]["agent_hints"]
    assert hints["harness_id"] == "openhands"
    assert hints["task_id"] == "task_1"
    assert hints["session_id"] == "conv_1"
    assert hints["turn_id"] == "response_1"
    assert hints["lifecycle"] == "resume"
    assert hints["expected_resume_ms"] == 750


def test_duplicate_or_late_action_cannot_reopen_finished_call() -> None:
    lifecycle = OpenHandsLifecycle(conversation_id="conv_1")
    lifecycle.on_event(
        event("ObservationEvent", "observation_1", tool_call_id="call_1")
    )
    lifecycle.on_event(
        event("ActionEvent", "action_1", tool_call_id="call_1", llm_response_id="r1")
    )
    assert lifecycle.current.lifecycle is Lifecycle.RESUME
    assert lifecycle.pending_tool_calls == frozenset()


def test_error_cancels_and_malformed_action_fails_closed() -> None:
    lifecycle = OpenHandsLifecycle(conversation_id="conv_1")
    lifecycle.on_event(event("ConversationErrorEvent", "error_1"))
    assert lifecycle.current.lifecycle is Lifecycle.CANCELLED

    malformed = OpenHandsLifecycle(conversation_id="conv_1")
    with pytest.raises(ValueError, match="tool_call_id"):
        malformed.on_event(event("ActionEvent", "action_1"))


def test_terminal_state_is_sticky_during_late_event_replay() -> None:
    lifecycle = OpenHandsLifecycle(conversation_id="conv_1")
    lifecycle.terminal(call_id="conversation-finished")
    lifecycle.on_event(
        event("ActionEvent", "late", tool_call_id="late", llm_response_id="r1")
    )
    assert lifecycle.current.lifecycle is Lifecycle.TERMINAL
    assert lifecycle.pending_tool_calls == frozenset()
