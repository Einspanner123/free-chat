"""Exercise native event edge contracts without claiming live Harness acceptance."""

from typing import Any

import pytest
from freechat_contracts import Lifecycle
from freechat_harness_adapters import OpenCodeLifecycle, OpenHandsLifecycle


@pytest.mark.parametrize(
    "factory,identity",
    [
        (OpenCodeLifecycle, "session_id"),
        (OpenHandsLifecycle, "conversation_id"),
    ],
)
@pytest.mark.parametrize("bad", ["identity", "horizon"])
def test_adapter_rejects_invalid_initial_scope(factory: Any, identity: str, bad: str) -> None:
    kwargs = {
        identity: "" if bad == "identity" else "session",
        "expected_resume_ms": -1 if bad == "horizon" else 1,
    }
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize("kind", ["step-start", "step-finish", "text"])
def test_opencode_message_parts(kind: str) -> None:
    adapter = OpenCodeLifecycle(session_id="s")
    state = adapter.on_event(
        {
            "type": "message.part.updated",
            "properties": {"part": {"sessionID": "s", "messageID": "m", "type": kind}},
        }
    )
    assert state.lifecycle == (Lifecycle.SPAWN if kind == "text" else Lifecycle.ACTIVE)


@pytest.mark.parametrize("status", ["busy", "active", "idle", {"type": "busy"}])
def test_opencode_session_status(status: Any) -> None:
    adapter = OpenCodeLifecycle(session_id="s")
    state = adapter.on_event(
        {"type": "session.status", "properties": {"sessionID": "s", "status": status}}
    )
    assert state.lifecycle == (Lifecycle.SPAWN if status == "idle" else Lifecycle.ACTIVE)
    adapter.terminal()
    assert (
        adapter.on_event(
            {"type": "session.status", "properties": {"sessionID": "s", "status": "busy"}}
        ).lifecycle
        == Lifecycle.TERMINAL
    )


@pytest.mark.parametrize(
    "kind,expected",
    [("session.error", Lifecycle.CANCELLED), ("session.deleted", Lifecycle.TERMINAL)],
)
def test_opencode_session_termination(kind: str, expected: Lifecycle) -> None:
    adapter = OpenCodeLifecycle(session_id="s")
    properties = {"sessionID": "s", "info": {"id": "s"}}
    assert adapter.on_event({"type": kind, "properties": properties}).lifecycle == expected
    assert adapter.pending_tool_calls == frozenset()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "session.error", "properties": {"sessionID": "foreign"}},
        {"type": "message.part.updated", "properties": []},
        {"type": ""},
        {"properties": {}},
    ],
)
def test_opencode_malformed_scope_is_rejected(event: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OpenCodeLifecycle(session_id="s").on_event(event)


def test_opencode_unknown_event_keeps_identity() -> None:
    adapter = OpenCodeLifecycle(session_id="s")
    before = adapter.current
    assert adapter.on_event({"type": "unrelated"}) == before


@pytest.mark.parametrize(
    "kind,source,expected",
    [
        ("PauseEvent", "", Lifecycle.TOOL_WAIT),
        ("InterruptEvent", "", Lifecycle.CANCELLED),
        ("MessageEvent", "user", Lifecycle.ACTIVE),
        ("MessageEvent", "assistant", Lifecycle.SPAWN),
        ("Unknown", "", Lifecycle.SPAWN),
    ],
)
def test_openhands_non_tool_events(kind: str, source: str, expected: Lifecycle) -> None:
    adapter = OpenHandsLifecycle(conversation_id="s")
    state = adapter.on_event({"type": kind, "id": "event", "source": source})
    assert state.lifecycle == expected


@pytest.mark.parametrize("event", [None, object(), {}, {"kind": "ActionEvent"}, {"id": "event"}])
def test_openhands_invalid_payloads_fail_closed(event: object) -> None:
    with pytest.raises(ValueError):
        OpenHandsLifecycle(conversation_id="s").on_event(event)


def test_openhands_model_payload_and_class_discriminator() -> None:
    class PauseEvent:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "python"
            return {"id": "pause"}

    adapter = OpenHandsLifecycle(conversation_id="s")
    assert adapter.on_event(PauseEvent()).lifecycle == Lifecycle.TOOL_WAIT

    class InvalidEvent:
        def model_dump(self, *, mode: str) -> list[str]:
            return []

    with pytest.raises(ValueError):
        adapter.on_event(InvalidEvent())
