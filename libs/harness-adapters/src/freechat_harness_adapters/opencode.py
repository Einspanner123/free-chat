from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from freechat_contracts import Lifecycle

from freechat_harness_adapters.adapters import HarnessCall, adapter_for

_TOOL_STATUS_RANK = {"pending": 0, "running": 1, "completed": 2, "error": 2}


class OpenCodeLifecycle:
    """Translate OpenCode session events into request-scoped lifecycle hints.

    The bridge is intentionally transport-neutral: an OpenCode plugin or SSE
    consumer feeds native event dictionaries to :meth:`on_event`, then calls
    :meth:`apply` immediately before the corresponding model request. One
    instance represents one OpenCode session.
    """

    def __init__(
        self,
        *,
        session_id: str,
        task_id: str | None = None,
        agent_id: str = "coder",
        branch_id: str = "main",
        expected_resume_ms: int = 30_000,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        if expected_resume_ms < 0:
            raise ValueError("expected_resume_ms must be non-negative")
        self._call = HarnessCall(
            task_id=task_id or session_id,
            session_id=session_id,
            agent_id=agent_id,
            branch_id=branch_id,
            lifecycle=Lifecycle.SPAWN,
            call_id=f"{session_id}:spawn",
        )
        self._expected_resume_ms = expected_resume_ms
        self._tool_status: dict[str, int] = {}
        self._pending_calls: set[str] = set()

    @property
    def current(self) -> HarnessCall:
        return self._call

    @property
    def pending_tool_calls(self) -> frozenset[str]:
        return frozenset(self._pending_calls)

    def on_event(self, event: Mapping[str, Any]) -> HarnessCall:
        event_type = _required_string(event, "type", context="OpenCode event")
        properties = _mapping(event.get("properties"))

        if event_type == "message.part.updated":
            part = _mapping(properties.get("part"))
            self._require_session(part)
            part_type = _required_string(part, "type", context=event_type)
            if part_type == "tool":
                self._on_tool_part(part)
            elif part_type == "step-start" or part_type == "step-finish":
                self._activate(_required_string(part, "messageID", context=part_type))
            return self._call

        if event_type == "session.status":
            self._require_session(properties)
            status = properties.get("status")
            status_type = (
                status.get("type") if isinstance(status, Mapping) else status
            )
            if status_type in {"busy", "active"}:
                self._activate(f"{self._call.session_id}:active")
            return self._call

        if event_type == "session.error":
            session_id = properties.get("sessionID")
            if session_id is None:
                return self._call
            if session_id != self._call.session_id:
                raise ValueError("OpenCode event belongs to a different session")
            self.cancel(call_id=f"{self._call.session_id}:error")
            return self._call

        if event_type == "session.deleted":
            info = _mapping(properties.get("info"))
            self._require_session(info, key="id")
            self.terminal(call_id=f"{self._call.session_id}:deleted")
            return self._call

        return self._call

    def apply(self, body: dict[str, Any]) -> dict[str, Any]:
        return adapter_for("opencode").apply(body, self._call)

    def terminal(self, *, call_id: str | None = None) -> None:
        self._pending_calls.clear()
        self._call = replace(
            self._call,
            lifecycle=Lifecycle.TERMINAL,
            call_id=call_id or self._call.call_id,
            expected_resume_ms=None,
        )

    def cancel(self, *, call_id: str | None = None) -> None:
        self._pending_calls.clear()
        self._call = replace(
            self._call,
            lifecycle=Lifecycle.CANCELLED,
            call_id=call_id or self._call.call_id,
            expected_resume_ms=None,
        )

    def _on_tool_part(self, part: Mapping[str, Any]) -> None:
        if self._call.lifecycle in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}:
            return
        call_id = _required_string(part, "callID", context="OpenCode tool part")
        message_id = _required_string(part, "messageID", context="OpenCode tool part")
        state = _mapping(part.get("state"))
        status = _required_string(state, "status", context="OpenCode tool state")
        try:
            rank = _TOOL_STATUS_RANK[status]
        except KeyError as error:
            raise ValueError(f"unsupported OpenCode tool status: {status}") from error

        previous_rank = self._tool_status.get(call_id, -1)
        if rank < previous_rank:
            return
        self._tool_status[call_id] = rank
        if rank < 2:
            self._pending_calls.add(call_id)
            self._call = replace(
                self._call,
                turn_id=message_id,
                call_id=call_id,
                lifecycle=Lifecycle.TOOL_WAIT,
                expected_resume_ms=self._expected_resume_ms,
            )
            return

        self._pending_calls.discard(call_id)
        lifecycle = Lifecycle.TOOL_WAIT if self._pending_calls else Lifecycle.RESUME
        self._call = replace(
            self._call,
            turn_id=message_id,
            call_id=call_id,
            lifecycle=lifecycle,
            expected_resume_ms=self._expected_resume_ms,
        )

    def _activate(self, call_id: str) -> None:
        if self._call.lifecycle in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}:
            return
        self._call = replace(
            self._call,
            call_id=call_id,
            lifecycle=Lifecycle.ACTIVE,
            expected_resume_ms=None,
        )

    def _require_session(self, values: Mapping[str, Any], *, key: str = "sessionID") -> None:
        if _required_string(values, key, context="OpenCode event") != self._call.session_id:
            raise ValueError("OpenCode event belongs to a different session")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_string(values: Mapping[str, Any], key: str, *, context: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires {key}")
    return value
