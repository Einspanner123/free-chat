from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from freechat_contracts import Lifecycle

from freechat_harness_adapters.adapters import HarnessCall, adapter_for


class OpenHandsLifecycle:
    """Translate OpenHands SDK events into explicit FreeChat lifecycle hints."""

    def __init__(
        self,
        *,
        conversation_id: str,
        task_id: str | None = None,
        agent_id: str = "coder",
        branch_id: str = "main",
        expected_resume_ms: int = 30_000,
    ) -> None:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if expected_resume_ms < 0:
            raise ValueError("expected_resume_ms must be non-negative")
        self._call = HarnessCall(
            task_id=task_id or conversation_id,
            session_id=conversation_id,
            agent_id=agent_id,
            branch_id=branch_id,
            lifecycle=Lifecycle.SPAWN,
            call_id=f"{conversation_id}:spawn",
        )
        self._expected_resume_ms = expected_resume_ms
        self._pending_calls: set[str] = set()
        self._finished_calls: set[str] = set()

    @property
    def current(self) -> HarnessCall:
        return self._call

    @property
    def pending_tool_calls(self) -> frozenset[str]:
        return frozenset(self._pending_calls)

    def on_event(self, event: object) -> HarnessCall:
        payload = _event_payload(event)
        kind = _event_kind(event, payload)
        event_id = _required_string(payload, "id", context=f"OpenHands {kind}")

        if self._call.lifecycle in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}:
            return self._call

        if kind == "ActionEvent":
            call_id = _required_string(payload, "tool_call_id", context=kind)
            if call_id in self._finished_calls:
                return self._call
            self._pending_calls.add(call_id)
            self._call = replace(
                self._call,
                turn_id=_optional_string(payload.get("llm_response_id")) or event_id,
                call_id=call_id,
                lifecycle=Lifecycle.TOOL_WAIT,
                expected_resume_ms=self._expected_resume_ms,
            )
        elif kind in {"ObservationEvent", "UserRejectObservation", "AgentErrorEvent"}:
            call_id = _required_string(payload, "tool_call_id", context=kind)
            self._finished_calls.add(call_id)
            self._pending_calls.discard(call_id)
            lifecycle = Lifecycle.TOOL_WAIT if self._pending_calls else Lifecycle.RESUME
            self._call = replace(
                self._call,
                call_id=call_id,
                lifecycle=lifecycle,
                expected_resume_ms=self._expected_resume_ms,
            )
        elif kind == "PauseEvent":
            self._call = replace(
                self._call,
                call_id=event_id,
                lifecycle=Lifecycle.TOOL_WAIT,
                expected_resume_ms=self._expected_resume_ms,
            )
        elif kind in {"InterruptEvent", "ConversationErrorEvent"}:
            self.cancel(call_id=event_id)
        elif kind == "MessageEvent":
            source = payload.get("source")
            if source == "user":
                self._activate(event_id)
        return self._call

    def apply(self, body: dict[str, Any]) -> dict[str, Any]:
        return adapter_for("openhands").apply(body, self._call)

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

    def _activate(self, call_id: str) -> None:
        if self._call.lifecycle in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}:
            return
        self._call = replace(
            self._call,
            call_id=call_id,
            lifecycle=Lifecycle.ACTIVE,
            expected_resume_ms=None,
        )


def _event_payload(event: object) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        return event
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="python")
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("OpenHands event must be a mapping or Pydantic model")


def _event_kind(event: object, payload: Mapping[str, Any]) -> str:
    kind = payload.get("kind") or payload.get("type")
    if isinstance(kind, str) and kind:
        return kind
    class_name = type(event).__name__
    if class_name != "dict":
        return class_name
    raise ValueError("OpenHands event requires kind or type")


def _required_string(values: Mapping[str, Any], key: str, *, context: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires {key}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
