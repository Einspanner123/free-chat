from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from freechat_contracts import AgentHints, Lifecycle, PrefixScope, ReuseClass


@dataclass(frozen=True, slots=True)
class ReuseForecast:
    """Caller-provided future reuse estimate, not an inferred lifecycle guarantee."""

    call_id: str
    probability: float
    after_ms: int
    observed_at: datetime
    expires_at: datetime
    evidence_reference: str
    task_id: str
    session_id: str
    agent_id: str
    branch_id: str = "main"

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.evidence_reference.strip():
            raise ValueError("forecast requires call identity and evidence reference")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("invalid forecast probability")
        if isinstance(self.after_ms, bool) or not isinstance(self.after_ms, int):
            raise ValueError("forecast horizon must be integral milliseconds")
        if not 0 <= self.after_ms <= 86_400_000:
            raise ValueError("invalid forecast horizon")
        if self.observed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("forecast timestamps require timezone")
        if not 0 < (self.expires_at - self.observed_at).total_seconds() <= 3600:
            raise ValueError("forecast validity must be positive and at most one hour")

    def applicable(self, call: HarnessCall, now: datetime) -> bool:
        return (self.task_id, self.session_id, self.agent_id, self.branch_id, self.call_id) == (
            call.task_id,
            call.session_id,
            call.agent_id,
            call.branch_id,
            call.call_id,
        ) and self.observed_at <= now < self.expires_at


@dataclass(frozen=True, slots=True)
class HarnessCall:
    task_id: str
    session_id: str
    agent_id: str
    branch_id: str = "main"
    parent_agent_id: str | None = None
    parent_branch_id: str | None = None
    turn_id: str | None = None
    call_id: str | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    expected_resume_ms: int | None = None
    priority: int = 0
    deadline_ms: int | None = None
    privacy_domain: str = "default"
    reuse_forecast: ReuseForecast | None = None


class HarnessAdapter:
    """Framework-specific naming at the edge, one semantic contract inside."""

    def __init__(self, harness_id: str, identity_keys: tuple[str, ...]) -> None:
        self.harness_id = harness_id
        self.identity_keys = identity_keys

    def hints(self, call: HarnessCall) -> AgentHints:
        prefix_scope = PrefixScope.BRANCH if call.parent_branch_id else PrefixScope.AGENT
        reuse_class = (
            ReuseClass.GROWING_HISTORY
            if call.lifecycle not in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}
            else ReuseClass.EPHEMERAL_REASONING
        )
        call_id = call.call_id or f"{call.task_id}:{call.agent_id}:{call.turn_id or 'call'}"
        forecast = call.reuse_forecast
        usable = (
            forecast is not None
            and forecast.applicable(call, datetime.now(UTC))
            and call.lifecycle not in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}
        )
        probability = 0.0
        resume_ms = call.expected_resume_ms
        metadata = {"reuse_forecast_status": "unavailable"}
        if usable and forecast is not None:
            probability = forecast.probability
            resume_ms = forecast.after_ms
            metadata = {
                "reuse_forecast_status": "caller_supplied",
                "reuse_forecast_call_id": forecast.call_id,
                "reuse_forecast_task_id": forecast.task_id,
                "reuse_forecast_session_id": forecast.session_id,
                "reuse_forecast_agent_id": forecast.agent_id,
                "reuse_forecast_branch_id": forecast.branch_id,
                "reuse_forecast_observed_at": forecast.observed_at.isoformat(),
                "reuse_forecast_expires_at": forecast.expires_at.isoformat(),
                "reuse_forecast_evidence_reference": forecast.evidence_reference,
            }
        return AgentHints(
            harness_id=self.harness_id,
            task_id=call.task_id,
            session_id=call.session_id,
            agent_id=call.agent_id,
            parent_agent_id=call.parent_agent_id,
            branch_id=call.branch_id,
            parent_branch_id=call.parent_branch_id,
            turn_id=call.turn_id,
            call_id=call_id,
            lifecycle=call.lifecycle,
            prefix_scope=prefix_scope,
            reuse_class=reuse_class,
            expected_reuse_probability=probability,
            expected_resume_ms=resume_ms,
            priority=call.priority,
            deadline_ms=call.deadline_ms,
            privacy_domain=call.privacy_domain,
            metadata=metadata,
        )

    def apply(self, body: dict[str, Any], call: HarnessCall) -> dict[str, Any]:
        request = dict(body)
        extension = dict(request.get("freechat", {}))
        extension["agent_hints"] = self.hints(call).model_dump(mode="json")
        request["freechat"] = extension
        return request


ADAPTERS: dict[str, HarnessAdapter] = {
    "openai-agents": HarnessAdapter("openai-agents", ("run_id", "agent_name", "tool_call_id")),
    "langgraph": HarnessAdapter("langgraph", ("thread_id", "checkpoint_id", "task_id")),
    "opencode": HarnessAdapter("opencode", ("session_id", "message_id", "part_id")),
    "openhands": HarnessAdapter("openhands", ("conversation_id", "event_id", "tool_call_id")),
}


def adapter_for(harness_id: str) -> HarnessAdapter:
    try:
        return ADAPTERS[harness_id]
    except KeyError as error:
        raise ValueError(f"unsupported harness: {harness_id}") from error
