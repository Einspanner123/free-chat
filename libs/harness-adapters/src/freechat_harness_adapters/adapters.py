from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freechat_contracts import AgentHints, Lifecycle, PrefixScope, ReuseClass


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
        return AgentHints(
            harness_id=self.harness_id,
            task_id=call.task_id,
            session_id=call.session_id,
            agent_id=call.agent_id,
            parent_agent_id=call.parent_agent_id,
            branch_id=call.branch_id,
            parent_branch_id=call.parent_branch_id,
            turn_id=call.turn_id,
            call_id=call.call_id or f"{call.task_id}:{call.agent_id}:{call.turn_id or 'call'}",
            lifecycle=call.lifecycle,
            prefix_scope=prefix_scope,
            reuse_class=reuse_class,
            expected_reuse_probability=0.9 if reuse_class is ReuseClass.GROWING_HISTORY else 0.0,
            expected_resume_ms=call.expected_resume_ms,
            priority=call.priority,
            deadline_ms=call.deadline_ms,
            privacy_domain=call.privacy_domain,
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
