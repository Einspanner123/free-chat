from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from freechat_contracts import Lifecycle

from freechat_harness_adapters.adapters import HarnessCall, adapter_for


def bind_langgraph_task(
    config: Mapping[str, Any],
    *,
    task_id: str,
) -> dict[str, Any]:
    """Attach FreeChat task identity to a checkpoint config without mutation.

    LangGraph checkpoint snapshots retain the thread/checkpoint coordinates but
    may omit application-specific configurable values. Call this when resuming
    from ``graph.get_state(...).config`` so task identity is explicit.
    """

    if not task_id:
        raise ValueError("task_id is required")
    bound = dict(config)
    configurable = dict(_mapping(config.get("configurable")))
    configurable["freechat_task_id"] = task_id
    checkpoint_id = _optional_string(configurable.get("checkpoint_id"))
    if checkpoint_id is not None:
        configurable["freechat_checkpoint_id"] = checkpoint_id
    bound["configurable"] = configurable
    return bound


@dataclass(frozen=True, slots=True)
class LangGraphRequestContext:
    """Stable FreeChat identity derived from a LangGraph runnable config."""

    task_id: str
    thread_id: str
    checkpoint_id: str
    checkpoint_namespace: str
    node: str

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        node: str | None = None,
    ) -> LangGraphRequestContext:
        configurable = _mapping(config.get("configurable"))
        metadata = _mapping(config.get("metadata"))
        thread_id = _required_string(configurable, "thread_id")
        checkpoint_id = _optional_string(
            configurable.get("freechat_checkpoint_id")
        ) or _optional_string(configurable.get("checkpoint_id"))
        checkpoint_namespace = (
            _optional_string(configurable.get("checkpoint_ns"))
            or _optional_string(metadata.get("langgraph_checkpoint_ns"))
            or "main"
        )
        node_name = node or _optional_string(metadata.get("langgraph_node")) or "graph"
        task_id = _optional_string(configurable.get("freechat_task_id")) or thread_id
        return cls(
            task_id=task_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id or "pending-checkpoint",
            checkpoint_namespace=checkpoint_namespace,
            node=node_name,
        )

    def apply(
        self,
        body: dict[str, Any],
        *,
        lifecycle: Lifecycle,
        expected_resume_ms: int | None = None,
    ) -> dict[str, Any]:
        call = HarnessCall(
            task_id=self.task_id,
            session_id=self.thread_id,
            agent_id=self.node,
            branch_id=self.checkpoint_namespace,
            turn_id=self.checkpoint_id,
            call_id=f"{self.checkpoint_id}:{self.node}:{lifecycle.value}",
            lifecycle=lifecycle,
            expected_resume_ms=expected_resume_ms,
        )
        return adapter_for("langgraph").apply(body, call)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = _optional_string(values.get(key))
    if value is None:
        raise ValueError(f"LangGraph config requires configurable.{key}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
