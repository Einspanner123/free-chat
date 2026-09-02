from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

from agents import Model, ModelSettings, RunHooks
from freechat_contracts import Lifecycle

from freechat_harness_adapters.adapters import HarnessCall, adapter_for


class OpenAIAgentsLifecycle:
    """Run-local bridge between OpenAI Agents callbacks and FreeChat hints.

    Create one instance per SDK run; sharing an instance across concurrent runs
    is intentionally unsupported because SDK hooks can execute in copied async
    contexts. Tool completion marks the next LLM request as ``resume``;
    tool start records ``tool_wait`` for the lifecycle/control path without
    pretending that a completed model request knew which tool would execute.
    """

    def __init__(self, call: HarnessCall, *, expected_resume_ms: int = 30_000) -> None:
        if expected_resume_ms < 0:
            raise ValueError("expected_resume_ms must be non-negative")
        self._initial = call
        self._expected_resume_ms = expected_resume_ms
        self._current = call
        self._turn = 0

    @property
    def current(self) -> HarnessCall:
        return self._current

    def hooks(self) -> OpenAIAgentsRunHooks:
        return OpenAIAgentsRunHooks(self)

    def wrap_model(self, model: Model) -> OpenAIAgentsLifecycleModel:
        return OpenAIAgentsLifecycleModel(model, self)

    def begin_agent(self, agent_name: str | None) -> None:
        current = self._current
        agent_id = agent_name or current.agent_id
        self._current = replace(
            current,
            agent_id=agent_id,
            lifecycle=Lifecycle.SPAWN,
            call_id=f"{current.task_id}:{agent_id}:spawn",
            expected_resume_ms=None,
        )

    def before_llm(self) -> None:
        current = self._current
        self._turn += 1
        turn = self._turn
        lifecycle = (
            Lifecycle.RESUME if current.lifecycle is Lifecycle.RESUME else Lifecycle.ACTIVE
        )
        self._current = replace(
            current,
            turn_id=str(turn),
            call_id=f"{current.task_id}:{current.agent_id}:llm:{turn}",
            lifecycle=lifecycle,
            expected_resume_ms=(
                self._expected_resume_ms if lifecycle is Lifecycle.RESUME else None
            ),
        )

    def tool_wait(self, tool_call_id: str | None) -> None:
        current = self._current
        call_id = tool_call_id or current.call_id
        self._current = replace(
            current,
            call_id=call_id,
            lifecycle=Lifecycle.TOOL_WAIT,
            expected_resume_ms=self._expected_resume_ms,
        )

    def resume(self, tool_call_id: str | None) -> None:
        current = self._current
        call_id = tool_call_id or current.call_id
        self._current = replace(
            current,
            call_id=call_id,
            lifecycle=Lifecycle.RESUME,
            expected_resume_ms=self._expected_resume_ms,
        )

    def terminal(self) -> None:
        self._current = replace(
            self._current,
            lifecycle=Lifecycle.TERMINAL,
            expected_resume_ms=None,
        )

    def cancel(self) -> None:
        self._current = replace(
            self._current,
            lifecycle=Lifecycle.CANCELLED,
            expected_resume_ms=None,
        )

    def reset(self) -> None:
        self._current = self._initial
        self._turn = 0


class OpenAIAgentsRunHooks(RunHooks[Any]):
    def __init__(self, lifecycle: OpenAIAgentsLifecycle) -> None:
        self._lifecycle = lifecycle

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        del context
        self._lifecycle.begin_agent(getattr(agent, "name", None))

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        del context, agent, system_prompt, input_items
        self._lifecycle.before_llm()

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        del agent, tool
        self._lifecycle.tool_wait(getattr(context, "tool_call_id", None))

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: object,
    ) -> None:
        del agent, tool, result
        self._lifecycle.resume(getattr(context, "tool_call_id", None))

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        del context, from_agent
        self._lifecycle.begin_agent(getattr(to_agent, "name", None))

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        del context, agent, output
        self._lifecycle.terminal()


class OpenAIAgentsLifecycleModel(Model):
    """Model decorator that adds run-local FreeChat hints to SDK requests."""

    def __init__(self, delegate: Model, lifecycle: OpenAIAgentsLifecycle) -> None:
        self._delegate = delegate
        self._lifecycle = lifecycle
        self._adapter = adapter_for("openai-agents")

    def _settings(self, settings: ModelSettings) -> ModelSettings:
        raw_extra_body = cast(dict[str, Any] | None, settings.extra_body)
        extra_body = dict(raw_extra_body or {})
        enriched = self._adapter.apply(extra_body, self._lifecycle.current)
        return replace(settings, extra_body=cast(Any, enriched))

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> Any:
        return await self._delegate.get_response(
            system_instructions,
            input,
            self._settings(model_settings),
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        async for event in self._delegate.stream_response(
            system_instructions,
            input,
            self._settings(model_settings),
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        ):
            yield event

    async def close(self) -> None:
        await self._delegate.close()

    def get_retry_advice(self, request: Any) -> Any:
        return self._delegate.get_retry_advice(request)
