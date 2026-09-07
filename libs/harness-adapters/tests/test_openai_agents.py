from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from agents import Agent, Model, ModelSettings, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import ModelTracing
from agents.testing import ScriptedModel, assistant_message, function_call
from agents.usage import Usage
from freechat_contracts import Lifecycle
from freechat_harness_adapters import HarnessCall
from freechat_harness_adapters.openai_agents import OpenAIAgentsLifecycle


class RecordingModel(Model):
    def __init__(self) -> None:
        self.settings: list[ModelSettings] = []

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
    ) -> ModelResponse:
        del (
            system_instructions,
            input,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.settings.append(model_settings)
        return ModelResponse(output=[], usage=Usage(), response_id=None)

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
    ) -> Any:
        del (
            system_instructions,
            input,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.settings.append(model_settings)
        yield "event"


async def invoke(model: Model, settings: ModelSettings) -> Any:
    return await model.get_response(
        None,
        [],
        settings,
        [],
        None,
        [],
        ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )


@pytest.mark.asyncio
async def test_hooks_carry_resume_into_next_sdk_model_request() -> None:
    lifecycle = OpenAIAgentsLifecycle(
        HarnessCall(task_id="task", session_id="session", agent_id="planner"),
        expected_resume_ms=500,
    )
    hooks = lifecycle.hooks()
    delegate = RecordingModel()
    model = lifecycle.wrap_model(delegate)
    agent = SimpleNamespace(name="planner")
    tool_context = SimpleNamespace(tool_call_id="tool-call-1")

    await hooks.on_agent_start(None, agent)
    await hooks.on_llm_start(None, agent, None, [])
    await invoke(model, ModelSettings(extra_body={"existing": "preserved"}))
    await hooks.on_tool_start(tool_context, agent, SimpleNamespace(name="read_file"))
    assert lifecycle.current.lifecycle is Lifecycle.TOOL_WAIT
    await hooks.on_tool_end(
        tool_context,
        agent,
        SimpleNamespace(name="read_file"),
        "contents",
    )
    await hooks.on_llm_start(None, agent, None, [])
    await invoke(model, ModelSettings())

    first = cast(dict[str, Any] | None, delegate.settings[0].extra_body)
    second = cast(dict[str, Any] | None, delegate.settings[1].extra_body)
    assert first is not None and first["existing"] == "preserved"
    assert first["freechat"]["agent_hints"]["lifecycle"] == "active"
    assert second is not None
    assert second["freechat"]["agent_hints"]["lifecycle"] == "resume"
    assert second["freechat"]["agent_hints"]["call_id"] == "task:planner:llm:2"
    assert second["freechat"]["agent_hints"]["expected_resume_ms"] == 500


@pytest.mark.asyncio
async def test_streaming_uses_same_sdk_request_extension() -> None:
    lifecycle = OpenAIAgentsLifecycle(
        HarnessCall(task_id="task", session_id="session", agent_id="planner")
    )
    delegate = RecordingModel()
    model = lifecycle.wrap_model(delegate)
    lifecycle.before_llm()

    events = [
        event
        async for event in model.stream_response(
            None,
            [],
            ModelSettings(),
            [],
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    ]

    assert events == ["event"]
    extra_body = cast(dict[str, Any] | None, delegate.settings[0].extra_body)
    assert extra_body is not None
    assert extra_body["freechat"]["agent_hints"]["harness_id"] == "openai-agents"


@pytest.mark.asyncio
async def test_real_sdk_runner_carries_tool_resume_to_second_model_call() -> None:
    @function_tool
    def read_file(path: str) -> str:
        return f"contents:{path}"

    delegate = ScriptedModel(
        [
            [function_call("read_file", {"path": "README.md"}, call_id="tool-1")],
            [assistant_message("done")],
        ]
    )
    lifecycle = OpenAIAgentsLifecycle(
        HarnessCall(task_id="task", session_id="session", agent_id="coder"),
        expected_resume_ms=500,
    )
    agent = Agent(
        name="coder",
        instructions="Read the requested file.",
        tools=[read_file],
        model=lifecycle.wrap_model(delegate),
    )

    result = await Runner.run(agent, "read README.md", hooks=lifecycle.hooks())

    assert result.final_output == "done"
    assert len(delegate.calls) == 2
    first = cast(dict[str, Any] | None, delegate.calls[0].model_settings.extra_body)
    second = cast(dict[str, Any] | None, delegate.calls[1].model_settings.extra_body)
    assert first is not None
    assert first["freechat"]["agent_hints"]["lifecycle"] == "active"
    assert second is not None
    assert second["freechat"]["agent_hints"]["lifecycle"] == "resume"
    assert second["freechat"]["agent_hints"]["expected_resume_ms"] == 500
    assert lifecycle.current.lifecycle is Lifecycle.TERMINAL
