from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from agents import Agent, Model, ModelSettings, Runner, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from freechat_harness_adapters import HarnessCall
from freechat_harness_adapters.openai_agents import OpenAIAgentsLifecycle
from openai import AsyncOpenAI


class RecordingModel(Model):
    def __init__(self, delegate: Model) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def _record(self, settings: ModelSettings) -> None:
        extra_body = cast(dict[str, Any] | None, settings.extra_body)
        self.calls.append({"extra_body": extra_body or {}})

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
        self._record(model_settings)
        return await self._delegate.get_response(
            system_instructions,
            input,
            model_settings,
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
        self._record(model_settings)
        async for event in self._delegate.stream_response(
            system_instructions,
            input,
            model_settings,
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


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    client = AsyncOpenAI(
        api_key=arguments.api_key,
        base_url=arguments.gateway.rstrip("/") + "/v1",
        max_retries=0,
        timeout=arguments.timeout,
    )
    delegate = RecordingModel(
        OpenAIChatCompletionsModel(model=arguments.model, openai_client=client)
    )
    lifecycle = OpenAIAgentsLifecycle(
        HarnessCall(
            task_id=arguments.task_id,
            session_id=arguments.session_id,
            agent_id="coder",
        ),
        expected_resume_ms=arguments.expected_resume_ms,
    )
    tool_paths: list[str] = []
    expected_heading = next(
        line.strip()
        for line in (arguments.repository / "README.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )

    @function_tool
    def read_file(path: str) -> str:
        """Read a UTF-8 repository file after constraining it to the repository root."""
        requested = (arguments.repository / path).resolve()
        repository = arguments.repository.resolve()
        if not requested.is_relative_to(repository):
            raise ValueError("path escapes repository root")
        tool_paths.append(path)
        return str(
            requested.read_text(encoding="utf-8")[: arguments.tool_output_characters]
        )

    agent = Agent(
        name="coder",
        instructions=(
            "You are a coding agent. You must call read_file on README.md before answering. "
            "After the tool returns, state only the first Markdown heading from that file."
        ),
        tools=[read_file],
        model=lifecycle.wrap_model(delegate),
        model_settings=ModelSettings(temperature=0, max_tokens=64),
    )
    started_ns = time.perf_counter_ns()
    try:
        result = await Runner.run(agent, "Inspect README.md.", hooks=lifecycle.hooks())
    finally:
        await client.close()
    finished_ns = time.perf_counter_ns()
    lifecycles = [
        call["extra_body"].get("freechat", {}).get("agent_hints", {}).get("lifecycle")
        for call in delegate.calls
    ]
    if lifecycles[:2] != ["active", "resume"]:
        raise RuntimeError(f"expected active/resume model calls, observed {lifecycles}")
    if tool_paths != ["README.md"]:
        raise RuntimeError(f"expected one README.md tool call, observed {tool_paths}")
    final_output = str(result.final_output)
    return {
        "schema": 1,
        "task_id": arguments.task_id,
        "session_id": arguments.session_id,
        "gateway": arguments.gateway,
        "model": arguments.model,
        "elapsed_ms": (finished_ns - started_ns) / 1_000_000,
        "final_output": final_output,
        "expected_heading": expected_heading,
        "answer_matches_expected": final_output.strip() == expected_heading,
        "tool_paths": tool_paths,
        "terminal_lifecycle": lifecycle.current.lifecycle,
        "model_calls": delegate.calls,
    }


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Real OpenAI Agents SDK lifecycle run")
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--expected-resume-ms", type=int, default=500)
    parser.add_argument("--tool-output-characters", type=int, default=2_000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = await run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
