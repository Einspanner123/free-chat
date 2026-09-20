from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from agents import Agent, Model, ModelSettings, RunConfig, Runner, function_tool
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
    readme: Path = arguments.repository / "README.md"
    if readme.is_symlink():
        raise ValueError("benchmark README must not be a symlink")
    with readme.open(encoding="utf-8") as source:
        tool_contents = source.read(arguments.tool_output_characters)
    expected_heading = next(
        line.strip() for line in tool_contents.splitlines() if line.startswith("#")
    )

    @function_tool
    def read_file(path: str) -> str:
        """Read README.md, the only file exposed by this benchmark."""
        if path != "README.md":
            raise ValueError("only README.md is available")
        tool_paths.append(path)
        return tool_contents

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
        async with asyncio.timeout(arguments.timeout):
            result = await Runner.run(
                agent,
                "Inspect README.md.",
                hooks=lifecycle.hooks(),
                max_turns=arguments.max_turns,
                run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False),
            )
    except BaseException:
        lifecycle.cancel()
        raise
    finally:
        await client.close()
    finished_ns = time.perf_counter_ns()
    lifecycles = [
        call["extra_body"].get("freechat", {}).get("agent_hints", {}).get("lifecycle")
        for call in delegate.calls
    ]
    final_output = str(result.final_output)
    checks = {
        "active_resume": lifecycles == ["active", "resume"],
        "one_readme_tool": tool_paths == ["README.md"],
        "answer_matches_expected": final_output.strip() == expected_heading,
        "terminal": lifecycle.current.lifecycle == "terminal",
    }
    return {
        "schema": 1,
        "accepted": all(checks.values()),
        "checks": checks,
        "scope": "real-sdk-model-requests; no standalone tool-wait control event",
        "sdk_cloud_tracing": False,
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
    parser.add_argument("--max-turns", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.timeout <= 0 or arguments.max_turns <= 0:
        parser.error("timeout and max-turns must be positive")
    if not 1 <= arguments.tool_output_characters <= 65_536:
        parser.error("tool-output-characters must be between 1 and 65536")
    payload = await run(arguments)
    print(json.dumps(payload, indent=2, default=str))
    if not payload["accepted"]:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
