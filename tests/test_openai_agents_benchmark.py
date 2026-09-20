from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from freechat_contracts import Lifecycle

from benchmarks import openai_agents_e2e


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_cli_rejects_incomplete_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    accepted: bool,
) -> None:
    async def fake_run(arguments: Any) -> dict[str, Any]:
        assert arguments.max_turns == 4
        return {"accepted": accepted, "checks": {"answer_matches_expected": accepted}}

    monkeypatch.setattr(openai_agents_e2e, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--gateway",
            "http://127.0.0.1:8080",
            "--api-key",
            "local-test-key",
            "--model",
            "local-model",
            "--repository",
            ".",
            "--task-id",
            "task",
            "--session-id",
            "session",
        ],
    )
    if accepted:
        await openai_agents_e2e.async_main()
    else:
        with pytest.raises(SystemExit) as error:
            await openai_agents_e2e.async_main()
        assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["accepted"] is accepted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "option,value",
    [
        ("--timeout", "0"),
        ("--max-turns", "0"),
        ("--tool-output-characters", "0"),
        ("--tool-output-characters", "65537"),
    ],
)
async def test_cli_rejects_unbounded_or_empty_input(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--gateway",
            "http://127.0.0.1:8080",
            "--api-key",
            "local-test-key",
            "--model",
            "local-model",
            "--repository",
            ".",
            "--task-id",
            "task",
            "--session-id",
            "session",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        await openai_agents_e2e.async_main()
    assert error.value.code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_runner_failures_cancel_local_state_and_close_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timeout: bool,
) -> None:
    (tmp_path / "README.md").write_text("# Test project\n", encoding="utf-8")
    lifecycles: list[openai_agents_e2e.OpenAIAgentsLifecycle] = []
    base = openai_agents_e2e.OpenAIAgentsLifecycle

    class CapturingLifecycle(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            lifecycles.append(self)

    closed: list[bool] = []

    class Client:
        async def close(self) -> None:
            closed.append(True)

    async def fail_run(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["run_config"].tracing_disabled is True
        assert kwargs["run_config"].trace_include_sensitive_data is False
        assert kwargs["max_turns"] == 4
        if timeout:
            await asyncio.sleep(10)
        raise RuntimeError("injected runner failure")

    monkeypatch.setattr(openai_agents_e2e, "OpenAIAgentsLifecycle", CapturingLifecycle)
    monkeypatch.setattr(openai_agents_e2e, "AsyncOpenAI", lambda **kwargs: Client())
    monkeypatch.setattr(openai_agents_e2e.Runner, "run", fail_run)
    arguments = argparse.Namespace(
        api_key="local",
        gateway="http://127.0.0.1:8080",
        model="test",
        task_id="t",
        session_id="s",
        expected_resume_ms=500,
        repository=tmp_path,
        tool_output_characters=256,
        timeout=0.02 if timeout else 1,
        max_turns=4,
    )
    with pytest.raises(TimeoutError if timeout else RuntimeError):
        await openai_agents_e2e.run(arguments)
    assert closed == [True]
    assert lifecycles[0].current.lifecycle is Lifecycle.CANCELLED
