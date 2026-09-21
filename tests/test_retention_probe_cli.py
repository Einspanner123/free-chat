"""Fixture HTTP/log inputs validate the probe, not engine retention or performance."""

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from benchmarks import probe_lifecycle_retention as probe


@pytest.mark.parametrize("status", [200, 500])
def test_probe_request_metadata_and_errors(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["agent_lifecycle"]["priority"] == 100
        assert body["agent_lifecycle"]["expected_resume_ms"] == 1000
        return httpx.Response(
            status, json={"id": "response", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        args: dict[str, Any] = dict(
            server_url="http://fixture/",
            model="model",
            prompt="hello",
            task_id="task",
            call_id="call",
            phase="tool_wait",
            expected_resume_ms=1000,
        )
        if status == 200:
            result = probe._request(client, **args)
            assert (result.response_id, result.prompt_tokens, result.completion_tokens) == (
                "response",
                10,
                1,
            )
            assert result.elapsed_ms >= 0
        else:
            with pytest.raises(RuntimeError, match="500"):
                probe._request(client, **args)


def test_event_parser_ignores_unrelated_records_and_counts_one_hit(tmp_path: Path) -> None:
    log = tmp_path / "input.log"
    log.write_text(
        "ordinary log\n"
        + json.dumps({"metadata": None})
        + "\n"
        + "INFO FREECHAT_CACHE_EVENT "
        + json.dumps(
            {"metadata": {"call_id": "target"}, "event_type": "hit", "block_ids": [[1, 2], [3]]}
        )
        + "\n"
        + json.dumps({"metadata": {"call_id": "other"}})
    )
    events = probe._events_for_call(log, "target")
    assert len(events) == 1
    assert probe._hit_tokens(events, 16) == 48
    assert probe._hit_tokens([], 16) == 0
    with pytest.raises(ValueError, match="more than one"):
        probe._hit_tokens(events * 2, 16)


@pytest.mark.parametrize("flag", ["--prefix-repetitions", "--pressure-requests", "--block-tokens"])
def test_invalid_probe_cli_parameters(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setattr(sys, "argv", ["probe", "--cache-events", "unused", flag, "0"])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2


@respx.mock
def test_probe_cli_keeps_mechanism_only_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    log = tmp_path / "input.log"
    log.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--server-url",
            "http://fixture",
            "--cache-events",
            str(log),
            "--pressure-requests",
            "2",
        ],
    )
    route = respx.post("http://fixture/v1/chat/completions").respond(
        200, json={"id": "fixture", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    )
    probe.main()
    result = json.loads(capsys.readouterr().out)
    assert route.call_count == 4
    assert [r["phase"] for r in result["requests"]] == ["tool_wait", "active", "active", "resume"]
    assert result["performance_claim_admissible"] is False
    assert result["evidence_level"] == "MECHANISM_ONLY"
    assert result["resume_hit_tokens"] == 0
