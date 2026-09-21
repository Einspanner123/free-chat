"""Probe orchestration tests; simulated devices are not hardware evidence."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import urllib.request
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from test_validate_worker_image import valid_inspection, valid_probe

from benchmarks import serving_baseline as baseline
from tools import validate_worker_image as image


@pytest.mark.parametrize("payload", ["{}", "[]", "[1]", "[{},{}]"])
def test_image_inspection_rejects_ambiguous_records(payload: str) -> None:
    with pytest.raises(RuntimeError):
        image.inspect_image("image", lambda _: image.CommandResult(payload, "", 0))


@pytest.mark.parametrize(
    "output,error,code", [("bad", "", 0), ("", "denied", 1), ("failed", "", 1)]
)
def test_image_probe_failures_are_not_accepted(output: str, error: str, code: int) -> None:
    with pytest.raises(RuntimeError):
        image.inspect_image("image", lambda _: image.CommandResult(output, error, code))


@pytest.mark.parametrize("gpu", [None, "1"])
def test_image_probe_passes_device_and_validates_shape(gpu: str | None) -> None:
    calls: list[Sequence[str]] = []

    def run(command: Sequence[str]) -> image.CommandResult:
        calls.append(command)
        return image.CommandResult("{}", "", 0)

    assert image.probe_image("pinned-image", gpu, run) == {}
    assert ("--gpus" in calls[0]) == (gpu is not None)
    assert "pinned-image" in calls[0]
    with pytest.raises(RuntimeError, match="invalid record"):
        image.probe_image("image", gpu, lambda _: image.CommandResult("[]", "", 0))


def test_command_runner_retains_failure_details() -> None:
    result = image.run_command(
        [sys.executable, "-c", "import sys;print('out');print('err',file=sys.stderr);sys.exit(7)"]
    )
    assert result.returncode == 7 and result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


@pytest.mark.parametrize("mode,exit_code", [("valid", None), ("invalid", 1), ("error", 2)])
def test_image_cli_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, mode: str, exit_code: int | None
) -> None:
    monkeypatch.setattr(sys, "argv", ["validate", "image", "--gpu", "0"])

    def run(command: Sequence[str]) -> image.CommandResult:
        if mode == "error":
            return image.CommandResult("", "not available", 1)
        payload: Any = [valid_inspection()] if command[1] == "image" else valid_probe()
        if mode == "invalid" and isinstance(payload, dict):
            payload["cuda_available"] = False
        return image.CommandResult(json.dumps(payload), "", 0)

    monkeypatch.setattr(image, "run_command", run)
    if exit_code is None:
        image.main()
    else:
        with pytest.raises(SystemExit) as error:
            image.main()
        assert error.value.code == exit_code
    out, err = capsys.readouterr()
    assert json.loads(err if mode == "error" else out)["accepted"] == (mode == "valid")


@pytest.mark.parametrize("strategy,count", [("unknown", 1), ("direct-vllm", 2), ("least-load", 0)])
def test_baseline_invalid_routes(strategy: str, count: int) -> None:
    with pytest.raises(ValueError):
        baseline.select_endpoint(strategy, [baseline.Endpoint("a", "http://a", "0")] * count, 0)


def test_missing_counter_rejected() -> None:
    with pytest.raises(ValueError):
        baseline.prometheus_counter("# no counter", "missing")


@pytest.mark.parametrize("content", [True, False])
def test_stream_parser_usage_and_first_token(
    monkeypatch: pytest.MonkeyPatch, content: bool
) -> None:
    events = [
        b": heartbeat\n",
        b"data: \n",
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n',
    ]
    if content:
        events += [
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"!"}}]}\n',
        ]
    events.append(b"data: [DONE]\n")
    seen: list[Any] = []

    def open_url(request: Any, **options: Any) -> io.BytesIO:
        seen.append(request)
        assert options["timeout"] == 1
        return io.BytesIO(b"".join(events))

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    result = baseline.stream_chat(
        baseline.Endpoint("a", "http://a/", "0"),
        model="m",
        messages=[],
        max_tokens=2,
        timeout_seconds=1,
    )
    assert result.content == ("hi!" if content else "")
    assert (result.input_tokens, result.output_tokens) == (10, 2)
    assert result.started_ms <= result.first_token_ms <= result.finished_ms
    if not content:
        assert result.first_token_ms == result.finished_ms
    assert json.loads(seen[0].data)["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("returncode", [0, 1])
def test_get_text_and_sampler_command(monkeypatch: pytest.MonkeyPatch, returncode: int) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"ok"))
    assert baseline.get_text("http://worker/health", 1) == "ok"
    sampler = baseline.GpuSampler("run", "1", 0.01)

    def run(command: list[str], **options: Any) -> Any:
        assert command[1] == "--id=1"
        sampler._stop.set()
        return SimpleNamespace(returncode=returncode, stdout="GPU-fixture, 42")

    monkeypatch.setattr(subprocess, "run", run)
    with sampler:
        sampler._thread.join(timeout=2)
    assert len(sampler.samples) == (1 if returncode == 0 else 0)
    if sampler.samples:
        assert sampler.samples[0]["utilization_percent"] == 42


def baseline_args() -> argparse.Namespace:
    return argparse.Namespace(
        endpoint=["a,http://a,0"],
        concurrency=1,
        timeout=1,
        prefix_repetitions=1,
        run_id="run",
        gpu_index="0",
        sample_interval=0.01,
        tasks=1,
        strategy="direct-vllm",
        model="model",
        max_tokens=2,
        trial_id=1,
        model_revision="revision",
    )


@pytest.mark.parametrize("mode", ["ok", "no_samples", "infer_error", "concurrent"])
def test_baseline_orchestration_checks_evidence(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    args = baseline_args()
    if mode == "concurrent":
        args.concurrency = 2
    monkeypatch.setattr(baseline, "get_text", lambda *a: "vllm:prefix_cache_hits_total 0")

    class Sampler:
        samples = [] if mode == "no_samples" else [{"utilization_percent": 42}]

        def __init__(self, *a: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    endpoints = []

    def stream(endpoint: Any, **kwargs: Any) -> Any:
        endpoints.append(endpoint)
        assert endpoint.active_requests == 1
        if mode == "infer_error":
            raise RuntimeError("inference failed")
        return baseline.StreamResult(0, 1, 2, 10, 2, "action")

    monkeypatch.setattr(baseline, "GpuSampler", Sampler)
    monkeypatch.setattr(baseline, "stream_chat", stream)
    if mode == "ok":
        result = baseline.run(args)
        assert [r["phase"] for r in result["requests"]] == ["active", "resume"]
        assert result["run"]["endpoints"][0]["active_requests"] == 0
    else:
        with pytest.raises((RuntimeError, ValueError)):
            baseline.run(args)
    assert all(endpoint.active_requests == 0 for endpoint in endpoints)


def test_stream_done_stops_before_trailing_data(monkeypatch: pytest.MonkeyPatch) -> None:
    response = io.BytesIO(
        b'data: {"choices":[{"delta":{"content":"accepted"}}]}\n'
        b"data: [DONE]\n"
        b'data: {"choices":[{"delta":{"content":"must not be consumed"}}]}\n'
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: response)
    result = baseline.stream_chat(
        baseline.Endpoint("a", "http://a", "0"),
        model="m",
        messages=[],
        max_tokens=2,
        timeout_seconds=1,
    )
    assert result.content == "accepted"
