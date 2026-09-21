"""Test validator rejection logic against HTTP fixtures, never label this GPU evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from freechat.control.v1 import control_pb2_grpc
from freechat_contracts.execution import ExecutionCommand, ExecutionReceipt, ExecutionStatus

from tools import validate_inference_loop as loop
from tools import validate_native_http as native

CAPACITY = {
    "num_blocks": 4,
    "block_size_tokens": 16,
    "block_bytes": 256,
    "allocated_bytes": 1024,
    "max_context_tokens": 1024,
    "gpu_name": "fixture",
    "total_vram_bytes": 4096,
    "compute_capability": "8.6",
    "gpu_uuid": "fixture",
}
RUNTIME = {
    "worker_id": "worker",
    "generation": 1,
    "engine_instance_id": "engine",
    "capacity": CAPACITY,
}


def stream(protocol: str, *, text: bool = True) -> bytes:
    event: dict[str, Any]
    if protocol.endswith("responses"):
        event = {"type": "response.output_text.delta", "delta": "hi" if text else ""}
    elif protocol.endswith("messages"):
        event = {"delta": {"text": "hi" if text else ""}}
    else:
        event = {"choices": [{"delta": {"content": "hi" if text else ""}}]}
    return (": ping\n\ndata: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()


@pytest.mark.parametrize("fault", ["none", "usage", "no_text", "duplicate"])
@respx.mock
async def test_native_validator_requires_protocol_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, fault: str
) -> None:
    commands: list[ExecutionCommand] = []
    seen: set[str] = set()

    async def observe(request: Any, **options: Any) -> Any:
        command = ExecutionCommand.model_validate_json(request.command_json)
        commands.append(command)
        assert options["metadata"] == (("authorization", "Bearer " + "t" * 32),)
        receipt = ExecutionReceipt(
            command=command,
            observation_sequence=len(commands),
            observed_at=datetime.now(UTC),
            status=ExecutionStatus.COMPLETED,
            quiescent=True,
            admission_closed=True,
        )
        return SimpleNamespace(receipt_json=receipt.model_dump_json())

    monkeypatch.setattr(
        control_pb2_grpc,
        "RequestExecutionServiceStub",
        lambda _: SimpleNamespace(Observe=observe),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/freechat/runtime":
            return httpx.Response(200, json=RUNTIME)
        if path == "/freechat/prepare":
            return httpx.Response(
                200,
                json={
                    "preparation_id": "prepared",
                    "budget": {"input_tokens": 8, "output_tokens": 16},
                },
            )
        if "x-freechat-worker-token" not in request.headers:
            return httpx.Response(401)
        decision = request.headers["x-freechat-internal-decision-id"]
        assert request.headers["x-freechat-internal-reserved-kv-bytes"] == "512"
        if decision in seen:
            return httpx.Response(200 if fault == "duplicate" else 409, json={})
        seen.add(decision)
        body = json.loads(request.content)
        if body["stream"]:
            return httpx.Response(200, content=stream(path, text=fault != "no_text"))
        return httpx.Response(200, json={"usage": {"prompt_tokens": 9 if fault == "usage" else 8}})

    respx.route(host="worker").mock(side_effect=handler)
    if fault == "none":
        await native.validate("http://worker", "127.0.0.1:1", "model", "t" * 32)
        assert len(seen) == 9
        assert sum(c.action.value == "abort" for c in commands) == 3
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert len(rows) == 10
    else:
        with pytest.raises(AssertionError):
            await native.validate("http://worker", "127.0.0.1:1", "model", "t" * 32)
        capsys.readouterr()


async def test_native_probe_rejects_missing_auth_before_io() -> None:
    with pytest.raises(ValueError, match="TOKEN"):
        await native.validate("http://never-contact", "127.0.0.1:1", "model", "")


@pytest.mark.parametrize("fault", ["none", "no_usage", "no_text", "no_reservation"])
@respx.mock
async def test_managed_validator_cannot_succeed_on_incomplete_http_results(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, fault: str
) -> None:
    monkeypatch.setenv("FREECHAT_WORKER_TOKEN", "t" * 32)
    monkeypatch.setenv("FREECHAT_VALIDATION_API_KEY", "key")
    respx.get("http://worker/freechat/runtime").respond(200, json=RUNTIME)
    calls = 0

    def infer(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.headers.get("authorization") != "Bearer key":
            return httpx.Response(401)
        calls += 1
        headers = {
            "x-freechat-reserved-kv-bytes": "0" if fault == "no_reservation" else "512",
            "x-freechat-decision-id": str(calls),
            "x-freechat-worker-id": "worker",
        }
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200, headers=headers, content=stream(request.url.path, text=fault != "no_text")
            )
        return httpx.Response(
            200,
            headers=headers,
            json={
                "usage": {} if fault == "no_usage" else {"prompt_tokens": 8, "completion_tokens": 1}
            },
        )

    respx.route(host="gateway").mock(side_effect=infer)
    args = argparse.Namespace(
        url="http://gateway", worker_url="http://worker", model="model", mode="sequential"
    )
    if fault == "none":
        await loop.validate(args)
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert len(rows) == 10
        assert rows[-1]["exceeds_one_pool"] is True
        assert rows[-1]["log_audit_required"] is True
        assert rows[-1]["completed_calls"] == 2
        assert calls == 12
    else:
        with pytest.raises(AssertionError):
            await loop.validate(args)
        capsys.readouterr()
