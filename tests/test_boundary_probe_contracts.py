"""Validate probe accounting and provenance; no model or GPU is exercised."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest

from benchmarks import analyze_offload_boundary as analysis
from benchmarks.harness_offload_boundary import NAMES, ProbeTransport


def records() -> dict[str, bytes]:
    result = {}
    index = []
    raw = b"fixture metrics"
    result["raw.prom"] = raw
    delta = {name: 0 for name in NAMES}
    delta.update(
        {
            "vllm:time_to_first_token_seconds_sum": 1,
            "vllm:request_prefill_kv_computed_tokens_sum": 10,
            "vllm:kv_offload_store_size_sum": 100,
            "vllm:kv_offload_load_size_sum": 40,
        }
    )
    for pair in range(2):
        for arm in [False, True]:
            name = f"trial-{pair}-{'candidate' if arm else 'baseline'}"
            index.append({"name": name, "warmup": False})
            result[name + "/result.json"] = json.dumps(
                {
                    "harness": "fixture",
                    "scenario": "pressure",
                    "preoffload": arm,
                    "artifacts": [{"name": "raw.prom", "sha256": hashlib.sha256(raw).hexdigest()}],
                    "adjusted_task_ms": 8 if arm else 10,
                    "raw_task_ms": 9 if arm else 11,
                    "calls": [{"deltas": delta}, {"deltas": delta}],
                    "correct": False,
                    "semantic_heading_present": True,
                }
            ).encode()
    result["index.json"] = json.dumps(index).encode()
    return result


def test_analysis_preserves_failures_and_cannot_promote_acceptance() -> None:
    report = analysis.analyze_records(records().__getitem__)
    group = report["groups"]["fixture/pressure"]
    assert group["strict_correct"] == {"baseline": 0, "preoffload": 0}
    assert group["terminal_unrestored_bytes"] == 240
    assert group["metrics"]["raw_task_ms"]["pairs"] == 2
    assert report["resume_performance_claim_admissible"] is False
    assert report["final_ab_acceptance"] is False


@pytest.mark.parametrize("fault", ["hash", "duplicate", "missing"])
def test_corrupt_or_unpaired_artifacts_are_rejected(fault: str) -> None:
    data = records()
    index = json.loads(data["index.json"])
    if fault == "hash":
        data["raw.prom"] = b"corrupted"
    elif fault == "duplicate":
        index.append(index[0])
    else:
        index.pop()
    data["index.json"] = json.dumps(index).encode()
    with pytest.raises(ValueError):
        analysis.analyze_records(data.__getitem__)


async def no_sleep(_: float) -> None:
    return None


@pytest.mark.parametrize("fault", ["none", "status", "reset", "preemption", "unpublished"])
async def test_probe_transport_validates_isolated_metrics_and_strips_untrusted_hints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    fault: str,
) -> None:
    probe = ProbeTransport("http://worker", "model", "fixture", enabled=True)
    requests: list[dict[str, Any]] = []
    metrics_calls = 0

    async def metrics(request: httpx.Request) -> httpx.Response:
        nonlocal metrics_calls
        metrics_calls += 1
        count = 0 if fault == "unpublished" else (metrics_calls - 1)
        if fault == "reset":
            count = max(0, 2 - metrics_calls)
        values = {name: 0 for name in NAMES}
        values["vllm:request_prompt_tokens_count"] = count
        values["vllm:num_preemptions_total"] = metrics_calls - 1 if fault == "preemption" else 0
        return httpx.Response(
            200,
            text="\n".join(
                f'{name}{{model_name="model",engine="0"}} {value}' for name, value in values.items()
            ),
        )

    async def infer(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(500 if fault == "status" else 200, json={"choices": []})

    await probe.transport.aclose()
    await probe.monitor.aclose()
    monkeypatch.setattr(probe, "transport", httpx.MockTransport(infer))
    probe.monitor = httpx.AsyncClient(
        base_url="http://worker", transport=httpx.MockTransport(metrics)
    )
    monkeypatch.setattr("benchmarks.harness_offload_boundary.asyncio.sleep", no_sleep)
    request = httpx.Request(
        "POST",
        "http://worker/v1/chat/completions",
        json={"model": "model", "freechat": {"tenant": "forged"}},
    )
    try:
        if fault == "none":
            await probe.handle_async_request(request)
            await probe.handle_async_request(request)
            assert len(probe.calls) == 2
            assert requests[1]["tool_choice"] == "none"
            assert len(probe.artifacts) == 8
        else:
            with pytest.raises(RuntimeError):
                await probe.handle_async_request(request)
            assert probe.calls == []
        assert "freechat" not in requests[0]
        assert requests[0]["kv_transfer_params"] == {"max_offload_tokens": 4096}
        assert requests[0]["cache_salt"] == probe.salt
    finally:
        await probe.aclose()
    capsys.readouterr()
