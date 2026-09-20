import argparse
import copy
import json
from typing import Any

import httpx
import pytest

from tools.validate_inference_loop import audit_log, concurrent_probe


def events() -> list[dict[str, Any]]:
    def event(kind: str, key: str, amount: int) -> dict[str, Any]:
        receipt = None
        if kind == "lease.released":
            receipt = {
                "command": {
                    "decision_id": key,
                    "request_id": "r-" + key,
                    "tenant_id": "tenant",
                    "worker_id": "gpu",
                    "worker_generation": 7,
                    "engine_instance_id": "engine",
                    "action": "abort" if key == "b" else "query",
                },
                "quiescent": True,
                "admission_closed": True,
                "status": "aborted" if key == "b" else "completed",
            }
        return {
            "event_id": kind + key,
            "event_type": kind,
            "aggregate_id": key,
            "aggregate_generation": 7,
            "tenant_id": "tenant",
            "payload": {
                "worker_id": "gpu",
                "request_id": "r-" + key,
                "reserved_kv_bytes_per_rank": amount,
                "execution_receipt": receipt,
                "decision": {"engine_instance_id": "engine"},
            },
        }

    return [
        event("route.decided", "a", 60),
        event("route.decided", "b", 30),
        event("lease.cancel_requested", "b", 30),
        event("lease.released", "a", 60),
        event("lease.released", "b", 30),
    ]


def audit(records: list[dict[str, Any]], pool: int = 100) -> dict[str, Any]:
    return audit_log(
        ["INFO lifecycle_event " + json.dumps(e) for e in records],
        pool=pool,
        worker="gpu",
        expected_routes=2,
        expected_cancels=1,
        expected_rejections=0,
    )


def test_log_audit_checks_peak_and_complete_release_without_files() -> None:
    report = audit(events())
    assert report["peak_reserved_bytes"] == 90
    assert report["routes"] == report["releases"] == 2
    assert report["unreleased"] == 0


def test_identical_event_redelivery_is_not_duplicate_accounting() -> None:
    records = events()
    records.insert(1, copy.deepcopy(records[0]))
    assert audit(records)["peak_reserved_bytes"] == 90


@pytest.mark.parametrize(
    "failure",
    [
        "oversubscription",
        "missing_release",
        "duplicate_release",
        "conflicting_event",
        "not_quiescent",
        "admission_open",
        "wrong_engine",
        "wrong_tenant",
        "wrong_generation",
        "wrong_request",
        "wrong_decision",
        "wrong_status",
        "wrong_action",
        "mixed_workers",
        "missing_cancel",
    ],
)
def test_log_audit_rejects_false_success(failure: str) -> None:
    records = events()
    pool = 100
    receipt = records[-1]["payload"]["execution_receipt"]
    if failure == "oversubscription":
        pool = 80
    elif failure == "missing_release":
        records.pop()
    elif failure == "duplicate_release":
        duplicate = copy.deepcopy(records[-1])
        duplicate["event_id"] = "other"
        records.append(duplicate)
    elif failure == "conflicting_event":
        duplicate = copy.deepcopy(records[0])
        duplicate["payload"]["reserved_kv_bytes_per_rank"] = 1
        records.insert(1, duplicate)
    elif failure == "not_quiescent":
        receipt["quiescent"] = False
    elif failure == "admission_open":
        receipt["admission_closed"] = False
    elif failure == "wrong_status":
        receipt["status"] = "running"
    elif failure == "wrong_action":
        receipt["command"]["action"] = "query"
    elif failure == "mixed_workers":
        records[1]["payload"]["worker_id"] = "other"
    elif failure == "missing_cancel":
        records.pop(2)
    else:
        field = {
            "wrong_engine": "engine_instance_id",
            "wrong_tenant": "tenant_id",
            "wrong_generation": "worker_generation",
            "wrong_request": "request_id",
            "wrong_decision": "decision_id",
        }[failure]
        receipt["command"][field] = "other"
    with pytest.raises(ValueError):
        audit(records, pool)


def test_no_events_cannot_pass() -> None:
    with pytest.raises(ValueError):
        audit([])


def test_only_exact_capacity_rejection_counts() -> None:
    records = ["lifecycle_event " + json.dumps(e) for e in events()]
    records.append(
        'admission_rejected {"request_id":"blocked","rejected":{"gpu":["vram_capacity"]}}'
    )
    result = audit_log(
        records,
        pool=100,
        worker="gpu",
        expected_routes=2,
        expected_cancels=1,
        expected_rejections=1,
    )
    assert result["capacity_rejections"] == 1
    records[-1] = records[-1].replace("vram_capacity", "worker_unhealthy")
    with pytest.raises(ValueError):
        audit_log(
            records,
            pool=100,
            worker="gpu",
            expected_routes=2,
            expected_cancels=1,
            expected_rejections=1,
        )


async def test_concurrent_probe_retries_and_counts_pending_release_backpressure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    long_calls, recovery_calls = 0, 0

    async def endpoint(request: httpx.Request) -> httpx.Response:
        nonlocal long_calls, recovery_calls
        body = json.loads(request.content)
        if body["max_tokens"] == 4:
            recovery_calls += 1
            accepted = recovery_calls == 2
        else:
            long_calls += 1
            accepted = long_calls in (9, 10)
        if not accepted:
            return httpx.Response(503, json={"detail": "not admitted"})
        return httpx.Response(
            200,
            headers={"x-freechat-reserved-kv-bytes": "1234"},
            json={"usage": {"completion_tokens": body["max_tokens"]}},
        )

    args = argparse.Namespace(model="qwen", url="http://gateway", concurrent_requests=8, rounds=1)
    runtime = {
        "worker_id": "gpu",
        "capacity": {
            "max_context_tokens": 1024,
            "block_size_tokens": 16,
            "block_bytes": 196608,
        },
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        await concurrent_probe(client, args, {}, runtime, 24969216)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["expected_routes"] == 4  # Readiness, two long calls, recovery.
    assert summary["expected_capacity_rejections"] == 15
    assert long_calls == 16 and recovery_calls == 2
