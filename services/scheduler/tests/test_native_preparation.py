from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest
from freechat_contracts.preparation import PreparedAdmission, TokenBudget, body_digest
from freechat_control_store import InMemoryStore
from freechat_scheduler import NoEligibleWorker, Scheduler
from freechat_scheduler.preparation import NativePreparer
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_ledger import RequestLedger
from test_scheduler import MODEL, add_worker, profile


def request(**changes: Any) -> Any:
    body = {"model": MODEL.model_id, "messages": [{"role": "user", "content": "私密 prompt"}]}
    return profile(
        input_tokens=0,
        output_tokens=1,
        native_protocol="/v1/chat/completions",
        native_request_json=json.dumps(body),
        **changes,
    )


def prepared(worker: str, req: Any, tokens: int = 17, **changes: Any) -> PreparedAdmission:
    return PreparedAdmission(
        preparation_id="a" * 64,
        worker_id=worker,
        generation=1,
        engine_instance_id="fixture-engine",
        budget=TokenBudget(
            tenant_id=req.tenant_id,
            protocol=req.native_protocol,
            body_sha256=req.native_body_sha256,
            prompt_sha256="b" * 64,
            input_tokens=tokens,
            output_tokens=16,
            expires_at=time.time() + 60,
            **changes,
        ),
    )


def test_transient_prompt_not_in_ledger_fingerprint_or_repr() -> None:
    req = request()
    assert "native_request_json" not in req.model_dump()
    assert "私密" not in repr(req)
    assert "私密" not in req.model_dump_json()
    changed = json.loads(req.native_request_json)
    changed["messages"][0]["content"] = "changed"
    assert body_digest(changed) != req.native_body_sha256


def test_prepared_route_uses_selected_worker_geometry_and_reservations() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    add_worker(registry, "b", node="ross")
    req = request()
    scheduler = Scheduler(registry)
    budgets = {"a": prepared("a", req, 40_000), "b": prepared("b", req, 17)}
    result = scheduler.route(req, prepared=budgets)
    assert result.worker_id == "b"
    assert result.reserved_kv_bytes_per_rank == 48 * 16384
    assert result.preparation == budgets["b"]
    assert "context_capacity" in result.rejected["a"]
    assert not result.kv_transfer.applicable
    with pytest.raises(NoEligibleWorker):
        scheduler.route(req, prepared=budgets, reserved={"b": (20 * 1024**3, 1)})


@pytest.mark.parametrize(
    "mutation", ["expired", "generation", "engine", "tenant", "body", "missing"]
)
def test_cas_replan_rejects_stale_preparation(mutation: str) -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    req = request()
    item = prepared("a", req)
    if mutation in {"generation", "engine"}:
        field = "generation" if mutation == "generation" else "engine_instance_id"
        item = item.model_copy(update={field: 2 if field == "generation" else "replacement"})
    else:
        variants: dict[str, dict[str, Any]] = {
            "expired": {"expires_at": time.time() - 1},
            "tenant": {"tenant_id": "foreign"},
            "body": {"body_sha256": "c" * 64},
        }
        changes = variants.get(mutation, {})
        item = item.model_copy(update={"budget": item.budget.model_copy(update=changes)})
    with pytest.raises(NoEligibleWorker):
        Scheduler(registry).route(req, prepared={} if mutation == "missing" else {"a": item})


async def test_preparation_filters_privacy_before_sending_prompt_and_disables_redirects() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    add_worker(registry, "b", node="remote")
    req = request(hints=profile().hints.model_copy(update={"allow_remote_worker": False}))
    seen: list[str] = []

    async def handler(message: httpx.Request) -> httpx.Response:
        seen.append(message.url.host)
        assert message.headers["x-freechat-internal-tenant"] == req.tenant_id
        return httpx.Response(200, json=prepared("a", req).model_dump())

    scheduler = Scheduler(registry)
    result = await NativePreparer("t" * 32, transport=httpx.MockTransport(handler)).prepare(
        req, scheduler.preparation_candidates(req)
    )
    assert set(result) == {"a"} and seen == ["a"]


@pytest.mark.parametrize("failure", ["redirect", "unavailable", "malformed", "wrong_incarnation"])
async def test_no_estimate_fallback_when_native_preparation_fails(failure: str) -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    req = request()

    async def handler(message: httpx.Request) -> httpx.Response:
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "http://untrusted/prepare"})
        if failure == "unavailable":
            raise httpx.ConnectError("offline", request=message)
        if failure == "malformed":
            return httpx.Response(200, json={})
        return httpx.Response(
            200, json=prepared("a", req).model_copy(update={"generation": 2}).model_dump()
        )

    scheduler = Scheduler(registry)
    result = await NativePreparer("t" * 32, transport=httpx.MockTransport(handler)).prepare(
        req, scheduler.preparation_candidates(req)
    )
    assert not result
    with pytest.raises(NoEligibleWorker):
        scheduler.route(req, prepared=result)


async def test_ledger_persists_measured_reservation_not_private_body() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    req = request()
    scheduler = Scheduler(registry)
    ledger = RequestLedger(InMemoryStore())
    item = prepared("a", req)
    decision = await ledger.reserve(
        req,
        "idempotent",
        lambda reserved: scheduler.route(req, prepared={"a": item}, reserved=reserved),
    )
    snapshot = await ledger.snapshot()
    assert snapshot.reservations[decision.decision_id].decision.preparation == item
    assert "私密" not in snapshot.model_dump_json()
    changed = req.model_copy(update={"native_body_sha256": "c" * 64})
    with pytest.raises(ValueError):
        await ledger.reserve(changed, "idempotent", lambda reserved: decision)


def test_native_profile_cannot_use_unprepared_counts() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    with pytest.raises(ValueError, match="native_preparation_required"):
        Scheduler(registry).route(request())


async def test_default_scheduler_requires_worker_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from freechat_scheduler.grpc_server import serve

    monkeypatch.delenv("FREECHAT_WORKER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="worker token"):
        await serve("127.0.0.1:0")


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [("healthy", False, "worker_unhealthy"), ("draining", True, "worker_draining")],
)
@pytest.mark.parametrize("prepared_before_change", [False, True])
def test_excluded_worker_retains_health_reason_after_preparation(
    field: str,
    value: bool,
    reason: str,
    prepared_before_change: bool,
) -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    add_worker(registry, "b", node="ross")
    req = request()
    budgets = {"a": prepared("a", req)}
    if prepared_before_change:
        budgets["b"] = prepared("b", req)
    worker = next(item for item in registry.snapshot()[1] if item.capabilities.worker_id == "b")
    registry.upsert(
        worker.capabilities,
        worker.telemetry.model_copy(update={field: value}),
    )
    scheduler = Scheduler(registry)
    assert [item.capabilities.worker_id for item in scheduler.preparation_candidates(req)] == ["a"]
    decision = scheduler.route(req, prepared=budgets)
    assert decision.worker_id == "a"
    assert reason in decision.rejected["b"]
    assert "native_preparation_missing_expired_or_mismatched" not in decision.rejected["b"]
