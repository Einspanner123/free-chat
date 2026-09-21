"""HTTP boundary checks using a Scheduler fixture, not GPU acceptance."""

from datetime import UTC, datetime
from typing import Any

import grpc
import pytest
from fastapi.testclient import TestClient
from freechat_contracts.cache_lifecycle import (
    MAX_CACHE_CONTROL_BYTES,
    CacheLifecycleCommand,
    CacheLifecycleReceipt,
    CacheLifecycleUpdate,
    PrefixLifecycleReceipt,
)
from freechat_contracts.execution import ExecutionCommand
from freechat_gateway import GatewayConfig, create_app
from freechat_gateway.routing import StaticSchedulerClient


class CacheScheduler(StaticSchedulerClient):
    calls: list[tuple[str, CacheLifecycleUpdate]]
    failure: Exception | None = None

    async def cache_lifecycle(
        self, tenant_id: str, update: CacheLifecycleUpdate
    ) -> CacheLifecycleReceipt:
        self.calls.append((tenant_id, update))
        if self.failure is not None:
            raise self.failure
        return CacheLifecycleReceipt(
            command=CacheLifecycleCommand(
                owner=ExecutionCommand(
                    tenant_id=tenant_id,
                    request_id="request",
                    decision_id=update.decision_id,
                    worker_id="worker",
                    worker_generation=1,
                    engine_instance_id="engine",
                    action="query",
                ),
                update=update,
                cache_generation=1,
            ),
            prefixes=(
                PrefixLifecycleReceipt(
                    request_id="internal",
                    sequence=update.sequence,
                    lifecycle=update.lifecycle,
                    resident_blocks=0,
                    protected_blocks=0,
                    status="not_resident",
                    applied_at_ms=100,
                    replayed=False,
                ),
            ),
            observed_at=datetime.now(UTC),
        )


@pytest.fixture
def runtime() -> Any:
    scheduler = CacheScheduler("worker", "http://worker:8000")
    # StaticSchedulerClient is frozen; fixture state is installed explicitly.
    object.__setattr__(scheduler, "calls", [])
    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        scheduler=scheduler,
    )
    with TestClient(app) as client:
        yield client, scheduler


BODY = {
    "decision_id": "decision",
    "sequence": 1,
    "lifecycle": "tool_wait",
    "expected_resume_ms": 1000,
}
HEADERS = {"authorization": "Bearer secret-key"}


@pytest.mark.parametrize("headers", [HEADERS, {"x-api-key": "secret-key"}])
def test_tenant_comes_from_authentication(runtime: Any, headers: dict[str, str]) -> None:
    client, scheduler = runtime
    response = client.post("/freechat/cache/lifecycle", json=BODY, headers=headers)
    assert response.status_code == 200
    assert scheduler.calls == [("tenant-a", CacheLifecycleUpdate.model_validate(BODY))]
    assert response.json()["command"]["owner"]["tenant_id"] == "tenant-a"


def test_authentication_precedes_body_validation(runtime: Any) -> None:
    client, scheduler = runtime
    response = client.post("/freechat/cache/lifecycle", content=b"not json")
    assert response.status_code == 401
    assert scheduler.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        {"tenant_id": "other"},
        {"worker_id": "other"},
        {"block_ids": [1]},
        {"sequence": True},
        {"expected_resume_ms": None},
    ],
)
def test_forged_or_invalid_intent_never_reaches_scheduler(
    runtime: Any, extra: dict[str, Any]
) -> None:
    client, scheduler = runtime
    response = client.post("/freechat/cache/lifecycle", json={**BODY, **extra}, headers=HEADERS)
    assert response.status_code == 422
    assert scheduler.calls == []


def test_body_limit(runtime: Any) -> None:
    client, scheduler = runtime
    response = client.post(
        "/freechat/cache/lifecycle", content=b"x" * (MAX_CACHE_CONTROL_BYTES + 1), headers=HEADERS
    )
    assert response.status_code == 413
    assert scheduler.calls == []


@pytest.mark.parametrize(
    "code,status",
    [
        (grpc.StatusCode.NOT_FOUND, 404),
        (grpc.StatusCode.FAILED_PRECONDITION, 409),
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
        (grpc.StatusCode.INTERNAL, 502),
    ],
)
def test_transport_errors_have_stable_public_status(runtime: Any, code: Any, status: int) -> None:
    client, scheduler = runtime
    failure = grpc.aio.AioRpcError(code, None, None, "private engine details")
    object.__setattr__(scheduler, "failure", failure)
    response = client.post("/freechat/cache/lifecycle", json=BODY, headers=HEADERS)
    assert response.status_code == status
    assert "private engine details" not in response.text


@pytest.mark.parametrize(
    "failure,status",
    [
        (NotImplementedError("no owner"), 503),
        (ValueError("mismatched receipt"), 502),
    ],
)
def test_unconfigured_or_invalid_response_is_not_success(
    runtime: Any, failure: Exception, status: int
) -> None:
    client, scheduler = runtime
    object.__setattr__(scheduler, "failure", failure)
    assert (
        client.post("/freechat/cache/lifecycle", json=BODY, headers=HEADERS).status_code == status
    )
