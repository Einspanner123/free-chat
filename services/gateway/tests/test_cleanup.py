import asyncio
from contextlib import suppress
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import anyio
import httpx
import pytest
from freechat_contracts import RequestProfile, RouteDecision
from freechat_gateway import app as gateway
from freechat_gateway.routing import SchedulerClient, StaticSchedulerClient


@pytest.mark.parametrize("uncertain", [False, True])
@pytest.mark.parametrize("close_mode", ["normal", "error", "timeout"])
async def test_cleanup_intent_survives_close_failure_and_cancel_scope(
    monkeypatch: pytest.MonkeyPatch, uncertain: bool, close_mode: str
) -> None:
    monkeypatch.setattr(gateway, "_CLEANUP_TIMEOUT_SECONDS", 0.02)
    profile = RequestProfile(
        tenant_id="tenant",
        model_id="model",
        input_tokens=1,
        output_tokens=1,
        hints={"harness_id": "test", "task_id": "task", "agent_id": "agent"},
    )
    decision = await StaticSchedulerClient("worker", "http://worker").route(profile)
    recorded: list[str] = []

    async def cancel(request: RequestProfile, route: RouteDecision) -> None:
        await anyio.sleep(0)
        recorded.append("cancel")

    async def release(request: RequestProfile, route: RouteDecision) -> None:
        await anyio.sleep(0)
        recorded.append("release")

    async def close() -> None:
        await anyio.sleep(0)
        if close_mode == "error":
            raise httpx.ReadError("close failed")
        if close_mode == "timeout":
            await anyio.sleep_forever()
        recorded.append("closed")

    scheduler = MagicMock()
    scheduler.cancel = AsyncMock(side_effect=cancel)
    scheduler.release = AsyncMock(side_effect=release)
    response = httpx.Response(200)
    monkeypatch.setattr(response, "aclose", close)
    keepalive = asyncio.create_task(asyncio.sleep(60))
    with anyio.CancelScope() as outer:
        outer.cancel()
        await gateway._finish_route(
            cast(SchedulerClient, scheduler),
            profile,
            decision,
            keepalive,
            response=response,
            uncertain=uncertain,
        )
    assert keepalive.done()
    assert recorded[-1] == ("cancel" if uncertain else "release")
    assert scheduler.cancel.await_count == int(uncertain)
    assert scheduler.release.await_count == int(not uncertain)
    assert ("closed" in recorded) == (close_mode == "normal")


@pytest.mark.parametrize("uncertain", [False, True])
@pytest.mark.parametrize("stage", ["close", "scheduler"])
async def test_cleanup_deadline_handles_transport_cancellation_without_scope_marker(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    uncertain: bool,
    stage: str,
) -> None:
    monkeypatch.setattr(gateway, "_CLEANUP_TIMEOUT_SECONDS", 0.02)
    profile = RequestProfile(
        tenant_id="tenant",
        model_id="model",
        input_tokens=1,
        output_tokens=1,
        hints={"harness_id": "test", "task_id": "task", "agent_id": "agent"},
    )
    decision = await StaticSchedulerClient("worker", "http://worker").route(profile)

    async def transport(*args: object) -> None:
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(60)
        # grpc.aio can produce a fresh cancellation without AnyIO's scope marker.
        raise asyncio.CancelledError()

    scheduler = MagicMock()
    scheduler.cancel = AsyncMock(side_effect=transport if stage == "scheduler" else None)
    scheduler.release = AsyncMock(side_effect=transport if stage == "scheduler" else None)
    response = httpx.Response(200)
    if stage == "close":
        monkeypatch.setattr(response, "aclose", transport)
    keepalive = asyncio.create_task(asyncio.sleep(60))
    with anyio.CancelScope() as outer:
        outer.cancel()
        await gateway._finish_route(
            cast(SchedulerClient, scheduler),
            profile,
            decision,
            keepalive,
            response=response,
            uncertain=uncertain,
        )
    assert keepalive.done()
    assert scheduler.cancel.await_count == int(uncertain)
    assert scheduler.release.await_count == int(not uncertain)
    assert ("upstream" if stage == "close" else "scheduler") + " cleanup timed out" in caplog.text
