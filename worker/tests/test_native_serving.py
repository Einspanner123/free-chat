from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionStatus
from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.native_serving import (
    CURRENT_ROUTE,
    AdmissionMiddleware,
    NativeEngineClient,
    NativeExecutionBackend,
)


class Engine:
    def __init__(self) -> None:
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            kv_transfer_config=None,
            ec_transfer_config=None,
        )
        self.engine_core = self
        self.quiet = True
        self.submissions: list[str] = []
        self.aborts: list[str] = []

    async def add_request(self, request_id: str, prompt: Any, params: Any, **kwargs: Any) -> Any:
        self.submissions.append(request_id)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        queue.put_nowait(SimpleNamespace(finished=True, request_id=request_id))
        return SimpleNamespace(request_id=request_id + "-internal", get=queue.get)

    async def abort(self, request_id: str, internal: bool) -> None:
        assert internal
        self.aborts.append(request_id)

    async def call_utility_async(self, method: str, request_id: str) -> dict[str, Any]:
        return {"request_id": request_id, "quiescent": self.quiet}


def command(**changes: Any) -> ExecutionCommand:
    return ExecutionCommand.model_validate(
        {
            "tenant_id": "tenant",
            "request_id": "req",
            "decision_id": "decision",
            "worker_id": "worker",
            "worker_generation": 1,
            "engine_instance_id": "engine",
            "action": "query",
            **changes,
        }
    )


def headers(**changes: str) -> dict[str, str]:
    return {
        "x-freechat-worker-token": "t" * 32,
        "x-freechat-internal-tenant": "tenant",
        "x-freechat-internal-request-id": "req",
        "x-freechat-internal-decision-id": "decision",
        "x-freechat-internal-worker-generation": "1",
        "x-freechat-internal-engine-instance-id": "engine",
        **changes,
    }


@pytest.fixture
def runtime(tmp_path: Path) -> Any:
    engine = Engine()
    backend = NativeExecutionBackend(engine)
    proxy = NativeEngineClient(engine, backend)
    driver = DurableExecutionDriver(
        tmp_path / "runtime.db",
        backend,
        worker_id="worker",
        generation=1,
        engine_instance_id="engine",
        create=True,
    )
    app = FastAPI()
    app.state.native_calls = 2

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict[str, Any]:
        await request.json()
        for index in range(app.state.native_calls):
            async for output in proxy.generate(
                {"type": "token", "prompt_token_ids": [1], "cache_salt": "tenant-salt"},
                SimpleNamespace(n=1, max_tokens=4),
                f"native-{index}",
                priority=0,
            ):
                assert output.request_id == f"native-{index}"
        return {"ok": True}

    wrapped = AdmissionMiddleware(app, driver=driver, backend=backend, token="t" * 32)
    yield engine, backend, driver, wrapped
    driver.close()


async def test_native_handler_multiple_engine_calls_have_one_release_boundary(runtime: Any) -> None:
    engine, backend, driver, app = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://worker"
    ) as client:
        response = await client.post("/v1/chat/completions", json={}, headers=headers())
        assert response.status_code == 200
        assert len(engine.submissions) == 2
        assert len(backend.routes) == 1
        assert (await driver.observe(command())).releasable
        duplicate = await client.post("/v1/chat/completions", json={}, headers=headers())
        assert duplicate.status_code == 409
        assert len(engine.submissions) == 2


@pytest.mark.parametrize(
    "mode", ["token", "generation", "duplicate_identity", "bypass", "background"]
)
async def test_native_ingress_rejects_unauthorized_or_untracked_execution(
    runtime: Any,
    mode: str,
) -> None:
    engine, _, _, app = runtime
    supplied: Any = headers()
    path, body = "/v1/chat/completions", {}
    expected = 409
    if mode == "token":
        supplied["x-freechat-worker-token"] = "wrong"
        expected = 401
    elif mode == "generation":
        supplied["x-freechat-internal-worker-generation"] = "2"
    elif mode == "duplicate_identity":
        supplied = [*supplied.items(), ("x-freechat-internal-tenant", "other")]
    elif mode == "bypass":
        path, expected = "/v1/completions", 404
    else:
        body = {"background": True}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://worker"
    ) as client:
        response = await client.post(path, json=body, headers=supplied)
    assert response.status_code == expected
    assert not engine.submissions


async def test_abort_before_http_admission_rejects_late_arrival(runtime: Any) -> None:
    engine, _, driver, app = runtime
    receipt = await driver.observe(command(action=ExecutionAction.ABORT))
    assert receipt.releasable and receipt.status == ExecutionStatus.NOT_ACCEPTED
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://worker"
    ) as client:
        response = await client.post("/v1/chat/completions", json={}, headers=headers())
    assert response.status_code == 409 and not engine.submissions


async def test_http_completion_cannot_replace_engine_quiescence(runtime: Any) -> None:
    engine, _, driver, app = runtime
    engine.quiet = False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://worker"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={}, headers=headers())
        ).status_code == 200
    assert not (await driver.observe(command())).releasable
    engine.quiet = True
    assert (await driver.observe(command())).releasable


async def test_closed_route_cannot_spawn_late_native_engine_call(runtime: Any) -> None:
    engine, backend, driver, _ = runtime
    key = await driver.admit(command(), {})
    backend.close_submission(key)
    token = CURRENT_ROUTE.set(key)
    try:
        with pytest.raises(ValueError, match="submission_closed"):
            await anext(backend.generate("prompt", SimpleNamespace(n=1), "late"))
    finally:
        CURRENT_ROUTE.reset(token)
    assert not engine.submissions
    assert (await driver.observe(command())).releasable


async def test_native_generation_without_http_admission_is_rejected(runtime: Any) -> None:
    _, backend, _, _ = runtime
    with pytest.raises(ValueError, match="without_admission"):
        await anext(backend.generate("prompt", SimpleNamespace(n=1), "bypass"))


@pytest.mark.parametrize("rendered", [[1], [2]])
async def test_preparation_is_required_and_actual_prompt_is_checked_before_gpu(
    runtime: Any, rendered: list[int],
) -> None:
    from freechat_worker.preparation import PreparationService

    engine, _, driver, app = runtime

    async def render(path: str, body: dict[str, Any]) -> tuple[list[int], int]:
        return rendered, 4

    app.preparer = PreparationService(render)
    app.app.state.native_calls = 1
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://worker"
    ) as client:
        denied = await client.post("/v1/chat/completions", json={}, headers=headers())
        assert denied.status_code == 409 and not engine.submissions
        prepared = await client.post(
            "/freechat/prepare", headers=headers(),
            json={"protocol": "/v1/chat/completions", "request": {}},
        )
        assert prepared.status_code == 200
        supplied = headers(
            **{"x-freechat-internal-preparation": prepared.json()["preparation_id"]}
        )
        changed = await client.post(
            "/v1/chat/completions", json={"max_tokens": 9}, headers=supplied,
        )
        assert changed.status_code == 409 and not engine.submissions
        if rendered == [1]:
            result = await client.post("/v1/chat/completions", json={}, headers=supplied)
            assert result.status_code == 200 and len(engine.submissions) == 1
        else:
            with pytest.raises(ValueError, match="prepared_budget"):
                await client.post("/v1/chat/completions", json={}, headers=supplied)
            assert not engine.submissions
        assert (await driver.observe(command())).releasable
