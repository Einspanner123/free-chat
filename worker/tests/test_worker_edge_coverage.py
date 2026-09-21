"""CPU failure/adapter contracts; native model execution still requires GPU tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import grpc
import httpx
import pytest
import uvicorn
from freechat.control.v1 import control_pb2_grpc
from freechat_worker import preparation, serve, telemetry
from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.preparation import NativeRenderer
from test_telemetry import collector, payload
from test_vllm_execution import CacheEngine


@pytest.fixture
def renderer(monkeypatch: pytest.MonkeyPatch) -> tuple[NativeRenderer, Any]:
    request = SimpleNamespace(
        tools=[],
        previous_response_id=None,
        max_output_tokens=7,
        truncation="disabled",
        max_completion_tokens=None,
        max_tokens=9,
        truncate_prompt_tokens=None,
    )
    service = SimpleNamespace(
        _check_model=AsyncMock(return_value=None),
        _validate_create_responses_input=MagicMock(return_value=None),
        response_store_lock=asyncio.Lock(),
        response_store={},
        use_harmony=False,
        _make_request=AsyncMock(return_value=(None, [{"prompt_token_ids": [1, 2]}])),
        _make_request_with_harmony=MagicMock(return_value=(None, [{"prompt_token_ids": [1, 2]}])),
        render_chat_request=AsyncMock(return_value=(None, [{"prompt_token_ids": [1, 2]}])),
        _merge_inline_system=True,
        _convert_anthropic_to_openai_request=MagicMock(return_value=request),
        model_config=SimpleNamespace(max_model_len=64),
        default_sampling_params={"temperature": 0},
        override_max_tokens=None,
    )
    parser = SimpleNamespace(model_validate=MagicMock(return_value=request))
    for name, cls in [
        ("vllm.entrypoints.openai.responses.protocol", "ResponsesRequest"),
        ("vllm.entrypoints.openai.chat_completion.protocol", "ChatCompletionRequest"),
        ("vllm.entrypoints.anthropic.protocol", "AnthropicMessagesRequest"),
    ]:
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**{cls: parser}))
    maximum = MagicMock(return_value=11)
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.serve.utils.api_utils",
        SimpleNamespace(get_max_tokens=maximum),
    )
    state = SimpleNamespace(
        openai_serving_responses=service,
        openai_serving_chat=service,
        anthropic_serving_messages=service,
    )
    return NativeRenderer(state), SimpleNamespace(
        request=request, service=service, maximum=maximum, parser=parser
    )


@pytest.mark.parametrize("protocol", ["/v1/responses", "/v1/chat/completions", "/v1/messages"])
async def test_renderer_delegates_protocol_and_budget(renderer: Any, protocol: str) -> None:
    adapter, state = renderer
    body = {"model": "local"}
    assert await adapter(protocol, body) == ([1, 2], 11)
    state.parser.model_validate.assert_called_once_with(body)
    state.maximum.assert_called_once_with(
        64,
        7 if protocol == "/v1/responses" else 9,
        2,
        {"temperature": 0},
        None,
        truncate_prompt_tokens=None,
    )
    if protocol == "/v1/messages":
        state.service._convert_anthropic_to_openai_request.assert_called_once_with(
            state.request,
            merge_inline_system=True,
        )


@pytest.mark.parametrize("change", [{"background": True}, {"n": 2}, {"use_beam_search": True}])
async def test_renderer_rejects_unadmitted_inference_modes(renderer: Any, change: Any) -> None:
    adapter, state = renderer
    with pytest.raises(ValueError, match="foreground_single_sample"):
        await adapter("/v1/chat/completions", change)
    state.parser.model_validate.assert_not_called()
    state.maximum.assert_not_called()


@pytest.mark.parametrize("harmony", [False, True])
async def test_responses_previous_identity_and_truncation(renderer: Any, harmony: bool) -> None:
    adapter, state = renderer
    previous = object()
    state.request.previous_response_id = "previous"
    state.request.truncation = "auto"
    state.request.tools = [SimpleNamespace(type="function")]
    state.service.response_store["previous"] = previous
    state.service.use_harmony = harmony
    assert await adapter("/v1/responses", {}) == ([1, 2], 11)
    method = state.service._make_request_with_harmony if harmony else state.service._make_request
    method.assert_called_once_with(state.request, previous)
    assert state.maximum.call_args.kwargs == {"truncate_prompt_tokens": -1}


@pytest.mark.parametrize("case", ["builtin", "model", "input", "previous"])
async def test_responses_errors_prevent_rendering_and_budget(renderer: Any, case: str) -> None:
    adapter, state = renderer
    if case == "builtin":
        state.request.tools = [SimpleNamespace(type="web_search")]
    elif case == "model":
        state.service._check_model.return_value = "model rejected"
    elif case == "input":
        state.service._validate_create_responses_input.return_value = "input rejected"
    else:
        state.request.previous_response_id = "missing"
    with pytest.raises(ValueError):
        await adapter("/v1/responses", {})
    state.service._make_request.assert_not_called()
    state.maximum.assert_not_called()


@pytest.mark.parametrize(
    "rendered",
    [
        "native error",
        (None, []),
        (None, [{}, {}]),
        (None, [{"type": "multimodal"}]),
        (None, [{"type": "token"}]),
    ],
)
async def test_chat_render_errors_cannot_produce_budget(renderer: Any, rendered: Any) -> None:
    adapter, state = renderer
    state.service.render_chat_request.return_value = rendered
    with pytest.raises(ValueError):
        await adapter("/v1/chat/completions", {})
    state.maximum.assert_not_called()


async def test_chat_completion_limit_precedes_legacy_limit(renderer: Any) -> None:
    adapter, state = renderer
    state.request.max_completion_tokens = 6
    state.request.truncate_prompt_tokens = 5
    await adapter("/v1/chat/completions", {})
    assert state.maximum.call_args.args[1] == 6
    assert state.maximum.call_args.kwargs == {"truncate_prompt_tokens": 5}
    with pytest.raises(ValueError, match="unsupported preparation protocol"):
        await adapter("/unknown", {})


@pytest.mark.parametrize("bounds", [{"limit": 0}, {"ttl": 0}, {"ttl": -1}])
def test_preparation_rejects_invalid_bounds(bounds: Any) -> None:
    with pytest.raises(ValueError, match="invalid preparation bounds"):
        preparation.PreparationService(AsyncMock(), **bounds)


@pytest.mark.parametrize("tenant,protocol", [("", "/v1/messages"), ("tenant", "/unknown")])
async def test_preparation_scope_is_validated_before_render(tenant: str, protocol: str) -> None:
    render = AsyncMock()
    with pytest.raises(ValueError, match="invalid preparation scope"):
        await preparation.PreparationService(render).prepare(tenant, protocol, {})
    render.assert_not_awaited()


@pytest.mark.parametrize("value", ["NaN", "Inf", "-1"])
def test_telemetry_invalid_values_cannot_be_healthy(value: str) -> None:
    with pytest.raises(ValueError, match="invalid metric value"):
        telemetry.samples(
            f'vllm:num_requests_running{{model_name="model",engine="0"}} {value}',
            "model",
            "0",
        )


def test_telemetry_filters_dimensions_and_rejects_duplicate_series() -> None:
    metric = 'vllm:num_requests_running{model_name="model",engine="0"} 2'
    extra = (
        '\nvllm:num_requests_running{model_name="other",engine="0"} 8'
        '\nvllm:num_requests_running{model_name="model",engine="0",le="1"} 3'
        '\nvllm:num_requests_running{model_name="model",engine="0",reason="x"} 4'
    )
    assert telemetry.samples(metric + extra, "model", "0") == {"vllm:num_requests_running": 2}
    with pytest.raises(ValueError, match="ambiguous metric series"):
        telemetry.samples(metric + "\n" + metric, "model", "0")
    probe = collector()
    with pytest.raises(ValueError, match="one served model"):
        telemetry.TelemetryCollector(probe.capabilities.model_copy(update={"models": ()}), "engine")
    with pytest.raises(ValueError, match="non-integral"):
        probe.collect(
            payload(0, 0).replace('engine="0"} 2', 'engine="0"} 2.5'),
            free_vram_bytes=20,
            observed_at=datetime.now(UTC),
        )


@pytest.mark.parametrize("failure", [None, "health", "gpu", "heartbeat"])
async def test_telemetry_run_closes_channel_on_each_external_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any, failure: str | None
) -> None:
    capabilities = tmp_path / "caps.json"
    capabilities.write_text(collector().capabilities.model_dump_json())
    channel = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(grpc.aio, "insecure_channel", lambda _: channel)
    stub = object()
    monkeypatch.setattr(control_pb2_grpc, "WorkerControlServiceStub", lambda _: stub)
    response = SimpleNamespace(text=payload(0, 0), raise_for_status=MagicMock())
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = response
    if failure == "health":
        client.get.side_effect = RuntimeError("health failed")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    process = SimpleNamespace(
        returncode=1 if failure == "gpu" else 0,
        communicate=AsyncMock(return_value=(b"256\n", b"")),
    )
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    send = AsyncMock(return_value="accepted")
    if failure == "heartbeat":
        send.side_effect = RuntimeError("heartbeat failed")
    monkeypatch.setattr(telemetry, "heartbeat", send)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    args = argparse.Namespace(
        capabilities=capabilities,
        calibration=[],
        engine_instance_id="engine",
        scheduler="127.0.0.1:12345",
        samples=1,
        interval=1,
    )
    if failure is None:
        await telemetry.run(args)
        snapshot = send.call_args.args[1]
        assert snapshot.free_vram_bytes == 256 * 1024**2
        assert snapshot.queue_depth == 3
        assert json.loads(capsys.readouterr().out)["status"] == "accepted"
        assert spawn.call_args.args[:2] == ("nvidia-smi", "--id=0")
    else:
        with pytest.raises(RuntimeError):
            await telemetry.run(args)
        if failure in {"health", "gpu"}:
            send.assert_not_awaited()
    channel.close.assert_awaited_once()


async def test_heartbeat_serializes_fenced_identity() -> None:
    probe = collector()
    snapshot = probe.collect(payload(0, 0), free_vram_bytes=30, observed_at=datetime.now(UTC))
    captured = []

    class Stub:
        def Heartbeat(self, stream: Any) -> Any:
            async def read() -> Any:
                captured.extend([item async for item in stream])
                return SimpleNamespace(status="accepted")

            return SimpleNamespace(read=read)

    assert await telemetry.heartbeat(Stub(), snapshot) == "accepted"
    assert len(captured) == 1
    assert captured[0].worker_id == "worker"
    assert captured[0].generation == 1
    assert captured[0].context.tenant_id == "system"
    assert captured[0].telemetry_json == snapshot.model_dump_json()


@pytest.mark.parametrize("samples,interval", [(0, 1), (1, 0), (1, 16)])
def test_telemetry_cli_bounds(monkeypatch: pytest.MonkeyPatch, samples: int, interval: int) -> None:
    run = AsyncMock()
    monkeypatch.setattr(telemetry, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "telemetry",
            "--capabilities",
            "unused",
            "--engine-instance-id",
            "engine",
            "--scheduler",
            "127.0.0.1:1",
            "--samples",
            str(samples),
            "--interval",
            str(interval),
        ],
    )
    with pytest.raises(SystemExit) as error:
        telemetry.main()
    assert error.value.code == 2
    run.assert_not_called()


def test_telemetry_cli_passes_valid_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    run = AsyncMock()
    monkeypatch.setattr(telemetry, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "telemetry",
            "--capabilities",
            "caps",
            "--engine-instance-id",
            "engine",
            "--scheduler",
            "127.0.0.1:1",
            "--samples",
            "1",
            "--interval",
            "0.1",
        ],
    )
    telemetry.main()
    assert run.await_args is not None
    assert run.await_args.args[0].interval == 0.1


@pytest.mark.parametrize("mode", ["success", "port", "init", "http", "registration"])
async def test_managed_serve_always_closes_journal_control_and_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    monkeypatch.setenv("FREECHAT_WORKER_TOKEN", "t" * 32)
    monkeypatch.delenv("FREECHAT_RUNTIME_ID", raising=False)
    engine: Any = CacheEngine()
    engine.model_config = SimpleNamespace(model="/model")
    engine.get_supported_tasks = AsyncMock(return_value=("generate",))
    exited = []

    @asynccontextmanager
    async def engine_context(args: Any) -> AsyncIterator[CacheEngine]:
        try:
            yield engine
        finally:
            exited.append(True)

    app = SimpleNamespace(state=SimpleNamespace())
    init = AsyncMock(side_effect=RuntimeError("init failed") if mode == "init" else None)
    api = SimpleNamespace(
        build_async_engine_client=engine_context,
        build_app=MagicMock(return_value=app),
        init_app_state=init,
    )
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.api_server", api)
    monkeypatch.setattr(serve, "measure_capacity", AsyncMock(return_value=object()))
    control = MagicMock()
    control.add_insecure_port.return_value = 0 if mode == "port" else 1234
    control.start = AsyncMock()
    control.stop = AsyncMock()
    monkeypatch.setattr(grpc.aio, "server", lambda: control)
    drivers = []

    def driver_factory(*args: Any, **kwargs: Any) -> DurableExecutionDriver:
        driver = DurableExecutionDriver(*args, **kwargs)
        drivers.append(driver)
        return driver

    monkeypatch.setattr(serve, "DurableExecutionDriver", driver_factory)
    registration_started = asyncio.Event()
    registration_cancelled = asyncio.Event()

    async def register() -> None:
        registration_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            registration_cancelled.set()

    monkeypatch.setattr(serve, "artifact_identity", lambda _: "sha")
    monkeypatch.setattr(serve, "capabilities", lambda *_, **__: object())
    monkeypatch.setattr(serve, "RegistrationLoop", lambda *args: SimpleNamespace(run=register))

    class Server:
        started = False

        def __init__(self, config: Any) -> None:
            assert config.host == "127.0.0.1"

        async def serve(self) -> None:
            if mode == "http":
                raise RuntimeError("http failed")
            if mode == "registration":
                await asyncio.sleep(0.06)
                self.started = True
                await asyncio.wait_for(registration_started.wait(), timeout=1)

    monkeypatch.setattr(uvicorn, "Server", Server)
    args = SimpleNamespace(
        worker_id="worker",
        state_dir=tmp_path,
        worker_generation=1,
        scheduler_target="127.0.0.1:12345" if mode == "registration" else None,
        node_id="node",
        port=8000,
        control_port=50052,
        host="127.0.0.1",
        uvicorn_log_level="error",
    )
    if mode in {"port", "init", "http"}:
        with pytest.raises(RuntimeError):
            await serve.serve(args)
    else:
        await serve.serve(args)
    assert exited == [True]
    assert len(drivers) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        drivers[0]._db.execute("SELECT 1")
    if mode == "port":
        control.start.assert_not_awaited()
    else:
        control.stop.assert_awaited_once_with(grace=5)
    if mode == "registration":
        assert registration_cancelled.is_set()


@pytest.mark.parametrize("case", ["token", "worker_id"])
async def test_serve_rejects_bad_identity_before_creating_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    monkeypatch.setenv("FREECHAT_WORKER_TOKEN", "" if case == "token" else "t" * 32)
    with pytest.raises(ValueError):
        await serve.serve(SimpleNamespace(worker_id="../escape", state_dir=tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("generation", [0, 4])
def test_serve_cli_validates_generation_and_native_defaults(
    monkeypatch: pytest.MonkeyPatch, generation: int
) -> None:
    validate = MagicMock()
    api = SimpleNamespace(
        FlexibleArgumentParser=argparse.ArgumentParser,
        make_arg_parser=lambda parser: parser,
        validate_parsed_serve_args=validate,
    )
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.api_server", api)
    run = AsyncMock()
    monkeypatch.setattr(serve, "serve", run)
    monkeypatch.setattr(
        sys, "argv", ["serve", "--worker-id", "worker", "--worker-generation", str(generation)]
    )
    if generation == 0:
        with pytest.raises(SystemExit) as error:
            serve.main()
        assert error.value.code == 2
        run.assert_not_called()
        validate.assert_not_called()
    else:
        serve.main()
        assert run.await_args is not None
        args = run.await_args.args[0]
        assert args.host == "127.0.0.1"
        assert args.async_scheduling is False
        assert args.worker_extension_cls == "freechat_worker.capacity.CapacityExtension"
        validate.assert_called_once_with(args)


@pytest.mark.parametrize("failure", ["cancelled", "handler"])
async def test_native_handler_failure_closes_route_and_restores_context(
    tmp_path: Path, failure: str
) -> None:
    from freechat_worker.native_serving import (
        CURRENT_ROUTE,
        AdmissionMiddleware,
        NativeExecutionBackend,
    )
    from test_native_serving import headers

    backend = NativeExecutionBackend(CacheEngine())
    gate = DurableExecutionDriver(
        tmp_path / "native.sqlite",
        backend,
        worker_id="worker",
        generation=1,
        engine_instance_id="engine",
        create=True,
    )
    captured = []

    async def handler(scope: Any, receive: Any, send: Any) -> None:
        captured.append(CURRENT_ROUTE.get())
        assert (await receive())["body"] == b"{}"
        assert await receive() == {"type": "http.disconnect"}
        if failure == "cancelled":
            raise asyncio.CancelledError
        raise RuntimeError("native handler failed")

    middleware = AdmissionMiddleware(handler, driver=gate, backend=backend, token="t" * 32)
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": b"{}"},
            {"type": "http.disconnect"},
        ]
    )
    prior = CURRENT_ROUTE.set("caller-context")
    try:
        with pytest.raises(asyncio.CancelledError if failure == "cancelled" else RuntimeError):
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": [(key.encode(), value.encode()) for key, value in headers().items()],
                },
                receive,
                AsyncMock(),
            )
        assert CURRENT_ROUTE.get() == "caller-context"
        assert len(captured) == 1
        assert captured[0] is not None
        assert backend.routes[captured[0]].closed
        assert not backend.routes[captured[0]].aborted
    finally:
        CURRENT_ROUTE.reset(prior)
        gate.close()


@pytest.mark.parametrize(
    "case,status", [("disconnect", None), ("oversize", 413), ("list", 409), ("json", 409)]
)
async def test_native_ingress_body_failure_never_admits(case: str, status: int | None) -> None:
    from freechat_worker.native_serving import AdmissionMiddleware, NativeExecutionBackend

    backend = NativeExecutionBackend(CacheEngine())
    gate = MagicMock()
    gate.identity = ("worker", 1, "engine")
    gate.admit = AsyncMock()
    app = AsyncMock()
    middleware = AdmissionMiddleware(app, driver=gate, backend=backend, token="t" * 32)
    message = {
        "disconnect": {"type": "http.disconnect"},
        "oversize": {"type": "http.request", "body": b"x" * (4 * 1024 * 1024 + 1)},
        "list": {"type": "http.request", "body": b"[]"},
        "json": {"type": "http.request", "body": b"invalid"},
    }[case]
    send = AsyncMock()
    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [(b"x-freechat-worker-token", b"t" * 32)],
        },
        AsyncMock(return_value=message),
        send,
    )
    gate.admit.assert_not_awaited()
    app.assert_not_awaited()
    assert backend.routes == {}
    if status is None:
        send.assert_not_awaited()
    else:
        assert send.await_args_list[0].args[0]["status"] == status


async def test_native_proxy_rejects_unknown_routes_and_untracked_entrypoints() -> None:
    from freechat_worker.native_serving import (
        CURRENT_ROUTE,
        NativeEngineClient,
        NativeExecutionBackend,
    )

    backend = NativeExecutionBackend(CacheEngine())
    client = NativeEngineClient(backend.children.engine, backend)
    for name in ("add_request", "encode"):
        with pytest.raises(AttributeError, match="untracked"):
            getattr(client, name)
    with pytest.raises(ValueError, match="abort_without_route_identity"):
        await client.abort("external")
    with pytest.raises(ValueError, match="native_route_unknown"):
        await backend.abort("missing")
    with pytest.raises(ValueError, match="invalid"):
        await backend.submit("bad", {"unknown": True})
    await backend.submit("route", {})
    with pytest.raises(ValueError, match="already_admitted"):
        await backend.submit("route", {})
    token = CURRENT_ROUTE.set("route")
    try:
        backend.routes["route"].children = ["placeholder"] * 128
        with pytest.raises(ValueError, match="child_limit"):
            await anext(client.generate("text", SimpleNamespace(n=1), "external"))
        assert len(backend.routes["route"].children) == 128
        backend.close_submission("route")
        with pytest.raises(ValueError, match="submission_closed"):
            await anext(client.generate("text", SimpleNamespace(n=1), "external"))
    finally:
        CURRENT_ROUTE.reset(token)
