"""CPU gRPC-handler boundaries with abort-faithful contexts and external fakes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import grpc
import pytest
from freechat.control.v1 import control_pb2
from freechat_contracts import WorkerTelemetry
from freechat_scheduler import grpc_server
from freechat_scheduler.grpc_server import SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry, PersistentWorkerRegistry
from freechat_scheduler.request_ledger import RequestLedger
from freechat_scheduler.scheduler import Scheduler
from test_scheduler import MODEL, add_worker, profile


class Aborted(Exception):
    def __init__(self, code: grpc.StatusCode, detail: str) -> None:
        self.code, self.detail = code, detail
        super().__init__(detail)


class Context:
    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self.metadata = metadata

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self.metadata

    async def abort(self, code: grpc.StatusCode, detail: str) -> None:
        raise Aborted(code, detail)


def route_request(**changes: Any) -> Any:
    return control_pb2.RouteRequest(
        context=control_pb2.RequestContext(request_id="request", tenant_id="tenant-a"),
        hints=control_pb2.AgentHints(harness_id="test", task_id="task", agent_id="agent"),
        model=MODEL.model_id,
        input_tokens=2,
        output_tokens=3,
        **changes,
    )


@pytest.fixture
def service() -> tuple[SchedulerGrpcService, InMemoryWorkerRegistry, RequestLedger]:
    registry = InMemoryWorkerRegistry()
    ledger = RequestLedger()
    return SchedulerGrpcService(Scheduler(registry), ledger), registry, ledger


@pytest.mark.parametrize("case", ["native", "bad_lifecycle", "no_worker"])
async def test_route_rejection_creates_no_reservation(service: Any, case: str) -> None:
    handler, _, ledger = service
    request = route_request()
    if case == "native":
        request.native_protocol = "/v1/messages"
        request.native_request_json = '{"model":"' + MODEL.model_id + '"}'
    elif case == "bad_lifecycle":
        request.hints.lifecycle = "not-a-lifecycle"
    with pytest.raises(Aborted) as error:
        await handler.Route(request, Context())
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert not (await ledger.snapshot()).reservations


@pytest.mark.parametrize("method", ["RenewLease", "Release", "Cancel", "ExplainDecision"])
async def test_unknown_lease_cannot_be_mutated_or_explained(service: Any, method: str) -> None:
    handler, _, ledger = service
    with pytest.raises(Aborted) as error:
        await getattr(handler, method)(
            control_pb2.LeaseRequest(
                context=control_pb2.RequestContext(tenant_id="tenant"),
                decision_id="missing",
                worker_id="worker",
                worker_generation=1,
            ),
            Context(),
        )
    assert error.value.code == (
        grpc.StatusCode.NOT_FOUND
        if method == "ExplainDecision"
        else grpc.StatusCode.FAILED_PRECONDITION
    )
    assert not (await ledger.snapshot()).reservations


async def test_explain_uses_original_owned_decision(service: Any) -> None:
    handler, registry, ledger = service
    add_worker(registry, "worker", node="ross")
    req = profile()
    decision = await ledger.reserve(req, "once", lambda _: handler._scheduler.route(req))
    response = await handler.ExplainDecision(
        control_pb2.LeaseRequest(
            context=control_pb2.RequestContext(tenant_id=req.tenant_id),
            decision_id=decision.decision_id,
            worker_id="worker",
            worker_generation=1,
        ),
        Context(),
    )
    assert response.decision_id == decision.decision_id
    assert response.worker_id == "worker"


@pytest.mark.parametrize("case", ["missing", "oversized", "malformed", "not_found"])
async def test_cache_configuration_and_payload_rejections(service: Any, case: str) -> None:
    handler, _, _ = service
    cache = AsyncMock()
    if case != "missing":
        handler._cache = cache
    cache.apply.side_effect = KeyError("private lookup")
    payload = (
        "x" * 16385
        if case == "oversized"
        else "invalid"
        if case == "malformed"
        else '{"decision_id":"decision","sequence":1,"lifecycle":"resume"}'
    )
    with pytest.raises(Aborted) as error:
        await handler.UpdateCacheLifecycle(
            control_pb2.CacheLifecycleUpdateRequest(
                context=control_pb2.RequestContext(tenant_id="tenant"), update_json=payload
            ),
            Context(),
        )
    assert error.value.code == {
        "missing": grpc.StatusCode.UNAVAILABLE,
        "not_found": grpc.StatusCode.NOT_FOUND,
    }.get(case, grpc.StatusCode.FAILED_PRECONDITION)
    if case == "not_found":
        assert "private lookup" not in error.value.detail
        assert cache.apply.await_args.args[0] == "tenant"
    else:
        cache.apply.assert_not_awaited()


@pytest.mark.parametrize(
    "code",
    [
        grpc.StatusCode.FAILED_PRECONDITION,
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.PERMISSION_DENIED,
    ],
)
async def test_cache_worker_transport_codes_are_sanitized(service: Any, code: Any) -> None:
    handler, _, _ = service
    cache = AsyncMock()
    cache.apply.side_effect = grpc.aio.AioRpcError(code, None, None, "private worker detail")
    handler._cache = cache
    with pytest.raises(Aborted) as error:
        await handler.UpdateCacheLifecycle(
            control_pb2.CacheLifecycleUpdateRequest(
                update_json='{"decision_id":"decision","sequence":1,"lifecycle":"terminal"}'
            ),
            Context(),
        )
    assert error.value.code == (
        code
        if code
        in {
            grpc.StatusCode.FAILED_PRECONDITION,
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
        }
        else grpc.StatusCode.UNAVAILABLE
    )
    assert "private worker detail" not in error.value.detail
    cache.apply.assert_awaited_once()


@pytest.mark.parametrize(
    "method",
    ["Route", "RenewLease", "Release", "Cancel", "ExplainDecision", "UpdateCacheLifecycle"],
)
async def test_control_authentication_precedes_payload_access(service: Any, method: str) -> None:
    handler, _, ledger = service
    handler._token = "secret"
    with pytest.raises(Aborted) as error:
        await getattr(handler, method)(None, Context())
    assert error.value.code == grpc.StatusCode.UNAUTHENTICATED
    assert not (await ledger.snapshot()).reservations


@pytest.mark.parametrize("case", ["worker", "generation", "endpoint", "json"])
async def test_registration_envelope_cannot_replace_registry_identity(case: str) -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    caps = registry.snapshot()[1][0].capabilities
    request = control_pb2.WorkerRegistration(
        worker_id="other" if case == "worker" else caps.worker_id,
        generation=2 if case == "generation" else caps.generation,
        endpoint="http://other:8000" if case == "endpoint" else caps.endpoint,
        capabilities_json="invalid" if case == "json" else caps.model_dump_json(),
    )
    before = registry.snapshot()
    with pytest.raises(Aborted) as error:
        await WorkerGrpcService(registry).Register(request, Context())
    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert registry.snapshot() == before


async def test_registration_emits_stable_event_with_system_scope() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    caps = registry.snapshot()[1][0].capabilities
    emitter = AsyncMock()
    handler = WorkerGrpcService(registry, emitter=emitter)
    request = control_pb2.WorkerRegistration(
        context=control_pb2.RequestContext(idempotency_key="once", tenant_id="forged"),
        worker_id=caps.worker_id,
        generation=caps.generation,
        endpoint=caps.endpoint,
        capabilities_json=caps.model_dump_json(),
    )
    await handler.Register(request, Context())
    await handler.Register(request, Context())
    first, second = [call.args[0] for call in emitter.emit.await_args_list]
    assert first.event_id == second.event_id
    assert first.tenant_id == "system" and first.aggregate_id == "worker"


@pytest.mark.parametrize("case", ["worker", "generation", "json", "unregistered"])
async def test_heartbeat_rejects_unknown_or_mismatched_owner(case: str) -> None:
    registry = InMemoryWorkerRegistry()
    if case != "unregistered":
        add_worker(registry, "worker", node="ross")
    snapshot = WorkerTelemetry(worker_id="worker", generation=1, free_vram_bytes=20)
    request = control_pb2.WorkerHeartbeat(
        worker_id="other" if case == "worker" else "worker",
        generation=2 if case == "generation" else 1,
        telemetry_json="invalid" if case == "json" else snapshot.model_dump_json(),
    )

    async def stream() -> AsyncIterator[Any]:
        yield request

    before = registry.snapshot()
    with pytest.raises(Aborted) as error:
        await anext(WorkerGrpcService(registry).Heartbeat(stream(), Context()))
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert registry.snapshot() == before


@pytest.mark.parametrize("kind", ["emit", "logs"])
async def test_flush_acknowledges_only_successfully_delivered_events(
    service: Any, caplog: Any, kind: str
) -> None:
    handler, registry, ledger = service
    add_worker(registry, "worker", node="ross")
    req = profile()
    await ledger.reserve(req, "once", lambda _: handler._scheduler.route(req))
    assert (await ledger.snapshot()).pending
    caplog.set_level(logging.INFO)
    emitter = AsyncMock()
    if kind == "emit":
        handler._emitter = emitter
        emitter.emit.side_effect = RuntimeError("NATS down")
        with pytest.raises(RuntimeError, match="NATS down"):
            await handler.flush_events()
        assert (await ledger.snapshot()).pending
        emitter.emit.side_effect = None
    else:
        handler._log_events = True
    await handler.flush_events()
    assert not (await ledger.snapshot()).pending
    if kind == "logs":
        assert "lifecycle_event" in caplog.text


@pytest.mark.parametrize("failure", ["flush", "expiry", "tick", "tick_errors", "none"])
async def test_maintenance_continues_after_failures_and_cancels_cleanly(
    service: Any, monkeypatch: pytest.MonkeyPatch, caplog: Any, failure: str
) -> None:
    handler, _, ledger = service
    flush = AsyncMock(side_effect=RuntimeError("flush failed") if failure == "flush" else None)
    expire = AsyncMock(side_effect=RuntimeError("expiry failed") if failure == "expiry" else None)
    monkeypatch.setattr(handler, "flush_events", flush)
    monkeypatch.setattr(ledger, "expire", expire)
    execution = AsyncMock()
    execution.tick.return_value = {"worker": "unavailable"} if failure == "tick_errors" else {}
    if failure == "tick":
        execution.tick.side_effect = RuntimeError("tick failed")
    handler._execution = execution
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await handler.maintain()
    flush.assert_awaited_once()
    expire.assert_awaited_once()
    execution.tick.assert_awaited_once()
    if failure != "none":
        assert "failed" in caplog.text or "unavailable" in caplog.text


@pytest.mark.parametrize("mode", ["simple", "external", "groups", "port"])
async def test_scheduler_startup_and_owned_resources_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, mode: str
) -> None:
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from freechat_control_store import InMemoryStore

    for key in (
        "ETCD_ENDPOINT",
        "NATS_URL",
        "FREECHAT_GROUP_RUNTIME_CONFIG",
        "FREECHAT_REQUEST_EXECUTION_CONFIG",
        "FREECHAT_RETIRED_EXECUTION_ROUTES",
    ):
        monkeypatch.delenv(key, raising=False)

    class Store(InMemoryStore):
        closed = False

        async def close(self) -> None:
            self.closed = True

    store = Store()
    replay = AsyncMock()
    close_nats = AsyncMock()
    if mode == "external":
        monkeypatch.setenv("ETCD_ENDPOINT", "http://etcd:2379")
        monkeypatch.setenv("NATS_URL", "nats://fixture:4222")
        monkeypatch.setattr(grpc_server, "EtcdHttpStore", lambda _: store)
        monkeypatch.setattr(
            grpc_server,
            "connect_lifecycle_stream",
            AsyncMock(return_value=("nats-client", object())),
        )
        monkeypatch.setattr(grpc_server, "close_lifecycle_stream", close_nats)
        monkeypatch.setattr(
            grpc_server,
            "DurableLifecycleEmitter",
            lambda *_: SimpleNamespace(replay=replay, emit=AsyncMock()),
        )
    group_started = asyncio.Event()
    group_cancelled = asyncio.Event()
    tick = AsyncMock()

    async def group_run() -> None:
        group_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            group_cancelled.set()

    if mode == "groups":
        config = tmp_path / "group.json"
        config.write_text(
            json.dumps(
                {
                    "mode": "local-contract",
                    "owner_id": "owner",
                    "inventory": {
                        "generation": 1,
                        "devices": [
                            {"gpu_id": "gpu", "node_id": "node", "compute_capability": "8.6"}
                        ],
                    },
                    "endpoints": {},
                }
            )
        )
        execution = tmp_path / "execution.json"
        execution.write_text('{"mode":"local-contract","endpoints":{}}')
        monkeypatch.setenv("FREECHAT_GROUP_RUNTIME_CONFIG", str(config))
        monkeypatch.setenv("FREECHAT_REQUEST_EXECUTION_CONFIG", str(execution))
        monkeypatch.setattr(
            grpc_server,
            "configured_reconciler",
            lambda *_: SimpleNamespace(
                tick=tick,
                run=group_run,
                snapshot=lambda: None,
            ),
        )

    control = MagicMock()
    control.add_insecure_port.return_value = 0 if mode == "port" else 12345
    control.start = AsyncMock()
    control.stop = AsyncMock()

    async def terminate() -> None:
        if mode == "groups":
            await asyncio.wait_for(group_started.wait(), timeout=1)
        else:
            await asyncio.sleep(0)

    control.wait_for_termination = AsyncMock(side_effect=terminate)
    monkeypatch.setattr(grpc.aio, "server", lambda: control)
    if mode == "port":
        with pytest.raises(RuntimeError, match="port unavailable"):
            await grpc_server.serve("127.0.0.1:0", contract_only=True)
        control.start.assert_not_awaited()
    else:
        await grpc_server.serve("127.0.0.1:0", contract_only=True)
        control.start.assert_awaited_once()
        control.stop.assert_awaited_once_with(grace=5)
        if mode == "external":
            replay.assert_awaited_once()
            assert store.closed
            close_nats.assert_awaited_once_with("nats-client")
        if mode == "groups":
            tick.assert_awaited_once()
            assert group_cancelled.is_set()


async def test_signal_runner_removes_handlers_after_serve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal
    from unittest.mock import MagicMock

    loop = asyncio.get_running_loop()
    add = MagicMock()
    remove = MagicMock()
    monkeypatch.setattr(loop, "add_signal_handler", add)
    monkeypatch.setattr(loop, "remove_signal_handler", remove)
    run = AsyncMock(side_effect=RuntimeError("start failed"))
    monkeypatch.setattr(grpc_server, "serve", run)
    with pytest.raises(RuntimeError, match="start failed"):
        await grpc_server._run_until_signal()
    assert [call.args[0] for call in add.call_args_list] == [signal.SIGTERM, signal.SIGINT]
    assert [call.args[0] for call in remove.call_args_list] == [signal.SIGTERM, signal.SIGINT]


def test_scheduler_entrypoint_runs_signal_managed_server(monkeypatch: pytest.MonkeyPatch) -> None:
    run = AsyncMock()
    monkeypatch.setattr(grpc_server, "_run_until_signal", run)
    grpc_server.run()
    run.assert_awaited_once()


@pytest.mark.parametrize(
    "failure", [None, "restore", "connect", "replay", "port", "start", "stop", "nats_close"]
)
async def test_scheduler_partial_startup_releases_each_acquired_resource_once(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from freechat_control_store import InMemoryStore

    for key in (
        "FREECHAT_GROUP_RUNTIME_CONFIG",
        "FREECHAT_REQUEST_EXECUTION_CONFIG",
        "FREECHAT_RETIRED_EXECUTION_ROUTES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ETCD_ENDPOINT", "http://etcd:2379")
    monkeypatch.setenv("NATS_URL", "nats://fixture:4222")
    released: list[str] = []

    class Store(InMemoryStore):
        async def close(self) -> None:
            released.append("etcd")

    store = Store()
    monkeypatch.setattr(grpc_server, "EtcdHttpStore", lambda _: store)
    if failure == "restore":
        monkeypatch.setattr(
            PersistentWorkerRegistry,
            "restore",
            AsyncMock(side_effect=RuntimeError("restore failed")),
        )
    connect = AsyncMock(return_value=("client", object()))
    if failure == "connect":
        connect.side_effect = RuntimeError("connect failed")
    monkeypatch.setattr(grpc_server, "connect_lifecycle_stream", connect)

    async def close_client(client: str) -> None:
        assert client == "client"
        released.append("nats")
        if failure == "nats_close":
            raise RuntimeError("nats close failed")

    monkeypatch.setattr(grpc_server, "close_lifecycle_stream", close_client)
    replay = AsyncMock(side_effect=RuntimeError("replay failed") if failure == "replay" else None)
    monkeypatch.setattr(
        grpc_server,
        "DurableLifecycleEmitter",
        lambda *_: SimpleNamespace(replay=replay, emit=AsyncMock()),
    )
    server = MagicMock()
    server.add_insecure_port.return_value = 0 if failure == "port" else 12345
    server.start = AsyncMock(
        side_effect=RuntimeError("start failed") if failure == "start" else None
    )
    server.wait_for_termination = AsyncMock()

    async def stop(*, grace: int) -> None:
        assert grace == 5
        released.append("grpc")
        if failure == "stop":
            raise RuntimeError("stop failed")

    server.stop = AsyncMock(side_effect=stop)
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(grpc.aio, "server", factory)
    if failure is None:
        await grpc_server.serve("127.0.0.1:0", contract_only=True)
    else:
        with pytest.raises(RuntimeError):
            await grpc_server.serve("127.0.0.1:0", contract_only=True)
    if failure in {"restore", "connect"}:
        assert released == ["etcd"]
        factory.assert_not_called()
        replay.assert_not_awaited()
    elif failure == "replay":
        assert released == ["nats", "etcd"]
        factory.assert_not_called()
    else:
        assert released == ["grpc", "nats", "etcd"]
        server.stop.assert_awaited_once_with(grace=5)
        if failure == "port":
            server.start.assert_not_awaited()
        else:
            server.start.assert_awaited_once()
