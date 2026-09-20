import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import grpc
import pytest
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import RouteDecision
from freechat_contracts.execution import (
    ExecutionAction,
    ExecutionCommand,
    ExecutionReceipt,
    ExecutionStatus,
)
from freechat_control_store import InMemoryStore
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler import grpc_server
from freechat_scheduler.group_runtime import LocalRuntimeEndpoint
from freechat_scheduler.request_execution import (
    ExecutionDriver,
    LocalGrpcExecutionDriver,
    LocalRequestExecutionService,
    RequestExecutionConfig,
    RequestExecutionReconciler,
)
from freechat_scheduler.request_ledger import RequestLedger, RequestState
from freechat_scheduler.scheduler import NoEligibleWorker
from freechat_worker.execution import DurableExecutionDriver, EngineObservation
from pydantic import SecretStr
from test_request_ledger import FaultStore, request, reserve, setup


def command(
    route: RouteDecision, action: ExecutionAction = ExecutionAction.QUERY
) -> ExecutionCommand:
    assert route.engine_instance_id
    return ExecutionCommand(
        tenant_id="tenant-a",
        request_id=route.request_id,
        decision_id=route.decision_id,
        worker_id=route.worker_id,
        worker_generation=route.worker_generation,
        engine_instance_id=route.engine_instance_id,
        action=action,
    )


def receipt(route: RouteDecision, **updates: Any) -> ExecutionReceipt:
    return ExecutionReceipt(
        command=command(route),
        observation_sequence=1,
        observed_at=datetime.now(UTC),
        status=ExecutionStatus.COMPLETED,
        quiescent=True,
        admission_closed=True,
    ).model_copy(update=updates)


@pytest.mark.parametrize("status", list(ExecutionStatus))
@pytest.mark.parametrize(
    "quiet,closed", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_only_terminal_quiescent_admission_closed_receipt_releases(
    status: ExecutionStatus, quiet: bool, closed: bool
) -> None:
    ledger, scheduler = RequestLedger(), setup()
    route = await reserve(ledger, scheduler, request())
    await ledger.release(route.decision_id, route.worker_id, route.worker_generation, "tenant-a")
    proof = receipt(route, status=status, quiescent=quiet, admission_closed=closed)
    await ledger.observe_execution(proof)
    state = (await ledger.snapshot()).reservations[route.decision_id]
    assert (state.state is RequestState.RELEASED) == proof.releasable
    assert state.execution_receipt == proof
    if not proof.releasable:
        with pytest.raises(NoEligibleWorker):
            await reserve(ledger, scheduler, request("next"))
    else:
        await reserve(ledger, scheduler, request("next"))


async def test_duplicate_receipt_is_idempotent_and_cannot_reopen_terminal_request() -> None:
    ledger, scheduler = RequestLedger(), setup()
    route = await reserve(ledger, scheduler, request())
    proof = receipt(route)
    await asyncio.gather(*(ledger.observe_execution(proof) for _ in range(8)))
    snapshot = await ledger.snapshot()
    assert len(snapshot.pending) == 2
    assert snapshot.reservations[route.decision_id].state is RequestState.RELEASED
    with pytest.raises(ValueError, match="terminal"):
        await ledger.observe_execution(
            proof.model_copy(update={"observation_sequence": 2, "status": ExecutionStatus.RUNNING})
        )
    await ledger.release(
        route.decision_id, route.worker_id, route.worker_generation, "tenant-a", cancelled=True
    )
    assert len((await ledger.snapshot()).pending) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", "other"),
        ("request_id", "other"),
        ("decision_id", "other"),
        ("worker_id", "other"),
        ("worker_generation", 999),
        ("engine_instance_id", "restarted"),
    ],
)
async def test_receipt_binding_cannot_release_another_execution(field: str, value: Any) -> None:
    ledger, scheduler = RequestLedger(), setup()
    route = await reserve(ledger, scheduler, request())
    proof = receipt(route, command=command(route).model_copy(update={field: value}))
    with pytest.raises((ValueError, KeyError)):
        await ledger.observe_execution(proof)
    assert len((await ledger.snapshot()).pending) == 1


@pytest.mark.parametrize(
    "when",
    [
        datetime.now(UTC) - timedelta(minutes=1),
        datetime.now(UTC) + timedelta(minutes=1),
        datetime(2026, 1, 1),
    ],
)
async def test_stale_future_and_naive_receipts_retain_capacity(when: datetime) -> None:
    ledger, scheduler = RequestLedger(), setup()
    route = await reserve(ledger, scheduler, request())
    with pytest.raises(ValueError, match="stale"):
        await ledger.observe_execution(receipt(route, observed_at=when))
    assert (await ledger.snapshot()).reservations[route.decision_id].state is RequestState.ACTIVE


async def test_observation_sequence_and_repeated_running_poll_do_not_flood_outbox() -> None:
    ledger, scheduler = RequestLedger(), setup()
    route = await reserve(ledger, scheduler, request())
    first = receipt(route, status=ExecutionStatus.RUNNING, quiescent=False, admission_closed=False)
    await ledger.observe_execution(first)
    next_receipt = first.model_copy(update={"observation_sequence": 2})
    await ledger.observe_execution(next_receipt)
    assert len((await ledger.snapshot()).pending) == 2
    with pytest.raises(ValueError, match="regressed"):
        await ledger.observe_execution(first)
    with pytest.raises(ValueError, match="conflict"):
        await ledger.observe_execution(next_receipt.model_copy(update={"quiescent": True}))


@pytest.mark.parametrize("failure", ["before", "after"])
async def test_terminal_receipt_and_release_event_are_one_commit(failure: str) -> None:
    store, scheduler = FaultStore(), setup()
    ledger = RequestLedger(store)
    route = await reserve(ledger, scheduler, request())
    proof = receipt(route)
    store.failure = failure
    with pytest.raises(OSError):
        await ledger.observe_execution(proof)
    rebuilt = RequestLedger(store)
    state = await rebuilt.snapshot()
    assert len(state.pending) == (1 if failure == "before" else 2)
    assert (state.reservations[route.decision_id].state is RequestState.RELEASED) == (
        failure == "after"
    )
    await rebuilt.observe_execution(proof)
    assert len((await rebuilt.snapshot()).pending) == 2


class FixtureExecutionDriver:
    """Observations only; does not implement model execution or real admission tombstones."""

    def __init__(self) -> None:
        self.commands: list[ExecutionCommand] = []
        self.status = ExecutionStatus.RUNNING
        self.quiet = False
        self.closed = False
        self.failure = False
        self.mismatch = False

    async def observe(self, request: ExecutionCommand) -> ExecutionReceipt:
        self.commands.append(request)
        if self.failure:
            raise ConnectionError("local fixture unavailable")
        return ExecutionReceipt(
            command=request
            if not self.mismatch
            else request.model_copy(update={"tenant_id": "wrong"}),
            observation_sequence=len(self.commands),
            observed_at=datetime.now(UTC),
            status=self.status,
            quiescent=self.quiet,
            admission_closed=self.closed,
        )


async def test_cancel_and_expired_completion_reconcile_abort_not_query() -> None:
    clock = [datetime.now(UTC)]
    ledger, scheduler = RequestLedger(clock=lambda: clock[0]), setup(slots=2)
    first = await reserve(ledger, scheduler, request("cancel"))
    second = await reserve(ledger, scheduler, request("expire"))
    await ledger.release(
        first.decision_id, first.worker_id, first.worker_generation, "tenant-a", cancelled=True
    )
    await ledger.release(second.decision_id, second.worker_id, second.worker_generation, "tenant-a")
    clock[0] += timedelta(seconds=31)
    await ledger.expire()
    assert (await ledger.snapshot()).reservations[second.decision_id].state is RequestState.EXPIRED
    # Late transport completion must not reset an abort intent.
    await ledger.release(first.decision_id, first.worker_id, first.worker_generation, "tenant-a")
    driver = FixtureExecutionDriver()
    driver.failure = True
    runtime = RequestExecutionReconciler(ledger, driver)
    assert len(await runtime.tick()) == 2
    assert {cmd.action for cmd in driver.commands} == {ExecutionAction.ABORT}
    assert len({cmd.operation_id for cmd in driver.commands}) == 2
    assert all(
        lease.state is not RequestState.RELEASED
        for lease in (await ledger.snapshot()).reservations.values()
    )


async def test_unbound_engine_and_mismatched_driver_response_fail_closed() -> None:
    ledger, scheduler, driver = RequestLedger(), setup(slots=2), FixtureExecutionDriver()
    req = request("unbound")
    await ledger.reserve(
        req,
        req.request_id,
        lambda held: scheduler.route(req, reserved=held).model_copy(
            update={"engine_instance_id": None}
        ),
    )
    route = await reserve(ledger, scheduler, request("bound"))
    driver.mismatch = True
    assert len(await RequestExecutionReconciler(ledger, driver).tick()) == 2
    assert len(driver.commands) == 1
    assert driver.commands[0].decision_id == route.decision_id


async def start_runtime(driver: ExecutionDriver) -> tuple[Any, LocalRuntimeEndpoint]:
    server = grpc.aio.server()
    endpoint = LocalRuntimeEndpoint(address="127.0.0.1:1", token=SecretStr("fixture-only-" * 4))
    control_pb2_grpc.add_RequestExecutionServiceServicer_to_server(  # type: ignore[no-untyped-call]
        LocalRequestExecutionService(
            driver,
            worker_id="worker-0",
            generation=1,
            engine_instance_id="fixture-engine",
            token=endpoint.token,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, endpoint.model_copy(update={"address": f"127.0.0.1:{port}"})


@pytest.mark.parametrize("arrived", [False, True])
async def test_durable_worker_gate_through_rpc_and_scheduler(tmp_path: Path, arrived: bool) -> None:
    class Backend:
        stopped = False
        submissions = 0

        async def submit(self, engine_request_id: str, payload: Any) -> None:
            self.submissions += 1

        async def abort(self, engine_request_id: str) -> None:
            pass  # Signal delivery deliberately does not stop this backend fixture.

        async def query(self, engine_request_id: str) -> EngineObservation:
            return EngineObservation(
                engine_request_id=engine_request_id,
                observed_at=datetime.now(UTC),
                status=ExecutionStatus.ABORTED if self.stopped else ExecutionStatus.RUNNING,
                quiescent=self.stopped,
                submission_fenced=self.stopped,
            )

    ledger, scheduler, backend = RequestLedger(), setup(), Backend()
    route = await reserve(ledger, scheduler, request())
    gate = DurableExecutionDriver(
        tmp_path / "worker.db",
        backend,
        worker_id=route.worker_id,
        generation=route.worker_generation,
        engine_instance_id="fixture-engine",
        create=True,
    )
    server, endpoint = await start_runtime(gate)
    reconciler = RequestExecutionReconciler(
        ledger,
        LocalGrpcExecutionDriver(
            RequestExecutionConfig(mode="local-contract", endpoints={"worker-0": endpoint})
        ),
    )
    try:
        if arrived:
            await gate.admit(command(route), {})
        await ledger.release(
            route.decision_id, route.worker_id, route.worker_generation, "tenant-a", cancelled=True
        )
        assert not await reconciler.tick()
        if arrived:
            assert (await ledger.snapshot()).reservations[
                route.decision_id
            ].state is RequestState.CANCEL_REQUESTED
            with pytest.raises(NoEligibleWorker):
                await reserve(ledger, scheduler, request("blocked"))
            backend.stopped = True
            assert not await reconciler.tick()
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(route), {})
        assert (await ledger.snapshot()).reservations[
            route.decision_id
        ].state is RequestState.RELEASED
        assert backend.submissions == int(arrived)
        await reserve(ledger, scheduler, request("next"))
        released = [
            event
            for event in (await ledger.snapshot()).pending.values()
            if event.event_type == "lease.released"
        ]
        assert len(released) == 1
        assert released[0].payload["execution_receipt"]["admission_closed"]
    finally:
        await server.stop(0)
        gate.close()


async def test_loopback_query_abort_and_authentication() -> None:
    ledger, scheduler, driver = RequestLedger(), setup(), FixtureExecutionDriver()
    route = await reserve(ledger, scheduler, request())
    server, endpoint = await start_runtime(driver)
    client = LocalGrpcExecutionDriver(
        RequestExecutionConfig(mode="local-contract", endpoints={"worker-0": endpoint})
    )
    try:
        assert (await client.observe(command(route))).status is ExecutionStatus.RUNNING
        assert (
            await client.observe(command(route, ExecutionAction.ABORT))
        ).command.action is ExecutionAction.ABORT
        bad = LocalGrpcExecutionDriver(
            RequestExecutionConfig(
                mode="local-contract",
                endpoints={"worker-0": endpoint.model_copy(update={"token": SecretStr("z" * 32)})},
            )
        )
        with pytest.raises(grpc.aio.AioRpcError) as denied:
            await bad.observe(command(route))
        assert denied.value.code() is grpc.StatusCode.UNAUTHENTICATED
        with pytest.raises(grpc.aio.AioRpcError) as stale:
            await client.observe(command(route).model_copy(update={"engine_instance_id": "other"}))
        assert stale.value.code() is grpc.StatusCode.FAILED_PRECONDITION
        missing = LocalGrpcExecutionDriver(
            RequestExecutionConfig(mode="local-contract", endpoints={})
        )
        with pytest.raises(ValueError, match="endpoint_missing"):
            await missing.observe(command(route))
        driver.mismatch = True
        with pytest.raises(grpc.aio.AioRpcError):
            await client.observe(command(route))
    finally:
        await server.stop(0)


async def test_default_scheduler_maintenance_releases_only_after_worker_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unused_tcp_port: int
) -> None:
    scheduler, store, driver = setup(), InMemoryStore(), FixtureExecutionDriver()
    server, endpoint = await start_runtime(driver)
    config_path = tmp_path / "execution.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "local-contract",
                "endpoints": {
                    "worker-0": {
                        "address": endpoint.address,
                        "token": endpoint.token.get_secret_value(),
                    }
                },
            }
        )
    )
    monkeypatch.setenv("FREECHAT_REQUEST_EXECUTION_CONFIG", str(config_path))
    for name in ("FREECHAT_GROUP_RUNTIME_CONFIG", "ETCD_ENDPOINT", "NATS_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(grpc_server, "InMemoryStore", lambda: store)
    monkeypatch.setattr(grpc_server, "InMemoryWorkerRegistry", lambda: scheduler._registry)
    address = f"127.0.0.1:{unused_tcp_port}"
    task = asyncio.create_task(grpc_server.serve(address))
    client, ledger = GrpcSchedulerClient(address), RequestLedger(store)
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            await asyncio.wait_for(channel.channel_ready(), 5)
        req = request()
        route = await client.route(req)
        assert route.engine_instance_id == "fixture-engine"
        await client.release(req, route)
        assert (await ledger.snapshot()).reservations[
            route.decision_id
        ].state is RequestState.COMPLETION_PENDING
        with pytest.raises(grpc.aio.AioRpcError):
            await client.route(request("blocked"))
        driver.status, driver.quiet, driver.closed = ExecutionStatus.COMPLETED, True, True
        async with asyncio.timeout(4):
            while (await ledger.snapshot()).reservations[
                route.decision_id
            ].state is not RequestState.RELEASED:
                await asyncio.sleep(0.02)
        assert (await client.route(request("next"))).worker_id == "worker-0"
        state = await ledger.snapshot()
        assert any(
            event.event_type == "lease.released" and event.payload["execution_receipt"]
            for event in state.pending.values()
        )
    finally:
        await client.aclose()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await server.stop(0)


async def test_cancelled_request_restores_and_releases_only_on_abort_confirmation() -> None:
    store, scheduler = InMemoryStore(), setup()
    ledger = RequestLedger(store)
    route = await reserve(ledger, scheduler, request())
    await ledger.release(
        route.decision_id, route.worker_id, route.worker_generation, "tenant-a", cancelled=True
    )
    rebuilt, driver = RequestLedger(store), FixtureExecutionDriver()
    runtime = RequestExecutionReconciler(rebuilt, driver)
    assert await runtime.tick() == {}
    assert (await rebuilt.snapshot()).reservations[
        route.decision_id
    ].state is RequestState.CANCEL_REQUESTED
    first_operation = driver.commands[-1].operation_id
    driver.status, driver.quiet, driver.closed = ExecutionStatus.ABORTED, True, True
    assert await runtime.tick() == {}
    assert driver.commands[-1].action is ExecutionAction.ABORT
    assert driver.commands[-1].operation_id == first_operation
    assert (await rebuilt.snapshot()).reservations[route.decision_id].state is RequestState.RELEASED
    assert await runtime.tick() == {}
    assert len(driver.commands) == 2


async def test_full_outbox_cannot_commit_receipt_without_release_event() -> None:
    ledger, scheduler = RequestLedger(max_pending=1), setup()
    route = await reserve(ledger, scheduler, request())
    proof = receipt(route)
    with pytest.raises(ValueError, match="outbox_full"):
        await ledger.observe_execution(proof)
    state = await ledger.snapshot()
    assert state.reservations[route.decision_id].execution_receipt is None
    assert state.reservations[route.decision_id].state is RequestState.ACTIVE
    await ledger.acknowledge_event(next(iter(state.pending)))
    await ledger.observe_execution(proof)
    assert (await ledger.snapshot()).reservations[route.decision_id].state is RequestState.RELEASED


async def test_runtime_identity_validation_and_nonlocal_peer_rejection() -> None:
    driver = FixtureExecutionDriver()
    with pytest.raises(ValueError, match="identity_required"):
        LocalRequestExecutionService(
            driver, worker_id="w", generation=1, engine_instance_id="e", token=SecretStr("short")
        )
    service = LocalRequestExecutionService(
        driver, worker_id="w", generation=1, engine_instance_id="e", token=SecretStr("x" * 32)
    )

    class Context:
        remote = True

        def peer(self) -> str:
            return "ipv4:192.0.2.1:1" if self.remote else "ipv4:127.0.0.1:1"

        def invocation_metadata(self) -> list[tuple[str, str]]:
            return [("authorization", "Bearer " + "x" * 32)]

        async def abort(self, code: grpc.StatusCode, detail: str) -> None:
            raise PermissionError(code)

    context = Context()
    with pytest.raises(PermissionError) as denied:
        await service.Observe(control_pb2.RequestExecutionCommand(command_json="{}"), context)
    assert denied.value.args[0] is grpc.StatusCode.PERMISSION_DENIED
    context.remote = False
    with pytest.raises(PermissionError) as malformed:
        await service.Observe(control_pb2.RequestExecutionCommand(command_json="{}"), context)
    assert malformed.value.args[0] is grpc.StatusCode.FAILED_PRECONDITION
