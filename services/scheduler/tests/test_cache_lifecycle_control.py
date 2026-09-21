"""Real loopback gRPC with a fake cache driver: transport evidence, not GPU evidence."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import grpc
import pytest
from freechat.control.cache_lifecycle import LocalCacheLifecycleService
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts.cache_lifecycle import (
    CacheLifecycleCommand,
    CacheLifecycleReceipt,
    CacheLifecycleUpdate,
    PrefixLifecycleReceipt,
)
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler.cache_lifecycle import CacheLifecycleController
from freechat_scheduler.grpc_server import SchedulerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_ledger import RequestLedger
from freechat_scheduler.scheduler import Scheduler
from pydantic import SecretStr
from test_native_preparation import prepared, request
from test_scheduler import add_worker

TOKEN = "test-control-token-" * 3


class RecordingCache:
    def __init__(self) -> None:
        self.commands: list[CacheLifecycleCommand] = []
        self.wrong_receipt = False

    async def apply(self, command: CacheLifecycleCommand) -> CacheLifecycleReceipt:
        self.commands.append(command)
        if self.wrong_receipt:
            command = command.model_copy(
                update={"owner": command.owner.model_copy(update={"tenant_id": "foreign"})}
            )
        return CacheLifecycleReceipt(
            command=command,
            observed_at=datetime.now(UTC),
            prefixes=(
                PrefixLifecycleReceipt(
                    request_id="internal",
                    sequence=command.update.sequence,
                    lifecycle=command.update.lifecycle,
                    resident_blocks=0,
                    protected_blocks=0,
                    status="not_resident",
                    applied_at_ms=100,
                    replayed=False,
                ),
            ),
        )


@pytest.fixture
async def runtime() -> AsyncIterator[Any]:
    driver = RecordingCache()
    server = grpc.aio.server()
    control_pb2_grpc.add_CacheLifecycleServiceServicer_to_server(  # type: ignore[no-untyped-call]
        LocalCacheLifecycleService(
            driver, identity=("worker", 1, "fixture-engine"), token=SecretStr(TOKEN)
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    worker = registry.snapshot()[1][0]
    registry.upsert(
        worker.capabilities.model_copy(update={"execution_endpoint": endpoint}),
        worker.telemetry,
    )
    scheduler = Scheduler(registry)
    ledger = RequestLedger()
    req = request()
    decision = await ledger.reserve(
        req,
        "once",
        lambda reserved: scheduler.route(
            req, prepared={"worker": prepared("worker", req)}, reserved=reserved
        ),
    )
    cache = CacheLifecycleController(ledger, registry, TOKEN)
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(scheduler, ledger, token=TOKEN, cache=cache),
        server,
    )
    await server.start()
    client = GrpcSchedulerClient(endpoint, token=TOKEN)
    update = CacheLifecycleUpdate(
        decision_id=decision.decision_id, sequence=1, lifecycle="tool_wait", expected_resume_ms=1000
    )
    try:
        yield client, cache, ledger, registry, driver, update, endpoint
    finally:
        await client.aclose()
        await server.stop(0)


async def test_update_roundtrip_does_not_mutate_reservations(runtime: Any) -> None:
    client, _, ledger, _, driver, update, _ = runtime
    before = await ledger.snapshot()
    receipt = await client.cache_lifecycle("tenant-a", update)
    assert receipt.command.update == update
    assert receipt.command.owner.tenant_id == "tenant-a"
    assert receipt.command.owner.engine_instance_id == "fixture-engine"
    assert driver.commands == [receipt.command]
    assert await ledger.snapshot() == before


@pytest.mark.parametrize("case", ["foreign_tenant", "unknown_decision", "replacement", "missing"])
async def test_owner_cannot_be_redirected(runtime: Any, case: str) -> None:
    client, _, ledger, registry, driver, update, _ = runtime
    tenant = "foreign" if case == "foreign_tenant" else "tenant-a"
    if case == "unknown_decision":
        update = update.model_copy(update={"decision_id": "unknown"})
    if case == "replacement":
        worker = registry.snapshot()[1][0]
        registry.upsert(
            worker.capabilities.model_copy(update={"generation": 2}),
            worker.telemetry.model_copy(update={"generation": 2, "engine_instance_id": "new"}),
        )
    if case == "missing":
        registry.remove("worker", 1)
    before = await ledger.snapshot()
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await client.cache_lifecycle(tenant, update)
    expected = (
        grpc.StatusCode.NOT_FOUND
        if case in {"foreign_tenant", "unknown_decision"}
        else (grpc.StatusCode.FAILED_PRECONDITION)
    )
    assert error.value.code() == expected
    assert driver.commands == []
    assert await ledger.snapshot() == before


async def test_scheduler_requires_control_authentication(runtime: Any) -> None:
    *_, endpoint = runtime
    client = GrpcSchedulerClient(endpoint, token="wrong")
    try:
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await client.cache_lifecycle("tenant-a", runtime[5])
        assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert runtime[4].commands == []
    finally:
        await client.aclose()


async def test_worker_rejects_wrong_receipt_binding(runtime: Any) -> None:
    client, _, _, _, driver, update, _ = runtime
    driver.wrong_receipt = True
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await client.cache_lifecycle("tenant-a", update)
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.parametrize("case", ["missing_token", "duplicate_token", "wrong_engine", "oversized"])
async def test_worker_control_boundary(runtime: Any, case: str) -> None:
    _, cache, _, _, driver, update, endpoint = runtime
    _, command = await cache._resolve_owner("tenant-a", update)
    metadata: tuple[tuple[str, str], ...] = (("authorization", f"Bearer {TOKEN}"),)
    if case == "missing_token":
        metadata = ()
    elif case == "duplicate_token":
        metadata = metadata + metadata
    elif case == "wrong_engine":
        command = command.model_copy(
            update={"owner": command.owner.model_copy(update={"engine_instance_id": "foreign"})}
        )
    payload = "x" * 16385 if case == "oversized" else command.model_dump_json()
    async with grpc.aio.insecure_channel(endpoint) as channel:
        stub = control_pb2_grpc.CacheLifecycleServiceStub(channel)  # type: ignore[no-untyped-call]
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.Apply(
                control_pb2.CacheLifecycleCommand(command_json=payload),
                metadata=metadata,
                timeout=2,
            )
    expected = (
        grpc.StatusCode.UNAUTHENTICATED
        if case.endswith("token")
        else (grpc.StatusCode.FAILED_PRECONDITION)
    )
    assert error.value.code() == expected
    assert driver.commands == []


async def test_cancelled_route_rejects_retain_without_changing_execution_state(
    runtime: Any,
) -> None:
    client, _, ledger, _, driver, update, _ = runtime
    await ledger.release(update.decision_id, "worker", 1, "tenant-a", cancelled=True)
    before = await ledger.snapshot()
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await client.cache_lifecycle("tenant-a", update)
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert driver.commands == []
    assert await ledger.snapshot() == before
