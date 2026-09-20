import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import grpc
import pytest
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import ModelCapability, RequestProfile, WorkerCapabilities, WorkerTelemetry
from freechat_control_store import InMemoryStore
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler import grpc_server
from freechat_scheduler.group_reconciler import (
    GroupReconciler,
    GroupRuntimeConfig,
    configured_reconciler,
)
from freechat_scheduler.group_runtime import (
    GroupAction,
    GroupCommand,
    GroupReceipt,
    LocalGroupRuntimeService,
    LocalGrpcGroupDriver,
    LocalRuntimeEndpoint,
    RankBudget,
    RuntimeStatus,
)
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.resource_groups import GroupController, GroupState
from freechat_scheduler.scheduler import NoEligibleWorker, Scheduler
from pydantic import SecretStr, ValidationError
from test_resource_groups import inventory, links, spec
from test_scheduler import profile


class FixtureDriver:
    """CPU contract observations only; never launches a model or asserts real H100 capacity."""

    def __init__(self) -> None:
        self.commands: list[GroupCommand] = []
        self.started: set[tuple[str, int]] = set()
        self.receipt_updates: dict[str, Any] = {}
        self.telemetry_updates: dict[str, Any] = {}
        self.capability_updates: dict[str, Any] = {}
        self.fail_after_start = False
        self.unavailable = False

    async def apply(self, command: GroupCommand) -> GroupReceipt:
        self.commands.append(command)
        if self.unavailable:
            raise ConnectionError("test runtime unavailable")
        now = datetime.now(UTC)
        group = command.spec
        instance = f"fixture-{group.group_id}-{command.generation}"
        if command.action is GroupAction.START:
            self.started.add((group.group_id, command.generation))
            if self.fail_after_start:
                self.fail_after_start = False
                raise TimeoutError("side effect succeeded but acknowledgement lost")
        status = {
            GroupAction.START: RuntimeStatus.READY,
            GroupAction.INSPECT: RuntimeStatus.READY,
            GroupAction.DRAIN: RuntimeStatus.DRAINED,
            GroupAction.STOP: RuntimeStatus.STOPPED,
        }[command.action]
        caps = WorkerCapabilities(
            worker_id=group.worker_id,
            generation=command.generation,
            endpoint="http://127.0.0.1:9/non-serving-fixture",
            node_id=command.node_id,
            gpu_id=group.gpu_ids[0],
            gpu_ids=group.gpu_ids,
            gpu_name="CPU fixture only",
            compute_capability="9.0",
            total_vram_bytes=80 * 1024**3,
            p2p_domain=command.node_id,
            network_domain="unknown",
            resource_group_id=group.group_id,
            resource_group_generation=command.generation,
            models=(
                ModelCapability(
                    model_id=group.model_id,
                    revision=group.model_revision,
                    tokenizer_revision="fixture",
                    architecture="dense",
                    attention="gqa",
                    max_context_tokens=4096,
                    dtype="bfloat16",
                    tensor_parallel_size=group.tensor_parallel_size,
                    kv_admission_bytes_per_token_per_rank=1024,
                    kv_block_size_tokens=16,
                ),
            ),
        ).model_copy(update=self.capability_updates)
        telemetry = WorkerTelemetry(
            worker_id=group.worker_id,
            generation=command.generation,
            free_vram_bytes=0,
            engine_instance_id=instance,
            kv_admission_available_bytes_per_rank=1024**3,
            observed_at=now,
        ).model_copy(update=self.telemetry_updates)
        return GroupReceipt(
            operation_id=command.operation_id,
            node_id=command.node_id,
            owner_id=command.owner_id,
            group_id=group.group_id,
            generation=command.generation,
            engine_instance_id=instance,
            observed_at=now,
            status=status,
            quiescent=status in {RuntimeStatus.DRAINED, RuntimeStatus.STOPPED},
            capabilities=caps,
            telemetry=telemetry,
            rank_budgets=tuple(
                RankBudget(gpu_id=gpu, available_bytes=1024**3) for gpu in group.gpu_ids
            ),
        ).model_copy(update=self.receipt_updates)


async def setup() -> tuple[GroupController, GroupReconciler, FixtureDriver, Scheduler]:
    controller = GroupController(InMemoryStore(), inventory(), links())
    await controller.reserve(spec(), "owner")
    driver, registry = FixtureDriver(), InMemoryWorkerRegistry()
    reconciler = GroupReconciler(controller, registry, driver, owner_id="owner")
    return controller, reconciler, driver, Scheduler(registry, group_snapshot=reconciler.snapshot)


def request() -> RequestProfile:
    return profile(model_id="test-model", input_tokens=16, output_tokens=16, local_node_id="node0")


async def test_readiness_route_drain_stop_and_release() -> None:
    controller, reconciler, driver, scheduler = await setup()
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())
    assert await reconciler.tick() == {}
    lease = (await controller.snapshot()).groups["group"]
    assert lease.state is GroupState.READY
    assert scheduler.route(request()).worker_id == "group"
    await reconciler.tick()
    assert driver.commands[-1].action is GroupAction.INSPECT
    await controller.transition("group", lease.generation, "owner", GroupState.DRAINING)
    await reconciler.tick()
    assert (await controller.snapshot()).groups["group"].state is GroupState.STOPPING
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())
    await reconciler.tick()
    assert (await controller.snapshot()).groups["group"].state is GroupState.RELEASED
    await controller.reserve(spec(), "owner", expected_previous_generation=lease.generation)
    await reconciler.tick()
    assert scheduler.route(request()).worker_generation > lease.generation
    assert len(driver.started) == 2


async def test_ack_loss_and_reconstruction_repeat_same_command_not_new_instance() -> None:
    controller, reconciler, driver, scheduler = await setup()
    driver.fail_after_start = True
    assert await reconciler.tick() == {"group": "TimeoutError"}
    assert (await controller.snapshot()).groups["group"].state is GroupState.STARTING
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())
    rebuilt = GroupReconciler(controller, reconciler.registry, driver, owner_id="owner")
    await rebuilt.tick()
    assert driver.commands[0].operation_id == driver.commands[1].operation_id
    assert len(driver.started) == 1
    assert (await controller.snapshot()).groups["group"].state is GroupState.READY


@pytest.mark.parametrize(
    "updates",
    [
        {"operation_id": "forged"},
        {"owner_id": "other"},
        {"node_id": "other"},
        {"group_id": "other"},
        {"generation": 9},
        {"status": RuntimeStatus.UNKNOWN},
        {"observed_at": datetime.now(UTC) - timedelta(minutes=1)},
        {"observed_at": datetime.now(UTC) + timedelta(minutes=1)},
        {"observed_at": datetime(2026, 1, 1)},
        {"capabilities": None},
        {"telemetry": None},
        {"quiescent": True},
        {"rank_budgets": ()},
        {"rank_budgets": (RankBudget(gpu_id="node0/gpu0", available_bytes=0),) * 4},
    ],
)
async def test_invalid_ready_receipt_never_routes(updates: dict[str, Any]) -> None:
    controller, reconciler, driver, scheduler = await setup()
    driver.receipt_updates = updates
    assert await reconciler.tick() == {"group": "ValueError"}
    assert (await controller.snapshot()).groups["group"].state is GroupState.STARTING
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())


@pytest.mark.parametrize(
    "updates",
    [
        {"kv_admission_available_bytes_per_rank": None},
        {"kv_admission_available_bytes_per_rank": 1024**3 + 1},
        {"engine_instance_id": "wrong"},
        {"healthy": False},
        {"draining": True},
        {"observed_at": datetime.now(UTC) - timedelta(minutes=1)},
    ],
)
async def test_invalid_worker_budget_or_identity_is_rejected(updates: dict[str, Any]) -> None:
    _, reconciler, driver, _ = await setup()
    driver.telemetry_updates = updates
    assert await reconciler.tick() == {"group": "ValueError"}


async def test_runtime_outage_hides_ready_group_without_freeing_it() -> None:
    controller, reconciler, driver, scheduler = await setup()
    await reconciler.tick()
    driver.unavailable = True
    assert await reconciler.tick() == {"group": "ConnectionError"}
    assert (await controller.snapshot()).groups["group"].state is GroupState.READY
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())
    with pytest.raises(ValueError, match="resources_busy"):
        await controller.reserve(spec("overlap"), "owner")
    driver.unavailable = False
    await reconciler.tick()
    assert scheduler.route(request()).worker_id == "group"


async def test_expiry_stops_before_release_and_unknown_is_not_quiescence() -> None:
    controller, reconciler, driver, _ = await setup()
    await reconciler.tick()
    controller.clock = lambda: datetime.now(UTC) + timedelta(minutes=2)
    driver.receipt_updates = {"status": RuntimeStatus.UNKNOWN, "quiescent": False}
    await reconciler.tick()
    assert driver.commands[-1].action is GroupAction.STOP
    assert (await controller.snapshot()).groups["group"].state is GroupState.STOPPING
    driver.receipt_updates = {"status": RuntimeStatus.STOPPED, "quiescent": False}
    await reconciler.tick()
    assert (await controller.snapshot()).groups["group"].state is GroupState.STOPPING
    driver.receipt_updates = {}
    await reconciler.tick()
    assert (await controller.snapshot()).groups["group"].state is GroupState.RELEASED


async def test_ownership_instance_fence_snapshot_copy_and_cancel() -> None:
    controller, reconciler, driver, _ = await setup()
    await controller.reserve(spec("foreign", node=1), "another-owner")
    await reconciler.tick()
    assert len(driver.started) == 1
    assert "foreign" not in reconciler.snapshot()[1].groups
    reconciler.snapshot()[1].groups.clear()
    assert reconciler.snapshot()[1].groups
    driver.receipt_updates = {"engine_instance_id": "different-instance"}
    assert await reconciler.tick() == {"group": "ValueError"}
    task = asyncio.create_task(reconciler.run())
    await asyncio.sleep(0)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def test_real_loopback_runtime_transport_and_identity_denial() -> None:
    controller, reconciler, driver, scheduler = await setup()
    server = grpc.aio.server()
    secret = SecretStr("local-test-token-" * 3)
    control_pb2_grpc.add_GroupRuntimeServiceServicer_to_server(  # type: ignore[no-untyped-call]
        LocalGroupRuntimeService(
            driver, node_id="node0", owner_id="owner", token=secret, store=InMemoryStore()
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    endpoint = LocalRuntimeEndpoint(address=f"127.0.0.1:{port}", token=secret)
    reconciler.driver = LocalGrpcGroupDriver({"node0": endpoint})
    try:
        assert await reconciler.tick() == {}
        assert scheduler.route(request()).worker_id == "group"
        command = GroupCommand.for_lease(
            (await controller.snapshot()).groups["group"], GroupAction.INSPECT
        )
        bad = LocalGrpcGroupDriver(
            {"node0": endpoint.model_copy(update={"token": SecretStr("bad" * 12)})}
        )
        with pytest.raises(grpc.aio.AioRpcError) as denied:
            await bad.apply(command)
        assert denied.value.code() is grpc.StatusCode.UNAUTHENTICATED
        wrong = command.model_copy(update={"owner_id": "other"})
        wrong = wrong.model_copy(update={"operation_id": wrong.identity()})
        with pytest.raises(grpc.aio.AioRpcError) as fenced:
            await reconciler.driver.apply(wrong)
        assert fenced.value.code() is grpc.StatusCode.FAILED_PRECONDITION
        async with grpc.aio.insecure_channel(endpoint.address) as channel:
            stub = control_pb2_grpc.GroupRuntimeServiceStub(channel)  # type: ignore[no-untyped-call]
            with pytest.raises(grpc.aio.AioRpcError):
                await stub.Apply(control_pb2.GroupRuntimeCommand(command_json="{}"))
    finally:
        await server.stop(0)


async def test_configuration_is_explicit_and_local_only() -> None:
    with pytest.raises(ValidationError):
        LocalRuntimeEndpoint(address="remote:50051", token=SecretStr("x" * 32))
    with pytest.raises(ValueError):
        LocalGroupRuntimeService(
            FixtureDriver(),
            node_id="n",
            owner_id="o",
            token=SecretStr("short"),
            store=InMemoryStore(),
        )
    config = GroupRuntimeConfig(
        mode="local-contract", owner_id="owner", inventory=inventory(), endpoints={}
    )
    runtime = configured_reconciler(config, InMemoryStore(), InMemoryWorkerRegistry())
    lease = await runtime.controller.reserve(spec(tp=1), "owner")
    with pytest.raises(ValueError, match="endpoint_missing"):
        await runtime.driver.apply(GroupCommand.for_lease(lease, GroupAction.START))
    assert "local-test-token" not in repr(config)


async def test_node_fences_survive_reconstruction_and_reject_delayed_commands() -> None:
    controller, _, driver, _ = await setup()
    store = InMemoryStore()

    def service() -> LocalGroupRuntimeService:
        return LocalGroupRuntimeService(
            driver, node_id="node0", owner_id="owner", token=SecretStr("x" * 32), store=store
        )

    lease = (await controller.snapshot()).groups["group"]
    start = GroupCommand.for_lease(lease, GroupAction.START)
    driver.fail_after_start = True
    with pytest.raises(TimeoutError):
        await service()._apply_fenced(start)
    await service()._apply_fenced(start)
    assert len(driver.started) == 1
    newer = GroupCommand.for_lease(lease.model_copy(update={"generation": 2}), GroupAction.START)
    with pytest.raises(ValueError, match="not_stopped"):
        await service()._apply_fenced(newer)
    overlap = lease.model_copy(update={"spec": spec("overlap")})
    with pytest.raises(ValueError, match="gpus_busy"):
        await service()._apply_fenced(GroupCommand.for_lease(overlap, GroupAction.START))
    stop = GroupCommand.for_lease(lease, GroupAction.STOP)
    await service()._apply_fenced(stop)
    with pytest.raises(ValueError, match="phase_conflict"):
        await service()._apply_fenced(start)
    await service()._apply_fenced(newer)
    with pytest.raises(ValueError, match="generation_regressed"):
        await service()._apply_fenced(stop)
    assert len(driver.started) == 2


async def test_default_serve_uses_configured_group_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unused_tcp_port: int
) -> None:
    controller, _, driver, _ = await setup()
    node = grpc.aio.server()
    token = "only-a-local-test-fixture-token-123456789"
    control_pb2_grpc.add_GroupRuntimeServiceServicer_to_server(  # type: ignore[no-untyped-call]
        LocalGroupRuntimeService(
            driver, node_id="node0", owner_id="owner", token=SecretStr(token), store=InMemoryStore()
        ),
        node,
    )
    port = node.add_insecure_port("127.0.0.1:0")
    await node.start()
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "local-contract",
                "owner_id": "owner",
                "inventory": inventory().model_dump(mode="json"),
                "endpoints": {"node0": {"address": f"127.0.0.1:{port}", "token": token}},
            }
        )
    )
    monkeypatch.setenv("FREECHAT_GROUP_RUNTIME_CONFIG", str(config_path))
    monkeypatch.delenv("ETCD_ENDPOINT", raising=False)
    monkeypatch.delenv("NATS_URL", raising=False)
    monkeypatch.setattr(grpc_server, "InMemoryStore", lambda: controller.store)
    address = f"127.0.0.1:{unused_tcp_port}"
    task = asyncio.create_task(grpc_server.serve(address))
    client = GrpcSchedulerClient(address)
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            await asyncio.wait_for(channel.channel_ready(), timeout=5)
        decision = await client.route(request())
        assert decision.worker_id == "group"
        assert decision.reserved_kv_bytes_per_rank == 32768
        await client.release(request(), decision)
    finally:
        await client.aclose()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await node.stop(0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_id", "other"),
        ("resource_group_id", None),
        ("generation", 20),
        ("gpu_ids", ("node1/gpu0",)),
        ("models", ()),
    ],
)
async def test_capability_binding_is_checked(field: str, value: Any) -> None:
    _, reconciler, driver, _ = await setup()
    driver.capability_updates = {field: value}
    assert await reconciler.tick() == {"group": "ValueError"}


async def test_drain_waits_and_budget_zero_is_valid_but_not_admissible() -> None:
    controller, reconciler, driver, scheduler = await setup()
    driver.telemetry_updates = {"kv_admission_available_bytes_per_rank": 0}
    driver.receipt_updates = {
        "rank_budgets": tuple(RankBudget(gpu_id=gpu, available_bytes=0) for gpu in spec().gpu_ids)
    }
    assert await reconciler.tick() == {}
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())
    lease = (await controller.snapshot()).groups["group"]
    await controller.transition("group", lease.generation, "owner", GroupState.DRAINING)
    driver.receipt_updates = {"status": RuntimeStatus.PENDING, "quiescent": False}
    await reconciler.tick()
    assert (await controller.snapshot()).groups["group"].state is GroupState.DRAINING


async def test_node_unknown_stale_and_changed_instance_do_not_release() -> None:
    controller, _, driver, _ = await setup()
    service = LocalGroupRuntimeService(
        driver, node_id="node0", owner_id="owner", token=SecretStr("x" * 32), store=InMemoryStore()
    )
    lease = (await controller.snapshot()).groups["group"]
    with pytest.raises(ValueError, match="incarnation_unknown"):
        await service._apply_fenced(GroupCommand.for_lease(lease, GroupAction.STOP))
    await service._apply_fenced(GroupCommand.for_lease(lease, GroupAction.START))
    stop = GroupCommand.for_lease(lease, GroupAction.STOP)
    driver.receipt_updates = {"observed_at": datetime.now(UTC) - timedelta(minutes=1)}
    with pytest.raises(ValueError, match="observation_stale"):
        await service._apply_fenced(stop)
    driver.receipt_updates = {"engine_instance_id": "surprise-replacement"}
    with pytest.raises(ValueError, match="instance_changed"):
        await service._apply_fenced(stop)


async def test_reconcile_loop_invalidates_view_on_store_or_clock_failure() -> None:
    _, reconciler, _, scheduler = await setup()
    await reconciler.tick()
    reconciler.clock = lambda: datetime(2026, 1, 1)
    with pytest.raises(ValueError, match="timezone"):
        await reconciler.tick()
    task = asyncio.create_task(reconciler.run())
    await asyncio.sleep(0.01)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    with pytest.raises(NoEligibleWorker):
        scheduler.route(request())


async def test_runtime_service_rejects_nonlocal_peer_and_malformed_command() -> None:
    service = LocalGroupRuntimeService(
        FixtureDriver(), node_id="n", owner_id="o", token=SecretStr("x" * 32), store=InMemoryStore()
    )

    class Context:
        remote = True

        def peer(self) -> str:
            return "ipv4:192.0.2.1:5" if self.remote else "ipv4:127.0.0.1:5"

        def invocation_metadata(self) -> list[tuple[str, str]]:
            return [("authorization", "Bearer " + "x" * 32)]

        async def abort(self, code: grpc.StatusCode, detail: str) -> None:
            raise PermissionError(code)

    context = Context()
    with pytest.raises(PermissionError) as remote:
        await service.Apply(control_pb2.GroupRuntimeCommand(command_json="{}"), context)
    assert remote.value.args[0] is grpc.StatusCode.PERMISSION_DENIED
    context.remote = False
    with pytest.raises(PermissionError) as invalid:
        await service.Apply(control_pb2.GroupRuntimeCommand(command_json="{}"), context)
    assert invalid.value.args[0] is grpc.StatusCode.FAILED_PRECONDITION


async def test_slow_group_does_not_block_other_groups_and_times_out_without_release() -> None:
    controller, reconciler, _, scheduler = await setup()
    await controller.reserve(spec("fast", node=1), "owner")

    class SlowDriver(FixtureDriver):
        async def apply(self, command: GroupCommand) -> GroupReceipt:
            if command.spec.group_id == "group":
                await asyncio.Event().wait()
            return await super().apply(command)

    driver = SlowDriver()
    reconciler.driver = driver
    tick = asyncio.create_task(reconciler.tick())
    try:
        async with asyncio.timeout(1):
            while not driver.started:
                await asyncio.sleep(0)
        assert not tick.done()
        assert await tick == {"group": "TimeoutError"}
    finally:
        tick.cancel()
        with suppress(asyncio.CancelledError):
            await tick
    assert scheduler.route(request()).worker_id == "fast"
    assert (await controller.snapshot()).groups["group"].state is GroupState.STARTING
