import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from freechat_contracts import RequestProfile, RouteDecision
from freechat_contracts.execution import (
    ExecutionAction,
    ExecutionCommand,
    ExecutionReceipt,
    ExecutionStatus,
)
from freechat_control_store import CompareFailed, InMemoryStore, KeyValue
from freechat_scheduler.grpc_server import SchedulerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_ledger import RequestLedger, RequestState
from freechat_scheduler.scheduler import NoEligibleWorker, RoutingStrategy, Scheduler
from freechat_trace_replay.bus import DurableLifecycleEmitter, LifecyclePublisher
from hypothesis import given, settings
from hypothesis import strategies as st
from test_scheduler import add_worker, profile

UNIT = 32 * 16_384


def setup(slots: int = 1, workers: int = 1) -> Scheduler:
    registry = InMemoryWorkerRegistry()
    for i in range(workers):
        add_worker(registry, f"worker-{i}", node="ross", free_vram_bytes=UNIT * slots)
    return Scheduler(registry, strategy=RoutingStrategy.LEAST_LOAD)


def request(identity: str = "request", tenant: str = "tenant-a") -> RequestProfile:
    return profile(request_id=identity, tenant_id=tenant, input_tokens=16, output_tokens=16)


async def reserve(
    ledger: RequestLedger, scheduler: Scheduler, req: RequestProfile
) -> RouteDecision:
    return await ledger.reserve(
        req, req.request_id, lambda held: scheduler.route(req, reserved=held)
    )


async def confirm_execution(ledger: RequestLedger, route: RouteDecision, tenant: str) -> None:
    assert route.engine_instance_id
    await ledger.observe_execution(
        ExecutionReceipt(
            command=ExecutionCommand(
                tenant_id=tenant,
                request_id=route.request_id,
                decision_id=route.decision_id,
                worker_id=route.worker_id,
                worker_generation=route.worker_generation,
                engine_instance_id=route.engine_instance_id,
                action=ExecutionAction.QUERY,
            ),
            observation_sequence=1,
            observed_at=ledger.clock(),
            status=ExecutionStatus.COMPLETED,
            quiescent=True,
            admission_closed=True,
        )
    )


class RacingStore(InMemoryStore):
    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        await asyncio.sleep(0)
        return await super().compare_and_put(key, expected_revision, value)


async def test_maintenance_expiry_runs_despite_publication_failure() -> None:
    clock = [datetime.now(UTC)]
    ledger, scheduler = RequestLedger(clock=lambda: clock[0]), setup()
    route = await reserve(ledger, scheduler, request())
    clock[0] += timedelta(seconds=31)

    class UnavailablePublisherService(SchedulerGrpcService):
        async def flush_events(self) -> None:
            raise ConnectionError("publisher unavailable")

    task = asyncio.create_task(UnavailablePublisherService(scheduler, ledger).maintain())
    try:
        async with asyncio.timeout(2):
            while (await ledger.snapshot()).reservations[
                route.decision_id
            ].state is RequestState.ACTIVE:
                await asyncio.sleep(0)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert len((await ledger.snapshot()).pending) == 2
    with pytest.raises(NoEligibleWorker):
        await reserve(ledger, scheduler, request("next"))


async def test_concurrent_controllers_cannot_overbook_snapshot_capacity() -> None:
    store, scheduler = RacingStore(), setup(workers=2)
    results = await asyncio.gather(
        *(reserve(RequestLedger(store), scheduler, request(str(i))) for i in range(12)),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 2
    assert sum(isinstance(result, NoEligibleWorker) for result in results) == 10
    state = await RequestLedger(store).snapshot()
    assert len(state.reservations) == len(state.pending) == 2
    assert {lease.decision.worker_id for lease in state.reservations.values()} == {
        "worker-0",
        "worker-1",
    }
    assert all(
        lease.decision.reserved_kv_bytes_per_rank == UNIT for lease in state.reservations.values()
    )


async def test_duplicate_reservation_and_release_do_not_repeat_state_or_event() -> None:
    ledger, scheduler, req = RequestLedger(RacingStore()), setup(), request()
    routes = await asyncio.gather(*(reserve(ledger, scheduler, req) for _ in range(8)))
    assert len({route.decision_id for route in routes}) == 1
    route = routes[0]
    with pytest.raises(ValueError, match="idempotency_conflict"):
        await reserve(ledger, scheduler, req.model_copy(update={"output_tokens": 17}))
    await asyncio.gather(
        *(
            ledger.release(
                route.decision_id, route.worker_id, route.worker_generation, req.tenant_id
            )
            for _ in range(8)
        )
    )
    state = await ledger.snapshot()
    assert len(state.pending) == 2
    assert state.reservations[route.decision_id].state is RequestState.COMPLETION_PENDING
    with pytest.raises(ValueError, match="not_active"):
        await reserve(ledger, scheduler, req)
    await confirm_execution(ledger, route, req.tenant_id)
    assert (await reserve(ledger, scheduler, request("next"))).worker_id == route.worker_id


async def test_tenant_worker_and_generation_fences() -> None:
    ledger, scheduler, req = RequestLedger(), setup(slots=2), request()
    route = await reserve(ledger, scheduler, req)
    with pytest.raises(KeyError):
        await ledger.require(route.decision_id, route.worker_id, route.worker_generation, "other")
    with pytest.raises(ValueError, match="fencing"):
        await ledger.release(
            route.decision_id, route.worker_id, route.worker_generation + 1, req.tenant_id
        )
    other = await reserve(ledger, scheduler, request(tenant="other"))
    assert other.decision_id != route.decision_id


async def test_renewal_expiry_cancellation_and_restart_hold_uncertain_capacity() -> None:
    clock = [datetime.now(UTC)]
    store = InMemoryStore()
    ledger, scheduler, req = RequestLedger(store, clock=lambda: clock[0]), setup(), request()
    route = await reserve(ledger, scheduler, req)
    original = (await ledger.snapshot()).reservations[route.decision_id].expires_at
    clock[0] += timedelta(seconds=10)
    await ledger.renew(
        route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, "renew-1"
    )
    extended = (await ledger.snapshot()).reservations[route.decision_id].expires_at
    assert extended == original + timedelta(seconds=10)
    clock[0] += timedelta(seconds=5)
    await ledger.renew(
        route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, "renew-1"
    )
    assert (await ledger.snapshot()).reservations[route.decision_id].expires_at == extended
    clock[0] = extended
    restored = RequestLedger(store, clock=lambda: clock[0])
    with pytest.raises(ValueError, match="not_renewable"):
        await restored.renew(
            route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, "renew-2"
        )
    await restored.expire()
    await restored.expire()
    assert (await restored.snapshot()).reservations[route.decision_id].state is RequestState.EXPIRED
    with pytest.raises(NoEligibleWorker):
        await reserve(restored, scheduler, request("blocked"))
    await restored.release(
        route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, cancelled=True
    )
    with pytest.raises(NoEligibleWorker):
        await reserve(restored, scheduler, request("still-blocked"))
    await restored.release(
        route.decision_id, route.worker_id, route.worker_generation, req.tenant_id
    )
    await confirm_execution(restored, route, req.tenant_id)
    await reserve(restored, scheduler, request("available"))


class FaultStore(InMemoryStore):
    failure: str | None = None

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        if key == "/freechat/request-ledger" and self.failure == "before":
            self.failure = None
            raise OSError("before commit")
        item = await super().compare_and_put(key, expected_revision, value)
        if key == "/freechat/request-ledger" and self.failure == "after":
            self.failure = None
            raise OSError("lost commit acknowledgement")
        return item


@pytest.mark.parametrize("failure", ["before", "after"])
@pytest.mark.parametrize("operation", ["reserve", "release", "renew"])
async def test_state_and_event_intent_commit_together(failure: str, operation: str) -> None:
    store = FaultStore()
    ledger, scheduler, req = RequestLedger(store), setup(), request()
    route = None if operation == "reserve" else await reserve(ledger, scheduler, req)

    async def perform() -> None:
        if route is None:
            await reserve(ledger, scheduler, req)
        elif operation == "release":
            await ledger.release(
                route.decision_id, route.worker_id, route.worker_generation, req.tenant_id
            )
        else:
            await ledger.renew(
                route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, "renew"
            )

    before = await ledger.snapshot()
    store.failure = failure
    with pytest.raises(OSError):
        await perform()
    after = await RequestLedger(store).snapshot()
    if failure == "before":
        assert after == before
    else:
        assert len(after.pending) == len(before.pending) + 1
        assert sum(lease.sequence for lease in after.reservations.values()) == (
            0 if operation == "reserve" else 1
        )
    await perform()
    final = await ledger.snapshot()
    assert len(final.reservations) == 1
    assert len(final.pending) == (1 if operation == "reserve" else 2)


class Publisher:
    def __init__(self, store: FaultStore) -> None:
        self.store = store
        self.fail_publish = False
        self.fail_ack = False
        self.ids: list[str] = []

    async def publish(
        self, subject: str, payload: bytes, *, headers: dict[str, str] | None = None
    ) -> Any:
        if self.fail_publish:
            raise OSError("bus unavailable")
        assert headers is not None
        self.ids.append(headers["Nats-Msg-Id"])
        if self.fail_ack:
            self.fail_ack = False
            self.store.failure = "before"
        return None


async def test_bus_outage_and_crash_after_publish_replay_stable_event_ids() -> None:
    store = FaultStore()
    ledger, scheduler = RequestLedger(store), setup()
    route = await reserve(ledger, scheduler, request())
    publisher = Publisher(store)
    emitter = DurableLifecycleEmitter(store, LifecyclePublisher(publisher))
    service = SchedulerGrpcService(scheduler, ledger, emitter)
    publisher.fail_publish = True
    with pytest.raises(OSError):
        await service.flush_events()
    assert len((await ledger.snapshot()).pending) == 1
    publisher.fail_publish = False
    publisher.fail_ack = True
    with pytest.raises(OSError):
        await service.flush_events()
    assert len((await ledger.snapshot()).pending) == 1
    await SchedulerGrpcService(scheduler, RequestLedger(store), emitter).flush_events()
    assert publisher.ids[0] == publisher.ids[1]
    assert not (await ledger.snapshot()).pending
    assert (await ledger.snapshot()).reservations[route.decision_id].state is RequestState.ACTIVE


async def test_full_outbox_does_not_commit_half_release_and_can_be_drained() -> None:
    ledger, scheduler, req = RequestLedger(max_pending=1), setup(), request()
    route = await reserve(ledger, scheduler, req)
    with pytest.raises(ValueError, match="outbox_full"):
        await ledger.release(
            route.decision_id, route.worker_id, route.worker_generation, req.tenant_id
        )
    state = await ledger.snapshot()
    assert state.reservations[route.decision_id].state is RequestState.ACTIVE
    await ledger.acknowledge_event(next(iter(state.pending)))
    await ledger.release(route.decision_id, route.worker_id, route.worker_generation, req.tenant_id)
    assert (await ledger.snapshot()).reservations[
        route.decision_id
    ].state is RequestState.COMPLETION_PENDING


@pytest.mark.parametrize("limit", ["max_records", "max_pending", "max_renewals", "max_bytes"])
def test_invalid_limits(limit: str) -> None:
    with pytest.raises(ValueError, match="limits"):
        RequestLedger(**{limit: 0})  # type: ignore[arg-type]


async def test_record_byte_and_renewal_history_limits_fail_closed() -> None:
    scheduler, req = setup(slots=2), request()
    tiny = RequestLedger(max_bytes=1)
    with pytest.raises(ValueError, match="bytes_exceeded"):
        await reserve(tiny, scheduler, req)
    assert not (await tiny.snapshot()).reservations
    ledger = RequestLedger(max_records=1, max_renewals=1)
    route = await reserve(ledger, scheduler, req)
    with pytest.raises(ValueError, match="ledger_full"):
        await reserve(ledger, scheduler, request("another"))
    await ledger.renew(
        route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, "one"
    )
    with pytest.raises(ValueError, match="history_full"):
        await ledger.renew(
            route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, "two"
        )
    with pytest.raises(ValueError, match="identity_required"):
        await ledger.renew(
            route.decision_id, route.worker_id, route.worker_generation, req.tenant_id, ""
        )


async def test_schema_old_records_and_clock_require_explicit_reconciliation() -> None:
    store = InMemoryStore()
    ledger = RequestLedger(store)
    await store.put(ledger.key, b'{"schema_version":9}')
    with pytest.raises(ValueError, match="schema"):
        await ledger.snapshot()
    old = InMemoryStore()
    await old.put("/freechat/leases/existing", b"unreconciled")
    with pytest.raises(ValueError, match="unreconciled"):
        await RequestLedger(old).snapshot()
    with pytest.raises(ValueError, match="timezone"):
        await reserve(RequestLedger(clock=lambda: datetime(2026, 9, 15)), setup(), request())


async def test_request_identity_and_invalid_planner_output() -> None:
    scheduler, ledger, req = setup(slots=2), RequestLedger(), request()
    with pytest.raises(ValueError, match="identity_required"):
        await ledger.reserve(req, "", lambda load: scheduler.route(req, reserved=load))
    with pytest.raises(ValueError, match="invalid_reservation_decision"):
        await ledger.reserve(
            req,
            "key",
            lambda load: scheduler.route(req, reserved=load).model_copy(
                update={"reserved_kv_bytes_per_rank": 0}
            ),
        )
    route = await reserve(ledger, scheduler, req)
    another = request("another")
    with pytest.raises(ValueError, match="identity_collision"):
        await ledger.reserve(
            another,
            "another",
            lambda _: route.model_copy(update={"request_id": another.request_id}),
        )
    snapshot = await ledger.snapshot()
    snapshot.reservations.clear()
    assert len((await ledger.snapshot()).reservations) == 1


class ContendedStore(InMemoryStore):
    calls = 0

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        self.calls += 1
        raise CompareFailed("contention")


async def test_contention_retries_are_bounded() -> None:
    store = ContendedStore()
    with pytest.raises(RuntimeError, match="retry_exhausted"):
        await reserve(RequestLedger(store), setup(), request())
    assert store.calls == 32


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=30))
def test_random_lifecycles_preserve_capacity_bound(operations: list[int]) -> None:
    async def scenario() -> None:
        clock = [datetime.now(UTC)]
        ledger = RequestLedger(clock=lambda: clock[0])
        scheduler = setup(slots=2, workers=2)
        for index, operation in enumerate(operations):
            state = await ledger.snapshot()
            held = [
                item
                for item in state.reservations.values()
                if item.state is not RequestState.RELEASED
            ]
            if operation == 0:
                with suppress(NoEligibleWorker):
                    await reserve(ledger, scheduler, request(str(index)))
            elif operation == 3:
                clock[0] += timedelta(seconds=31)
                await ledger.expire()
            elif held:
                lease = held[0]
                await ledger.release(
                    lease.decision.decision_id,
                    lease.decision.worker_id,
                    lease.decision.worker_generation,
                    lease.tenant_id,
                    cancelled=operation == 2,
                )
                if operation == 1:
                    await confirm_execution(ledger, lease.decision, lease.tenant_id)
            latest = await ledger.snapshot()
            for worker in ("worker-0", "worker-1"):
                used = sum(
                    item.decision.reserved_kv_bytes_per_rank
                    for item in latest.reservations.values()
                    if item.state is not RequestState.RELEASED and item.decision.worker_id == worker
                )
                assert 0 <= used <= 2 * UNIT

    asyncio.run(scenario())
