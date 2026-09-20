import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from itertools import combinations

import pytest
from freechat_control_store import CompareFailed, InMemoryStore, KeyValue
from freechat_scheduler.parallel import DeviceLink
from freechat_scheduler.resource_groups import (
    TRANSITIONS,
    Device,
    GroupController,
    GroupSpec,
    GroupState,
    Inventory,
)
from hypothesis import given, settings
from hypothesis import strategies as st


def inventory() -> Inventory:
    # Synthetic capacity for local tests, NOT an assertion about the H100 machines.
    return Inventory(
        generation=1,
        devices=tuple(
            Device(
                gpu_id=f"node{node}/gpu{gpu}",
                node_id=f"node{node}",
                memory_bytes=80 * 1024**3,
                compute_capability="9.0",
            )
            for node in range(3)
            for gpu in range(4)
        ),
    )


def links() -> tuple[DeviceLink, ...]:
    return tuple(
        DeviceLink(f"node{node}/gpu{left}", f"node{node}/gpu{right}", True, True, True)
        for node in range(3)
        for left, right in combinations(range(4), 2)
    )


def spec(name: str = "group", *, node: int = 0, start: int = 0, tp: int = 4) -> GroupSpec:
    return GroupSpec(
        group_id=name,
        worker_id=name,
        gpu_ids=tuple(f"node{node}/gpu{gpu}" for gpu in range(start, start + tp)),
        model_id="test-model",
        model_revision="test-revision",
        tensor_parallel_size=tp,
        memory_per_gpu_bytes=1024,
        evidence_reference="SYNTHETIC_TEST_ONLY",
        evidence_expires_at=datetime.now(UTC) + timedelta(hours=1),
        communication_fraction=0.1,
        measured_speedup=1.2,
    )


@pytest.mark.parametrize("tp,count", [(1, 12), (2, 6), (4, 3)])
async def test_three_node_layouts_reserve_exactly_twelve_gpus(tp: int, count: int) -> None:
    controller = GroupController(InMemoryStore(), inventory(), links())
    for node in range(3):
        for start in range(0, 4, tp):
            await controller.reserve(
                spec(f"g{node}-{start}", node=node, start=start, tp=tp), "owner"
            )
    ledger = await controller.snapshot()
    assert len(ledger.groups) == count
    assert len({gpu for lease in ledger.groups.values() for gpu in lease.spec.gpu_ids}) == 12


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("memory_unknown", "unmeasured"),
        ("memory_small", "insufficient"),
        ("unhealthy", "unhealthy"),
        ("cross_node", "cross_node"),
        ("unknown", "unknown_gpu"),
        ("evidence", "evidence_unavailable"),
        ("expired", "evidence_unavailable"),
        ("disconnected", "topology"),
        ("heterogeneous", "heterogeneous"),
    ],
)
async def test_admission_failures_leave_no_partial_reservation(fault: str, reason: str) -> None:
    inv, proposal, edges = inventory(), spec(), links()
    if fault in {"memory_unknown", "memory_small", "unhealthy", "heterogeneous"}:
        updates: dict[str, dict[str, object]] = {
            "memory_unknown": {"memory_bytes": None},
            "memory_small": {"memory_bytes": 1},
            "unhealthy": {"healthy": False},
            "heterogeneous": {"compute_capability": "8.6"},
        }
        inv = inv.model_copy(
            update={"devices": (inv.devices[0].model_copy(update=updates[fault]), *inv.devices[1:])}
        )
    elif fault == "cross_node":
        proposal = proposal.model_copy(
            update={"gpu_ids": ("node0/gpu0", "node1/gpu0", "node0/gpu2", "node0/gpu3")}
        )
    elif fault == "unknown":
        proposal = proposal.model_copy(update={"gpu_ids": ("missing",)})
    elif fault == "evidence":
        proposal = proposal.model_copy(update={"evidence_reference": None})
    elif fault == "expired":
        proposal = proposal.model_copy(update={"evidence_expires_at": datetime.now(UTC)})
    else:
        edges = ()
    controller = GroupController(InMemoryStore(), inv, edges)
    with pytest.raises(ValueError, match=reason):
        await controller.reserve(proposal, "owner")
    assert not (await controller.snapshot()).groups


async def test_expiration_quarantines_without_reassigning_live_gpus() -> None:
    now = datetime.now(UTC)
    clock = [now]
    store = InMemoryStore()
    controller = GroupController(store, inventory(), links(), clock=lambda: clock[0])
    proposal = spec()
    lease = await controller.reserve(proposal, "owner", 1)
    await controller.transition("group", lease.generation, "owner", GroupState.STARTING)
    await controller.transition(
        "group",
        lease.generation,
        "owner",
        GroupState.READY,
        engine_instance_id="engine",
        acknowledgement="ready",
    )
    clock[0] += timedelta(seconds=2)
    assert (await controller.quarantine_expired()).groups["group"].state is GroupState.QUARANTINED
    with pytest.raises(ValueError, match="resources_busy"):
        await controller.reserve(spec("other"), "other")
    with pytest.raises(ValueError, match="not_renewable"):
        await controller.renew("group", lease.generation, "owner")
    await controller.transition("group", lease.generation, "owner", GroupState.STOPPING)
    with pytest.raises(ValueError, match="quiescence"):
        await controller.transition("group", lease.generation, "owner", GroupState.RELEASED)
    await controller.transition(
        "group",
        lease.generation,
        "owner",
        GroupState.RELEASED,
        acknowledgement="stopped-and-reconciled",
    )
    restarted = GroupController(store, inventory(), links(), clock=lambda: clock[0])
    with pytest.raises(ValueError, match="previous_generation"):
        await restarted.reserve(proposal, "owner")
    new = await restarted.reserve(proposal, "owner", expected_previous_generation=lease.generation)
    with pytest.raises(ValueError, match="previous_generation"):
        await restarted.reserve(proposal, "owner")
    assert new.generation > lease.generation
    with pytest.raises(ValueError, match="fence"):
        await restarted.transition("group", lease.generation, "owner", GroupState.STARTING)


class RacingStore(InMemoryStore):
    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        await asyncio.sleep(0)
        return await super().compare_and_put(key, expected_revision, value)


async def test_competing_controllers_cannot_double_allocate() -> None:
    store = RacingStore()
    controllers = [GroupController(store, inventory(), links()) for _ in range(12)]
    results = await asyncio.gather(
        *(
            controller.reserve(spec(f"group{i}"), f"owner{i}")
            for i, controller in enumerate(controllers)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert len((await controllers[0].snapshot()).groups) == 1


async def test_idempotency_conflict_renew_and_snapshot_isolation() -> None:
    controller = GroupController(InMemoryStore(), inventory(), links())
    proposal = spec()
    first = await controller.reserve(proposal, "owner")
    assert await controller.reserve(proposal, "owner") == first
    with pytest.raises(ValueError, match="conflict"):
        await controller.reserve(proposal, "different")
    with pytest.raises(ValueError, match="fence"):
        await controller.renew("group", first.generation, "different")
    assert (
        await controller.renew("group", first.generation, "owner", 300)
    ).expires_at > first.expires_at
    snapshot = await controller.snapshot()
    snapshot.groups.clear()
    assert len((await controller.snapshot()).groups) == 1


async def test_changed_topology_requires_explicit_reconciliation() -> None:
    store = InMemoryStore()
    await GroupController(store, inventory(), links()).reserve(spec(), "owner")
    changed = inventory().model_copy(update={"generation": 2})
    with pytest.raises(ValueError, match="reconciliation"):
        await GroupController(store, changed, links()).snapshot()


@pytest.mark.parametrize("target", [GroupState.READY, GroupState.DRAINING, GroupState.RELEASED])
async def test_illegal_shortcuts_rejected(target: GroupState) -> None:
    controller = GroupController(InMemoryStore(), inventory(), links())
    lease = await controller.reserve(spec(), "owner")
    with pytest.raises(ValueError, match="transition"):
        await controller.transition("group", lease.generation, "owner", target)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.sampled_from([1, 2, 4]), min_size=1, max_size=16))
def test_property_overlapping_requests_never_partially_allocate(sizes: list[int]) -> None:
    async def exercise() -> None:
        controller = GroupController(InMemoryStore(), inventory(), links())
        for index, size in enumerate(sizes):
            with suppress(ValueError):
                await controller.reserve(spec(f"g{index}", node=index % 3, tp=size), "owner")
            ledger = await controller.snapshot()
            gpus = [gpu for lease in ledger.groups.values() for gpu in lease.spec.gpu_ids]
            assert len(gpus) == len(set(gpus)) <= 12
            assert all(
                len(lease.spec.gpu_ids) == lease.spec.tensor_parallel_size
                for lease in ledger.groups.values()
            )

    asyncio.run(exercise())


class FailingStore(InMemoryStore):
    fail = True

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        if self.fail:
            raise OSError("store unavailable")
        return await super().compare_and_put(key, expected_revision, value)


async def test_store_failure_does_not_update_memory_state() -> None:
    store = FailingStore()
    controller = GroupController(store, inventory(), links())
    with pytest.raises(OSError):
        await controller.reserve(spec(), "owner")
    assert not (await controller.snapshot()).groups
    store.fail = False
    assert (await controller.reserve(spec(), "owner")).generation == 1


class ContendedStore(InMemoryStore):
    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        raise CompareFailed("forced contention")


async def test_contention_is_bounded() -> None:
    controller = GroupController(ContendedStore(), inventory(), links())
    with pytest.raises(RuntimeError, match="exhausted"):
        await controller.reserve(spec(), "owner")


def test_release_is_terminal() -> None:
    assert not TRANSITIONS[GroupState.RELEASED]


async def test_engine_identity_cannot_change_within_allocation_generation() -> None:
    controller = GroupController(InMemoryStore(), inventory(), links())
    lease = await controller.reserve(spec(), "owner")
    await controller.transition("group", lease.generation, "owner", GroupState.STARTING)
    ready = await controller.transition(
        "group",
        lease.generation,
        "owner",
        GroupState.READY,
        engine_instance_id="engine-a",
        acknowledgement="fixture-ready",
    )
    for target in (GroupState.READY, GroupState.DRAINING):
        with pytest.raises(ValueError, match="engine_instance_fence"):
            await controller.transition(
                "group",
                lease.generation,
                "owner",
                target,
                engine_instance_id="engine-b",
                acknowledgement="fixture",
            )
    assert (
        await controller.transition(
            "group", lease.generation, "owner", GroupState.READY, engine_instance_id="engine-a"
        )
        == ready
    )
    assert (await controller.snapshot()).groups["group"] == ready


@pytest.mark.parametrize(
    "patch",
    [
        {"tensor_parallel_size": 3},
        {"gpu_ids": ["dup"] * 4},
        {"gpu_ids": ["node0/gpu0"]},
        {"evidence_expires_at": "2026-09-14T10:00:00"},
    ],
)
def test_invalid_group_schema(patch: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GroupSpec.model_validate({**spec().model_dump(), **patch})


def test_duplicate_inventory_devices() -> None:
    with pytest.raises(ValueError):
        Inventory(generation=1, devices=(inventory().devices[0], inventory().devices[0]))


@pytest.mark.parametrize("ttl", [0, -1, 3601])
async def test_invalid_lease_ttl(ttl: int) -> None:
    controller = GroupController(InMemoryStore(), inventory(), links())
    with pytest.raises(ValueError):
        await controller.reserve(spec(), "owner", ttl)
    with pytest.raises(ValueError):
        await controller.renew("missing", 1, "owner", ttl)


async def test_readiness_expiry_and_idempotent_state_ack() -> None:
    clock = [datetime.now(UTC)]
    controller = GroupController(InMemoryStore(), inventory(), links(), clock=lambda: clock[0])
    with pytest.raises(ValueError, match="previous_generation"):
        await controller.reserve(spec(), "owner", expected_previous_generation=99)
    lease = await controller.reserve(spec(), "owner", 1)
    with pytest.raises(ValueError):
        await controller.reserve(spec(), " ")
    await controller.transition("group", lease.generation, "owner", GroupState.STARTING)
    with pytest.raises(ValueError, match="readiness"):
        await controller.transition("group", lease.generation, "owner", GroupState.READY)
    assert (
        await controller.transition("group", lease.generation, "owner", GroupState.STARTING)
    ).state is GroupState.STARTING
    clock[0] += timedelta(seconds=2)
    with pytest.raises(ValueError, match="expired"):
        await controller.transition(
            "group",
            lease.generation,
            "owner",
            GroupState.READY,
            engine_instance_id="engine",
            acknowledgement="ready",
        )


@pytest.mark.parametrize(
    "origin,target", [(left, right) for left in GroupState for right in GroupState]
)
async def test_complete_lifecycle_transition_matrix(origin: GroupState, target: GroupState) -> None:
    store = InMemoryStore()
    controller = GroupController(store, inventory(), links())
    lease = await controller.reserve(spec(), "owner")
    ledger = await controller.snapshot()
    # Directly seed each state; this is state-machine validation, not engine evidence.
    ledger.groups["group"] = lease.model_copy(update={"state": origin})
    await store.put(controller.key, ledger.model_dump_json().encode())
    if target is origin or target in TRANSITIONS[origin]:
        result = await controller.transition(
            "group",
            lease.generation,
            "owner",
            target,
            engine_instance_id="engine",
            acknowledgement="fixture",
        )
        assert result.state is target
    else:
        with pytest.raises(ValueError, match="transition"):
            await controller.transition(
                "group",
                lease.generation,
                "owner",
                target,
                engine_instance_id="engine",
                acknowledgement="fixture",
            )
