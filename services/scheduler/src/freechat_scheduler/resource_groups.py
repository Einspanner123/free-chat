"""Static GPU groups with a single-key CAS ledger and fail-closed lifecycle.

No process is launched here. A deployment reconciler must perform external actions
and report fenced acknowledgements; reservation expiry never proves GPU quiescence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from freechat_control_store import CompareFailed, KeyValueStore
from pydantic import BaseModel, ConfigDict, Field, model_validator

from freechat_scheduler.parallel import DeviceLink, ParallelCandidate, ParallelMode, ParallelPlanner


class Device(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gpu_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    memory_bytes: int | None = Field(default=None, gt=0)
    healthy: bool = True
    compute_capability: str = Field(min_length=1)


class Inventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    generation: int = Field(ge=1)
    devices: tuple[Device, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self) -> Inventory:
        if len({device.gpu_id for device in self.devices}) != len(self.devices):
            raise ValueError("GPU identities must be globally unique")
        return self


class GroupSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    group_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    gpu_ids: tuple[str, ...] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tensor_parallel_size: int = Field(ge=1)
    memory_per_gpu_bytes: int = Field(gt=0)
    evidence_reference: str | None = None
    evidence_expires_at: datetime | None = None
    communication_fraction: float | None = Field(default=None, ge=0, le=1)
    measured_speedup: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def shape(self) -> GroupSpec:
        if self.tensor_parallel_size not in {1, 2, 4}:
            raise ValueError("supported static TP layouts are 1, 2, 4")
        if (
            len(set(self.gpu_ids)) != len(self.gpu_ids)
            or len(self.gpu_ids) != self.tensor_parallel_size
        ):
            raise ValueError("TP requires exactly its number of distinct GPUs")
        if self.evidence_expires_at is not None and self.evidence_expires_at.tzinfo is None:
            raise ValueError("evidence expiry requires a timezone")
        return self


class GroupState(StrEnum):
    RESERVED = "reserved"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPING = "stopping"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    RELEASED = "released"


TRANSITIONS = {
    GroupState.RESERVED: {GroupState.STARTING, GroupState.STOPPING, GroupState.FAILED},
    GroupState.STARTING: {GroupState.READY, GroupState.STOPPING, GroupState.FAILED},
    GroupState.READY: {GroupState.DRAINING, GroupState.FAILED},
    GroupState.DRAINING: {GroupState.STOPPING, GroupState.FAILED},
    GroupState.STOPPING: {GroupState.RELEASED, GroupState.FAILED},
    GroupState.FAILED: {GroupState.STOPPING},
    GroupState.QUARANTINED: {GroupState.STOPPING},
    GroupState.RELEASED: set(),
}


class GroupLease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spec: GroupSpec
    owner_id: str
    node_id: str
    generation: int = Field(ge=1)
    previous_generation: int = Field(default=0, ge=0)
    state: GroupState = GroupState.RESERVED
    expires_at: datetime
    engine_instance_id: str | None = None
    acknowledgement: str | None = None


class GroupLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    inventory_hash: str
    next_generation: int = 1
    groups: dict[str, GroupLease] = Field(default_factory=dict)


class GroupController:
    """CAS across the complete 12-GPU allocation, shared by controller replicas.

    Single-key transactions favor correctness at this scale; this is not a claim
    of validated etcd throughput or HA. Hardware evidence is operator-supplied.
    """

    def __init__(
        self,
        store: KeyValueStore,
        inventory: Inventory,
        links: tuple[DeviceLink, ...] = (),
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        key: str = "/freechat/resource-groups/static-ledger",
    ) -> None:
        self.store = store
        self.inventory = inventory
        self.links = links
        self.clock = clock
        self.key = key
        self.inventory_hash = hashlib.sha256(inventory.model_dump_json().encode()).hexdigest()

    def validate(self, spec: GroupSpec) -> None:
        devices = {item.gpu_id: item for item in self.inventory.devices}
        if not set(spec.gpu_ids) <= devices.keys():
            raise ValueError("unknown_gpu")
        selected = [devices[gpu] for gpu in spec.gpu_ids]
        if len({device.node_id for device in selected}) != 1:
            raise ValueError("cross_node_collective_forbidden")
        if any(not device.healthy for device in selected):
            raise ValueError("unhealthy_gpu")
        if any(device.memory_bytes is None for device in selected):
            raise ValueError("gpu_memory_unmeasured")
        if any((device.memory_bytes or 0) < spec.memory_per_gpu_bytes for device in selected):
            raise ValueError("insufficient_gpu_memory")
        if spec.tensor_parallel_size > 1:
            if (
                not spec.evidence_reference
                or spec.evidence_expires_at is None
                or spec.evidence_expires_at <= self.clock()
            ):
                raise ValueError("parallel_evidence_unavailable")
            decision = ParallelPlanner().evaluate(
                ParallelCandidate(
                    ParallelMode.TENSOR,
                    spec.gpu_ids,
                    len({device.compute_capability for device in selected}) == 1,
                    spec.communication_fraction,
                    spec.measured_speedup,
                ),
                self.links,
            )
            if not decision.accepted:
                raise ValueError(",".join(decision.reasons))

    async def snapshot(self) -> GroupLedger:
        _, ledger = await self._read()
        return ledger.model_copy(deep=True)

    async def _read(self) -> tuple[int, GroupLedger]:
        item = await self.store.get(self.key)
        if item is None:
            return 0, GroupLedger(inventory_hash=self.inventory_hash)
        ledger = GroupLedger.model_validate_json(item.value)
        if ledger.inventory_hash != self.inventory_hash:
            raise ValueError("inventory_changed_requires_reconciliation")
        return item.revision, ledger

    async def _mutate(self, update: Callable[[GroupLedger], GroupLedger]) -> GroupLedger:
        for _ in range(32):
            revision, ledger = await self._read()
            changed = update(ledger)
            try:
                await self.store.compare_and_put(
                    self.key, revision, changed.model_dump_json().encode()
                )
                return changed
            except CompareFailed:
                continue
        raise RuntimeError("allocation_contention_retry_exhausted")

    async def reserve(
        self,
        spec: GroupSpec,
        owner_id: str,
        ttl_seconds: int = 60,
        *,
        expected_previous_generation: int = 0,
    ) -> GroupLease:
        if not owner_id.strip() or not 1 <= ttl_seconds <= 3600:
            raise ValueError("invalid_owner_or_ttl")
        self.validate(spec)

        def update(ledger: GroupLedger) -> GroupLedger:
            self.validate(spec)
            current = ledger.groups.get(spec.group_id)
            if current is None and expected_previous_generation != 0:
                raise ValueError("previous_generation_mismatch")
            if current is not None and expected_previous_generation != (
                current.generation
                if current.state is GroupState.RELEASED
                else current.previous_generation
            ):
                raise ValueError("previous_generation_mismatch")
            if current is not None and current.state is not GroupState.RELEASED:
                if (
                    current.spec == spec
                    and current.owner_id == owner_id
                    and current.expires_at > self.clock()
                    and current.state not in {GroupState.FAILED, GroupState.QUARANTINED}
                ):
                    return ledger  # An idempotent retry does not extend the lease.
                raise ValueError("group_conflict")
            for lease in ledger.groups.values():
                if lease.state is not GroupState.RELEASED and (
                    set(spec.gpu_ids) & set(lease.spec.gpu_ids)
                    or spec.worker_id == lease.spec.worker_id
                ):
                    raise ValueError("resources_busy")
            lease = GroupLease(
                spec=spec,
                owner_id=owner_id,
                node_id=next(
                    item.node_id
                    for item in self.inventory.devices
                    if item.gpu_id == spec.gpu_ids[0]
                ),
                generation=ledger.next_generation,
                previous_generation=expected_previous_generation,
                expires_at=self.clock() + timedelta(seconds=ttl_seconds),
            )
            return ledger.model_copy(
                update={
                    "next_generation": ledger.next_generation + 1,
                    "groups": {**ledger.groups, spec.group_id: lease},
                }
            )

        return (await self._mutate(update)).groups[spec.group_id]

    async def transition(
        self,
        group_id: str,
        generation: int,
        owner_id: str,
        target: GroupState,
        *,
        engine_instance_id: str | None = None,
        acknowledgement: str | None = None,
    ) -> GroupLease:
        def update(ledger: GroupLedger) -> GroupLedger:
            lease = ledger.groups[group_id]
            if lease.generation != generation or lease.owner_id != owner_id:
                raise ValueError("generation_or_owner_fence")
            if (
                engine_instance_id is not None
                and lease.engine_instance_id is not None
                and engine_instance_id != lease.engine_instance_id
            ):
                raise ValueError("engine_instance_fence")
            if lease.state is target:
                return ledger
            if lease.expires_at <= self.clock() and target not in {
                GroupState.QUARANTINED,
                GroupState.STOPPING,
                GroupState.RELEASED,
            }:
                raise ValueError("lease_expired")
            if target not in TRANSITIONS[lease.state]:
                raise ValueError("invalid_group_transition")
            if target is GroupState.READY and (not engine_instance_id or not acknowledgement):
                raise ValueError("readiness_ack_required")
            if target is GroupState.RELEASED and not acknowledgement:
                raise ValueError("quiescence_ack_required")
            changed = lease.model_copy(
                update={
                    "state": target,
                    "engine_instance_id": engine_instance_id or lease.engine_instance_id,
                    "acknowledgement": acknowledgement,
                }
            )
            return ledger.model_copy(update={"groups": {**ledger.groups, group_id: changed}})

        return (await self._mutate(update)).groups[group_id]

    async def renew(
        self, group_id: str, generation: int, owner_id: str, ttl_seconds: int = 60
    ) -> GroupLease:
        if not 1 <= ttl_seconds <= 3600:
            raise ValueError("invalid_ttl")

        def update(ledger: GroupLedger) -> GroupLedger:
            lease = ledger.groups[group_id]
            if lease.generation != generation or lease.owner_id != owner_id:
                raise ValueError("generation_or_owner_fence")
            if lease.expires_at <= self.clock() or lease.state not in {
                GroupState.RESERVED,
                GroupState.STARTING,
                GroupState.READY,
                GroupState.DRAINING,
            }:
                raise ValueError("lease_not_renewable")
            lease = lease.model_copy(
                update={"expires_at": self.clock() + timedelta(seconds=ttl_seconds)}
            )
            return ledger.model_copy(update={"groups": {**ledger.groups, group_id: lease}})

        return (await self._mutate(update)).groups[group_id]

    async def quarantine_expired(self) -> GroupLedger:
        def update(ledger: GroupLedger) -> GroupLedger:
            return ledger.model_copy(
                update={
                    "groups": {
                        key: lease.model_copy(update={"state": GroupState.QUARANTINED})
                        if lease.expires_at <= self.clock()
                        and lease.state
                        not in {GroupState.RELEASED, GroupState.QUARANTINED, GroupState.STOPPING}
                        else lease
                        for key, lease in ledger.groups.items()
                    }
                }
            )

        return await self._mutate(update)
