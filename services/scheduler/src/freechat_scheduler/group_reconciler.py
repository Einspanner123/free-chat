"""Bounded reconciliation of desired group state with fenced runtime observations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from freechat_control_store import KeyValueStore
from pydantic import BaseModel, ConfigDict, Field

from freechat_scheduler.group_runtime import (
    GroupAction,
    GroupCommand,
    GroupDriver,
    GroupReceipt,
    LocalGrpcGroupDriver,
    LocalRuntimeEndpoint,
    RuntimeStatus,
)
from freechat_scheduler.parallel import DeviceLink
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.resource_groups import (
    GroupController,
    GroupLease,
    GroupLedger,
    GroupState,
    Inventory,
)

LOGGER = logging.getLogger(__name__)


class GroupRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["local-contract"]
    owner_id: str = Field(min_length=1)
    inventory: Inventory
    links: tuple[DeviceLink, ...] = ()
    endpoints: dict[str, LocalRuntimeEndpoint]


class GroupReconciler:
    def __init__(
        self,
        controller: GroupController,
        registry: InMemoryWorkerRegistry,
        driver: GroupDriver,
        *,
        owner_id: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.controller, self.registry, self.driver = controller, registry, driver
        self.owner_id, self.clock = owner_id, clock
        self._view = (
            datetime.min.replace(tzinfo=UTC),
            GroupLedger(inventory_hash=controller.inventory_hash),
        )
        self._lock = asyncio.Lock()

    def snapshot(self) -> tuple[datetime, GroupLedger]:
        return self._view[0], self._view[1].model_copy(deep=True)

    def invalidate(self) -> None:
        self._view = (datetime.min.replace(tzinfo=UTC), self._view[1])

    async def tick(self) -> dict[str, str]:
        async with self._lock:
            self.invalidate()
            started = self.clock()
            if started.tzinfo is None:
                raise ValueError("reconciler_clock_requires_timezone")
            ledger = await self.controller.quarantine_expired()
            errors: dict[str, str] = {}

            async def observe(lease: GroupLease) -> None:
                try:
                    async with asyncio.timeout(2):
                        await self._step(lease)
                except Exception as error:
                    # Retain allocation. Do not turn transport failures into stop confirmation.
                    errors[lease.spec.group_id] = type(error).__name__

            await asyncio.gather(
                *(
                    observe(lease)
                    for lease in ledger.groups.values()
                    if lease.owner_id == self.owner_id and lease.state is not GroupState.RELEASED
                )
            )
            current = await self.controller.snapshot()
            # Hide groups without a fresh valid runtime observation, even if persisted READY.
            current = current.model_copy(
                update={
                    "groups": {
                        key: value
                        for key, value in current.groups.items()
                        if value.owner_id == self.owner_id and key not in errors
                    }
                }
            )
            self._view = (started, current)
            return errors

    async def run(self) -> None:
        while True:
            try:
                errors = await self.tick()
                if errors:
                    LOGGER.warning("group observations unavailable: %s", errors)
            except Exception:
                self.invalidate()
                LOGGER.exception("group view invalidated after reconciliation failure")
            await asyncio.sleep(1)

    async def _transition(
        self, lease: GroupLease, target: GroupState, receipt: GroupReceipt | None = None
    ) -> GroupLease:
        return await self.controller.transition(
            lease.spec.group_id,
            lease.generation,
            self.owner_id,
            target,
            engine_instance_id=None if receipt is None else receipt.engine_instance_id,
            acknowledgement=None if receipt is None else receipt.model_dump_json(),
        )

    async def _step(self, lease: GroupLease) -> None:
        if lease.state is GroupState.RESERVED:
            lease = await self._transition(lease, GroupState.STARTING)
        elif lease.state in {GroupState.QUARANTINED, GroupState.FAILED}:
            lease = await self._transition(lease, GroupState.STOPPING)
        action = {
            GroupState.STARTING: GroupAction.START,
            GroupState.READY: GroupAction.INSPECT,
            GroupState.DRAINING: GroupAction.DRAIN,
            GroupState.STOPPING: GroupAction.STOP,
        }[lease.state]
        command = GroupCommand.for_lease(lease, action)
        receipt = await self.driver.apply(command)
        receipt.validate_envelope(command)
        if (
            receipt.observed_at.tzinfo is None
            or not 0 <= (self.clock() - receipt.observed_at).total_seconds() <= 5
        ):
            raise ValueError("runtime_observation_stale")
        if action in {GroupAction.START, GroupAction.INSPECT}:
            if receipt.status is not RuntimeStatus.READY:
                raise ValueError("runtime_not_ready")
            self._validate_ready(lease, receipt)
            assert receipt.capabilities is not None and receipt.telemetry is not None
            await self.registry.register(receipt.capabilities, receipt.telemetry)
            await self.registry.heartbeat(receipt.telemetry)
            await self._transition(lease, GroupState.READY, receipt)
        elif action is GroupAction.DRAIN:
            if receipt.status is RuntimeStatus.DRAINED and receipt.quiescent:
                await self._transition(lease, GroupState.STOPPING, receipt)
        elif receipt.status is RuntimeStatus.STOPPED and receipt.quiescent:
            await self._transition(lease, GroupState.RELEASED, receipt)

    def _validate_ready(self, lease: GroupLease, receipt: GroupReceipt) -> None:
        caps, telemetry, spec = receipt.capabilities, receipt.telemetry, lease.spec
        if caps is None or telemetry is None:
            raise ValueError("runtime_capability_and_budget_required")
        models = [m for m in caps.models if m.model_id == spec.model_id]
        if (
            caps.worker_id != spec.worker_id
            or caps.generation != lease.generation
            or caps.resource_group_id != spec.group_id
            or caps.resource_group_generation != lease.generation
            or caps.node_id != lease.node_id
            or caps.gpu_ids != spec.gpu_ids
            or len(models) != 1
            or len(caps.models) != 1
            or models[0].revision != spec.model_revision
            or models[0].tensor_parallel_size != spec.tensor_parallel_size
            or models[0].pipeline_parallel_size != 1
            or models[0].kv_admission_bytes_per_token_per_rank is None
            or models[0].kv_block_size_tokens is None
            or telemetry.worker_id != caps.worker_id
            or telemetry.generation != caps.generation
            or telemetry.engine_instance_id != receipt.engine_instance_id
            or not telemetry.healthy
            or telemetry.draining
            or receipt.quiescent
        ):
            raise ValueError("runtime_readiness_binding_mismatch")
        if (
            telemetry.observed_at.tzinfo is None
            or not 0 <= (self.clock() - telemetry.observed_at).total_seconds() <= 5
        ):
            raise ValueError("runtime_budget_stale")
        budgets = {item.gpu_id: item.available_bytes for item in receipt.rank_budgets}
        capacity = {item.gpu_id: item.memory_bytes for item in self.controller.inventory.devices}
        if (
            len(budgets) != len(receipt.rank_budgets)
            or set(budgets) != set(spec.gpu_ids)
            or any(
                capacity[gpu] is None or size > (capacity[gpu] or 0)
                for gpu, size in budgets.items()
            )
            or telemetry.kv_admission_available_bytes_per_rank != min(budgets.values())
        ):
            raise ValueError("runtime_rank_budget_mismatch")


def configured_reconciler(
    config: GroupRuntimeConfig, store: KeyValueStore, registry: InMemoryWorkerRegistry
) -> GroupReconciler:
    return GroupReconciler(
        GroupController(store, config.inventory, config.links),
        registry,
        LocalGrpcGroupDriver(config.endpoints),
        owner_id=config.owner_id,
    )
