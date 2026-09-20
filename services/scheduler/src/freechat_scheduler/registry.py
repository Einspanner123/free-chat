from __future__ import annotations

from dataclasses import dataclass, field

from freechat_contracts import WorkerCapabilities, WorkerTelemetry
from freechat_control_store import CompareFailed, KeyValueStore


@dataclass(slots=True)
class WorkerSnapshot:
    capabilities: WorkerCapabilities
    telemetry: WorkerTelemetry


@dataclass(slots=True)
class InMemoryWorkerRegistry:
    """Deterministic registry used by the scheduler core and simulation tests.

    The production etcd adapter must produce this immutable snapshot shape. It
    is intentionally separate from policy so etcd watches cannot mutate a route
    calculation halfway through a decision.
    """

    topology_generation: int = 1
    _workers: dict[str, WorkerSnapshot] = field(default_factory=dict)

    def upsert(
        self,
        capabilities: WorkerCapabilities,
        telemetry: WorkerTelemetry,
    ) -> None:
        if capabilities.worker_id != telemetry.worker_id:
            raise ValueError("capability and telemetry worker IDs differ")
        if capabilities.generation != telemetry.generation:
            raise ValueError("capability and telemetry generations differ")
        self._workers[capabilities.worker_id] = WorkerSnapshot(capabilities, telemetry)

    def remove(self, worker_id: str, generation: int) -> bool:
        current = self._workers.get(worker_id)
        if current is None or current.capabilities.generation != generation:
            return False
        del self._workers[worker_id]
        return True

    def snapshot(self) -> tuple[int, tuple[WorkerSnapshot, ...]]:
        return self.topology_generation, tuple(self._workers[key] for key in sorted(self._workers))

    async def register(
        self,
        capabilities: WorkerCapabilities,
        telemetry: WorkerTelemetry,
    ) -> None:
        current = self._workers.get(capabilities.worker_id)
        snapshot = _registration_snapshot(current, capabilities, telemetry)
        self.upsert(snapshot.capabilities, snapshot.telemetry)
        if current is None or current.capabilities != snapshot.capabilities:
            self.topology_generation += 1

    async def heartbeat(self, telemetry: WorkerTelemetry) -> None:
        current = self._workers.get(telemetry.worker_id)
        if current is None:
            raise ValueError("worker_not_registered")
        _validate_heartbeat(current, telemetry)
        self.upsert(current.capabilities, telemetry)


class PersistentWorkerRegistry(InMemoryWorkerRegistry):
    """Read-optimized registry whose authoritative records live in etcd."""

    _worker_prefix = "/freechat/workers/"
    _topology_key = "/freechat/meta/topology-generation"

    def __init__(self, store: KeyValueStore) -> None:
        super().__init__()
        self._store = store

    async def restore(self) -> None:
        self._workers.clear()
        generation = await self._store.get(self._topology_key)
        self.topology_generation = 1 if generation is None else int(generation.value)
        for item in await self._store.list_prefix(self._worker_prefix):
            capabilities, telemetry = _decode_snapshot(item.value)
            self.upsert(capabilities, telemetry)

    async def register(
        self,
        capabilities: WorkerCapabilities,
        telemetry: WorkerTelemetry,
    ) -> None:
        key = f"{self._worker_prefix}{capabilities.worker_id}"
        for _ in range(32):
            item = await self._store.get(key)
            current = None if item is None else WorkerSnapshot(*_decode_snapshot(item.value))
            snapshot = _registration_snapshot(current, capabilities, telemetry)
            if current is not None and current.capabilities == capabilities:
                self.upsert(current.capabilities, current.telemetry)
                return
            try:
                await self._store.compare_and_put(
                    key,
                    0 if item is None else item.revision,
                    _encode_snapshot(snapshot.capabilities, snapshot.telemetry),
                )
            except CompareFailed:
                continue
            self.upsert(snapshot.capabilities, snapshot.telemetry)
            self.topology_generation = await self._increment_topology_generation()
            return
        raise RuntimeError("registration_contention_retry_exhausted")

    async def heartbeat(self, telemetry: WorkerTelemetry) -> None:
        key = f"{self._worker_prefix}{telemetry.worker_id}"
        for _ in range(32):
            item = await self._store.get(key)
            if item is None:
                raise ValueError("worker_not_registered")
            current = WorkerSnapshot(*_decode_snapshot(item.value))
            _validate_heartbeat(current, telemetry)
            try:
                await self._store.compare_and_put(
                    key,
                    item.revision,
                    _encode_snapshot(current.capabilities, telemetry),
                )
            except CompareFailed:
                continue
            self.upsert(current.capabilities, telemetry)
            return
        raise RuntimeError("heartbeat_contention_retry_exhausted")

    async def _increment_topology_generation(self) -> int:
        while True:
            current = await self._store.get(self._topology_key)
            revision = 0 if current is None else current.revision
            generation = 2 if current is None else int(current.value) + 1
            try:
                await self._store.compare_and_put(
                    self._topology_key,
                    revision,
                    str(generation).encode(),
                )
            except CompareFailed:
                continue
            return generation


def _registration_snapshot(
    current: WorkerSnapshot | None,
    capabilities: WorkerCapabilities,
    telemetry: WorkerTelemetry,
) -> WorkerSnapshot:
    if (
        capabilities.worker_id != telemetry.worker_id
        or capabilities.generation != telemetry.generation
    ):
        raise ValueError("registration_telemetry_mismatch")
    if current is not None:
        if capabilities.generation < current.capabilities.generation:
            raise ValueError("stale_worker_registration")
        if capabilities.generation == current.capabilities.generation:
            if capabilities != current.capabilities:
                raise ValueError("registration_incarnation_conflict")
            # A retry must not reset engine identity, load or observation time.
            return current
    return WorkerSnapshot(capabilities, telemetry)


def _encode_snapshot(
    capabilities: WorkerCapabilities,
    telemetry: WorkerTelemetry,
) -> bytes:
    return (
        b'{"capabilities":'
        + capabilities.model_dump_json().encode()
        + b',"telemetry":'
        + telemetry.model_dump_json().encode()
        + b"}"
    )


def _validate_heartbeat(current: WorkerSnapshot, telemetry: WorkerTelemetry) -> None:
    if current.capabilities.generation != telemetry.generation:
        raise ValueError("worker_generation_mismatch")
    if telemetry.observed_at.tzinfo is None:
        raise ValueError("telemetry_timestamp_without_timezone")
    if telemetry.observed_at < current.telemetry.observed_at:
        raise ValueError("out_of_order_telemetry")
    instance = current.telemetry.engine_instance_id
    if instance is not None and telemetry.engine_instance_id != instance:
        raise ValueError("engine_instance_changed_requires_registration")


def _decode_snapshot(value: bytes) -> tuple[WorkerCapabilities, WorkerTelemetry]:
    import json

    record = json.loads(value)
    return (
        WorkerCapabilities.model_validate(record["capabilities"]),
        WorkerTelemetry.model_validate(record["telemetry"]),
    )
