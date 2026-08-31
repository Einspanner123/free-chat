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
        return self.topology_generation, tuple(
            self._workers[key] for key in sorted(self._workers)
        )

    async def register(
        self,
        capabilities: WorkerCapabilities,
        telemetry: WorkerTelemetry,
    ) -> None:
        self.upsert(capabilities, telemetry)
        self.topology_generation += 1

    async def heartbeat(self, telemetry: WorkerTelemetry) -> None:
        current = self._workers.get(telemetry.worker_id)
        if current is None:
            raise ValueError("worker_not_registered")
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
        await self._store.put(
            f"{self._worker_prefix}{capabilities.worker_id}",
            _encode_snapshot(capabilities, telemetry),
        )
        self.upsert(capabilities, telemetry)
        self.topology_generation = await self._increment_topology_generation()

    async def heartbeat(self, telemetry: WorkerTelemetry) -> None:
        current = self._workers.get(telemetry.worker_id)
        if current is None:
            raise ValueError("worker_not_registered")
        if current.capabilities.generation != telemetry.generation:
            raise ValueError("worker_generation_mismatch")
        await self._store.put(
            f"{self._worker_prefix}{telemetry.worker_id}",
            _encode_snapshot(current.capabilities, telemetry),
        )
        self.upsert(current.capabilities, telemetry)

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


def _decode_snapshot(value: bytes) -> tuple[WorkerCapabilities, WorkerTelemetry]:
    import json

    record = json.loads(value)
    return (
        WorkerCapabilities.model_validate(record["capabilities"]),
        WorkerTelemetry.model_validate(record["telemetry"]),
    )
