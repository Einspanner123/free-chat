from __future__ import annotations

from dataclasses import dataclass, field

from freechat_contracts import WorkerCapabilities, WorkerTelemetry


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
