import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from freechat_contracts import WorkerTelemetry
from freechat_control_store import CompareFailed, InMemoryStore, KeyValue
from freechat_scheduler.registry import InMemoryWorkerRegistry, PersistentWorkerRegistry
from test_persistence import capabilities


@pytest.mark.parametrize("persistent", [False, True])
async def test_registration_is_monotonic_and_retries_preserve_telemetry(persistent: bool) -> None:
    registry = PersistentWorkerRegistry(InMemoryStore()) if persistent else InMemoryWorkerRegistry()
    caps = capabilities(8)
    telemetry = WorkerTelemetry(
        worker_id=caps.worker_id,
        generation=8,
        free_vram_bytes=10,
        engine_instance_id="engine",
        active_requests=3,
    )
    await registry.register(caps, telemetry)
    topology = registry.topology_generation
    await registry.register(
        caps, telemetry.model_copy(update={"engine_instance_id": None, "active_requests": 0})
    )
    assert registry.snapshot()[1][0].telemetry == telemetry
    assert registry.topology_generation == topology
    with pytest.raises(ValueError, match="stale_worker_registration"):
        await registry.register(capabilities(7), telemetry.model_copy(update={"generation": 7}))
    with pytest.raises(ValueError, match="incarnation_conflict"):
        await registry.register(caps.model_copy(update={"endpoint": "http://other"}), telemetry)
    with pytest.raises(ValueError, match="telemetry_mismatch"):
        await registry.register(caps, telemetry.model_copy(update={"generation": 9}))
    await registry.register(
        capabilities(9), telemetry.model_copy(update={"generation": 9, "engine_instance_id": "new"})
    )
    assert registry.snapshot()[1][0].capabilities.generation == 9


async def test_stale_replica_heartbeat_cannot_restore_previous_generation() -> None:
    store = InMemoryStore()
    first, second = PersistentWorkerRegistry(store), PersistentWorkerRegistry(store)
    telemetry = WorkerTelemetry(
        worker_id=capabilities().worker_id, generation=3, free_vram_bytes=10
    )
    await first.register(capabilities(), telemetry)
    await second.restore()
    await first.register(capabilities(4), telemetry.model_copy(update={"generation": 4}))
    with pytest.raises(ValueError, match="generation_mismatch"):
        await second.heartbeat(telemetry)
    await second.restore()
    assert second.snapshot()[1][0].capabilities.generation == 4


class RacingStore(InMemoryStore):
    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        await asyncio.sleep(0)
        return await super().compare_and_put(key, expected_revision, value)


async def test_concurrent_registration_and_heartbeat_use_authoritative_cas() -> None:
    store = RacingStore()
    controllers = [PersistentWorkerRegistry(store) for _ in range(6)]
    now = datetime.now(UTC)
    telemetry = WorkerTelemetry(
        worker_id=capabilities().worker_id, generation=3, free_vram_bytes=10
    )
    await asyncio.gather(
        *(controller.register(capabilities(), telemetry) for controller in controllers)
    )
    # All writers share a generation but race newer observations; only monotonic writes survive.
    results = await asyncio.gather(
        *(
            controller.heartbeat(
                telemetry.model_copy(update={"observed_at": now + timedelta(seconds=i + 1)})
            )
            for i, controller in enumerate(controllers)
        ),
        return_exceptions=True,
    )
    assert all(result is None or isinstance(result, ValueError) for result in results)
    await controllers[0].restore()
    assert controllers[0].snapshot()[1][0].telemetry.observed_at == now + timedelta(seconds=6)


class RejectingStore(InMemoryStore):
    reject = False

    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        if self.reject:
            raise CompareFailed("forced contention")
        return await super().compare_and_put(key, expected_revision, value)


async def test_contention_is_bounded_and_failure_does_not_publish_local_state() -> None:
    store = RejectingStore()
    registry = PersistentWorkerRegistry(store)
    telemetry = WorkerTelemetry(
        worker_id=capabilities().worker_id, generation=3, free_vram_bytes=10
    )
    await registry.register(capabilities(), telemetry)
    store.reject = True
    with pytest.raises(RuntimeError, match="registration_contention"):
        await registry.register(capabilities(4), telemetry.model_copy(update={"generation": 4}))
    with pytest.raises(RuntimeError, match="heartbeat_contention"):
        await registry.heartbeat(telemetry)
    assert registry.snapshot()[1][0].capabilities.generation == 3


async def test_missing_authoritative_registration_rejects_heartbeat() -> None:
    registry = PersistentWorkerRegistry(InMemoryStore())
    with pytest.raises(ValueError, match="not_registered"):
        await registry.heartbeat(
            WorkerTelemetry(
                worker_id="absent",
                generation=1,
                free_vram_bytes=0,
            )
        )


class UnavailableStore(InMemoryStore):
    async def compare_and_put(self, key: str, expected_revision: int, value: bytes) -> KeyValue:
        raise OSError("unavailable")


async def test_unavailable_store_does_not_publish_registration() -> None:
    registry = PersistentWorkerRegistry(UnavailableStore())
    with pytest.raises(OSError):
        await registry.register(
            capabilities(),
            WorkerTelemetry(
                worker_id=capabilities().worker_id,
                generation=3,
                free_vram_bytes=0,
            ),
        )
    assert registry.snapshot()[1] == ()
