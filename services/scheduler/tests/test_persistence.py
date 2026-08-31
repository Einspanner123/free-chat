import asyncio

from freechat_contracts import ModelCapability, WorkerCapabilities, WorkerTelemetry
from freechat_control_store import InMemoryStore
from freechat_scheduler.grpc_server import LeaseBook
from freechat_scheduler.registry import PersistentWorkerRegistry
from test_leases import decision


def capabilities(generation: int = 3) -> WorkerCapabilities:
    return WorkerCapabilities(
        worker_id="ross-a6000",
        generation=generation,
        endpoint="http://ross-worker:8000",
        node_id="ross",
        gpu_id="0",
        gpu_name="NVIDIA RTX A6000",
        compute_capability="8.6",
        total_vram_bytes=48 * 1024**3,
        p2p_domain="ross-0",
        network_domain="weak-lan",
        models=(
            ModelCapability(
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                revision="model-sha",
                tokenizer_revision="tokenizer-sha",
                architecture="dense",
                attention="gqa",
                max_context_tokens=32_768,
                dtype="bfloat16",
            ),
        ),
    )


def test_registry_and_lease_restore_from_authoritative_store() -> None:
    async def scenario() -> None:
        store = InMemoryStore()
        first = PersistentWorkerRegistry(store)
        caps = capabilities()
        telemetry = WorkerTelemetry(
            worker_id=caps.worker_id,
            generation=caps.generation,
            free_vram_bytes=caps.total_vram_bytes,
            queue_depth=2,
        )
        await first.register(caps, telemetry)
        topology_generation = first.topology_generation

        restored = PersistentWorkerRegistry(store)
        await restored.restore()
        restored_generation, workers = restored.snapshot()
        assert restored_generation == topology_generation
        assert workers[0].capabilities == caps
        assert workers[0].telemetry.queue_depth == 2

        leases = LeaseBook(store)
        route = decision()
        await leases.remember(route)
        after_restart = LeaseBook(store)
        assert (
            await after_restart.require(route.decision_id, route.worker_id, route.worker_generation)
            == route
        )
        await after_restart.release(
            route.decision_id,
            route.worker_id,
            route.worker_generation,
        )

    asyncio.run(scenario())
