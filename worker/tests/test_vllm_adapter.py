from dataclasses import dataclass
from enum import StrEnum

from freechat_worker import CacheEventBuffer, GenerationGuard, VllmCacheHookBridge


class EventType(StrEnum):
    ALLOCATE = "allocate"


@dataclass
class Metadata:
    tenant_id: str = "tenant-a"
    cache_key: str = "salted-cache-key"
    task_id: str = "task"
    agent_id: str = "agent"
    branch_id: str = "main"
    call_id: str = "call"
    lifecycle: str = "active"
    worker_generation: int = 4
    cache_generation: int = 9


@dataclass
class Event:
    event_type: EventType
    request_id: str
    block_ids: tuple[tuple[int, ...], ...]
    metadata: Metadata | None
    reason: str


def bridge(buffer: CacheEventBuffer) -> VllmCacheHookBridge:
    return VllmCacheHookBridge(
        buffer,
        GenerationGuard(worker_generation=4, cache_generation=9),
        worker_id="ross-a6000",
        block_tokens=(16,),
        block_bytes=(4096,),
    )


def test_vllm_bridge_maps_authenticated_correlated_blocks() -> None:
    buffer = CacheEventBuffer(capacity=8, batch_size=8)
    hook = bridge(buffer)
    hook.on_cache_event(
        Event(EventType.ALLOCATE, "request", ((3, 4),), Metadata(), "slots")
    )

    assert hook.stats.accepted == 2
    assert buffer.stats.accepted == 2


def test_vllm_bridge_drops_stale_and_uncorrelated_events() -> None:
    buffer = CacheEventBuffer(capacity=8, batch_size=8)
    hook = bridge(buffer)
    hook.on_cache_event(
        Event(
            EventType.ALLOCATE,
            "request",
            ((3,),),
            Metadata(worker_generation=3),
            "slots",
        )
    )
    hook.on_cache_event(Event(EventType.ALLOCATE, "request", ((4,),), None, "slots"))

    assert hook.stats.stale == 1
    assert hook.stats.uncorrelated == 1
    assert buffer.stats.accepted == 0
