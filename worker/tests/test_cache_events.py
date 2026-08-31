import asyncio

import pytest
from freechat_contracts import CacheEvent, CacheEventKind
from freechat_worker import CacheEventBuffer


def event(block_id: str) -> CacheEvent:
    return CacheEvent(
        kind=CacheEventKind.ALLOCATE,
        tenant_id="tenant",
        worker_id="worker",
        worker_generation=3,
        cache_generation=7,
        block_id=block_id,
        cache_key=f"salted:{block_id}",
        token_count=16,
        bytes=4096,
    )


def test_hot_path_never_blocks_when_buffer_is_full() -> None:
    buffer = CacheEventBuffer(capacity=2, batch_size=2)
    assert buffer.try_emit(event("a"))
    assert buffer.try_emit(event("b"))
    assert not buffer.try_emit(event("c"))
    assert buffer.stats.accepted == 2
    assert buffer.stats.dropped == 1


def test_failed_delivery_is_requeued_with_observable_failure() -> None:
    async def scenario() -> None:
        buffer = CacheEventBuffer(capacity=4, batch_size=2)
        buffer.try_emit(event("a"))
        buffer.try_emit(event("b"))

        async def failing_sink(_: tuple[CacheEvent, ...]) -> None:
            raise RuntimeError("nats unavailable")

        with pytest.raises(RuntimeError, match="nats unavailable"):
            await buffer.drain_once(failing_sink)
        assert buffer.stats.delivered == 0
        assert buffer.stats.dropped == 0

        delivered: list[str] = []

        async def sink(events: tuple[CacheEvent, ...]) -> None:
            delivered.extend(item.block_id for item in events)

        assert await buffer.drain_once(sink) == 2
        assert sorted(delivered) == ["a", "b"]

    asyncio.run(scenario())
