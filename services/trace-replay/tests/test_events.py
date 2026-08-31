import asyncio

from freechat_trace_replay import EventEnvelope, IdempotentEventConsumer


def test_duplicate_event_is_not_applied_twice() -> None:
    asyncio.run(_assert_duplicate_event_is_not_applied_twice())


async def _assert_duplicate_event_is_not_applied_twice() -> None:
    consumer = IdempotentEventConsumer()
    event = EventEnvelope(
        event_type="lifecycle.active",
        tenant_id="tenant",
        aggregate_id="task",
        aggregate_generation=1,
        payload={},
    )
    calls = 0

    async def handler(_: EventEnvelope) -> None:
        nonlocal calls
        calls += 1

    assert await consumer.consume(event, handler)
    assert not await consumer.consume(event, handler)
    assert calls == 1


def test_stale_generation_is_ignored() -> None:
    asyncio.run(_assert_stale_generation_is_ignored())


async def _assert_stale_generation_is_ignored() -> None:
    consumer = IdempotentEventConsumer()

    async def handler(_: EventEnvelope) -> None:
        return None

    fresh = EventEnvelope(
        event_type="worker.status",
        tenant_id="system",
        aggregate_id="worker",
        aggregate_generation=4,
        payload={},
    )
    stale = fresh.model_copy(update={"event_id": "stale", "aggregate_generation": 3})
    assert await consumer.consume(fresh, handler)
    assert not await consumer.consume(stale, handler)
