import asyncio
from typing import Any

import pytest
from freechat_control_store import InMemoryStore
from freechat_trace_replay import EventEnvelope, LifecyclePublisher
from freechat_trace_replay.bus import DurableLifecycleEmitter


class FakeJetStream:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, dict[str, str] | None]] = []

    async def publish(
        self,
        subject: str,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> Any:
        self.calls.append((subject, payload, headers))
        return None


def test_lifecycle_subject_and_deduplication_header() -> None:
    async def scenario() -> None:
        stream = FakeJetStream()
        publisher = LifecyclePublisher(stream)
        event = EventEnvelope(
            event_type="lifecycle.tool_wait",
            tenant_id="tenant-a",
            aggregate_id="task",
            aggregate_generation=4,
            payload={},
        )
        subject = await publisher.publish(event, "langgraph")
        assert subject == "freechat.lifecycle.tenant-a.langgraph.lifecycle_tool_wait"
        assert stream.calls[0][2] is not None
        assert stream.calls[0][2]["Nats-Msg-Id"] == event.event_id

    asyncio.run(scenario())


def test_subject_injection_is_rejected() -> None:
    async def scenario() -> None:
        publisher = LifecyclePublisher(FakeJetStream())
        event = EventEnvelope(
            event_type="active",
            tenant_id="tenant.*",
            aggregate_id="task",
            aggregate_generation=1,
            payload={},
        )
        with pytest.raises(ValueError, match="subject segment"):
            await publisher.publish(event, "openhands")

    asyncio.run(scenario())


def test_durable_outbox_replays_after_publish_failure() -> None:
    class FailingJetStream(FakeJetStream):
        async def publish(
            self,
            subject: str,
            payload: bytes,
            *,
            headers: dict[str, str] | None = None,
        ) -> Any:
            raise RuntimeError("nats unavailable")

    async def scenario() -> None:
        store = InMemoryStore()
        event = EventEnvelope(
            event_type="lifecycle.tool_wait",
            tenant_id="tenant-a",
            aggregate_id="task",
            aggregate_generation=4,
            payload={},
        )
        emitter = DurableLifecycleEmitter(store, LifecyclePublisher(FailingJetStream()))
        with pytest.raises(RuntimeError, match="unavailable"):
            await emitter.emit(event, "langgraph")
        assert len(await store.list_prefix("/freechat/outbox/")) == 1

        recovered_stream = FakeJetStream()
        recovered = DurableLifecycleEmitter(store, LifecyclePublisher(recovered_stream))
        assert await recovered.replay() == 1
        assert len(await store.list_prefix("/freechat/outbox/")) == 0
        assert recovered_stream.calls[0][2] is not None
        assert recovered_stream.calls[0][2]["Nats-Msg-Id"] == event.event_id

    asyncio.run(scenario())
