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


async def test_retry_publishes_original_durable_envelope() -> None:
    from datetime import timedelta

    store = InMemoryStore()
    event = EventEnvelope(
        event_type="active",
        tenant_id="tenant-a",
        aggregate_id="task",
        aggregate_generation=1,
        payload={},
    )
    stream = FakeJetStream()

    class FailOnce(FakeJetStream):
        async def publish(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError):
        await DurableLifecycleEmitter(store, LifecyclePublisher(FailOnce())).emit(event, "agent")
    retry = event.model_copy(update={"occurred_at": event.occurred_at + timedelta(seconds=5)})
    await DurableLifecycleEmitter(store, LifecyclePublisher(stream)).emit(retry, "agent")
    assert stream.calls[0][1] == event.model_dump_json().encode()
    assert await store.list_prefix("/freechat/outbox/") == ()


@pytest.mark.parametrize("changed", ["payload", "tenant_id", "aggregate_generation", "harness"])
async def test_conflicting_event_identity_never_overwrites_or_acknowledges_pending(
    changed: str,
) -> None:
    store = InMemoryStore()
    event = EventEnvelope(
        event_type="active",
        tenant_id="tenant-a",
        aggregate_id="task",
        aggregate_generation=1,
        payload={},
    )

    class FailOnce(FakeJetStream):
        async def publish(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError):
        await DurableLifecycleEmitter(store, LifecyclePublisher(FailOnce())).emit(event, "agent")
    before = await store.list_prefix("/freechat/outbox/")
    update: dict[str, Any] = {
        "payload": {"different": True},
        "tenant_id": "tenant-b",
        "aggregate_generation": 2,
    }
    retry = event if changed == "harness" else event.model_copy(update={changed: update[changed]})
    stream = FakeJetStream()
    with pytest.raises(ValueError, match="lifecycle_event_id_conflict"):
        await DurableLifecycleEmitter(store, LifecyclePublisher(stream)).emit(
            retry, "other" if changed == "harness" else "agent"
        )
    assert stream.calls == []
    assert await store.list_prefix("/freechat/outbox/") == before


class ConnectionFixture:
    def __init__(self, *, stall: bool = False, fail_setup: bool = False) -> None:
        self.stall, self.fail_setup, self.closed = stall, fail_setup, False
        self.options: dict[str, Any] = {}

    async def connect(self, url: str, **options: Any) -> None:
        self.options = options
        if self.stall:
            await asyncio.Event().wait()

    def jetstream(self, **options: Any) -> Any:
        assert options["timeout"] == 5
        return self

    async def stream_info(self, name: str) -> None:
        assert name == "FREECHAT_LIFECYCLE"
        if self.fail_setup:
            raise RuntimeError("stream configuration rejected")

    async def close(self) -> None:
        self.closed = True


async def test_runtime_reconnect_has_no_finite_attempt_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freechat_trace_replay.bus as bus

    client = ConnectionFixture()
    monkeypatch.setattr(bus, "NatsClient", lambda: client)
    connected, _ = await bus.connect_lifecycle_stream("nats://fixture")
    assert id(connected) == id(client)
    assert client.options["max_reconnect_attempts"] == -1
    assert client.options["reconnect_time_wait"] == 2
    assert client.closed is False


@pytest.mark.parametrize("failure", ["timeout", "setup", "cancel"])
async def test_failed_or_cancelled_bus_setup_closes_client(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import freechat_trace_replay.bus as bus

    client = ConnectionFixture(stall=failure != "setup", fail_setup=failure == "setup")
    monkeypatch.setattr(bus, "NatsClient", lambda: client)
    if failure == "cancel":
        task = asyncio.create_task(bus.connect_lifecycle_stream("nats://fixture"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
            await bus.connect_lifecycle_stream("nats://fixture", startup_timeout=0.01)
    assert client.closed


@pytest.mark.parametrize("state", ["connected", "closed", "reconnecting", "timeout"])
async def test_bus_shutdown_always_closes_without_waiting_for_reconnect(state: str) -> None:
    from freechat_trace_replay.bus import close_lifecycle_stream
    from nats.errors import ConnectionClosedError, ConnectionReconnectingError

    class Client:
        closed = False

        async def drain(self) -> None:
            if state == "closed":
                raise ConnectionClosedError()
            if state == "reconnecting":
                raise ConnectionReconnectingError()
            if state == "timeout":
                await asyncio.Event().wait()

        async def close(self) -> None:
            self.closed = True

    client: Any = Client()
    await close_lifecycle_stream(client, timeout=0.01)
    assert client.closed
