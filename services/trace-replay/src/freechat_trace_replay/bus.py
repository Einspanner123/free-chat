from __future__ import annotations

import re
from typing import Any, Protocol

import nats
from freechat_control_store import CompareFailed, KeyValueStore
from nats.aio.client import Client as NatsClient
from nats.js.errors import NotFoundError
from pydantic import BaseModel, ConfigDict

from freechat_trace_replay.events import EventEnvelope

_SEGMENT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class JetStreamContext(Protocol):
    async def publish(
        self,
        subject: str,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> Any: ...


class LifecyclePublisher:
    def __init__(self, jetstream: JetStreamContext) -> None:
        self._jetstream = jetstream

    async def publish(self, event: EventEnvelope, harness_id: str) -> str:
        tenant = _safe_segment(event.tenant_id)
        harness = _safe_segment(harness_id)
        event_name = _safe_segment(event.event_type.replace(".", "_"))
        subject = f"freechat.lifecycle.{tenant}.{harness}.{event_name}"
        await self._jetstream.publish(
            subject,
            event.model_dump_json().encode(),
            headers={
                "Nats-Msg-Id": event.event_id,
                "traceparent": event.traceparent or "",
                "aggregate-generation": str(event.aggregate_generation),
            },
        )
        return subject


class DurableLifecycleEmitter:
    """Etcd-backed outbox with JetStream message-id deduplication.

    A successful return means the event received a JetStream publish ack. A
    failed publish leaves the outbox entry for replay after reconnect/restart.
    """

    _prefix = "/freechat/outbox/"

    def __init__(self, store: KeyValueStore, publisher: LifecyclePublisher) -> None:
        self._store = store
        self._publisher = publisher

    async def emit(self, event: EventEnvelope, harness_id: str) -> None:
        key = f"{self._prefix}{event.event_id}"
        record = _OutboxRecord(event=event, harness_id=harness_id)
        try:
            item = await self._store.compare_and_put(key, 0, record.model_dump_json().encode())
        except CompareFailed:
            existing = await self._store.get(key)
            if existing is None:
                return
            item = existing
        await self._publisher.publish(event, harness_id)
        await self._store.compare_and_delete(key, item.revision)

    async def replay(self) -> int:
        sent = 0
        for item in await self._store.list_prefix(self._prefix):
            record = _OutboxRecord.model_validate_json(item.value)
            await self._publisher.publish(record.event, record.harness_id)
            try:
                await self._store.compare_and_delete(item.key, item.revision)
            except CompareFailed:
                continue
            sent += 1
        return sent


class _OutboxRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: EventEnvelope
    harness_id: str


async def connect_lifecycle_stream(url: str) -> tuple[NatsClient, LifecyclePublisher]:
    client = await nats.connect(url, name="freechat-scheduler", connect_timeout=5)
    jetstream = client.jetstream()
    try:
        await jetstream.stream_info("FREECHAT_LIFECYCLE")
    except NotFoundError:
        await jetstream.add_stream(
            name="FREECHAT_LIFECYCLE",
            subjects=["freechat.lifecycle.>"],
            storage="file",
            duplicate_window=120,
        )
    return client, LifecyclePublisher(jetstream)


def _safe_segment(value: str) -> str:
    if not _SEGMENT.fullmatch(value):
        raise ValueError(f"invalid JetStream subject segment: {value!r}")
    return value
