from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any, Protocol

from freechat_control_store import CompareFailed, KeyValueStore
from nats.aio.client import Client as NatsClient
from nats.errors import ConnectionClosedError, ConnectionReconnectingError
from nats.js.errors import NotFoundError
from pydantic import BaseModel, ConfigDict

from freechat_trace_replay.events import EventEnvelope

LOGGER = logging.getLogger(__name__)

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
        subject = lifecycle_subject(event.tenant_id, harness_id, event.event_type)
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
            persisted = _OutboxRecord.model_validate_json(existing.value)
            # Retries may reconstruct the envelope with a later enqueue time.
            # Its first durable timestamp and bytes remain authoritative.
            if persisted.harness_id != harness_id or persisted.event.model_dump(
                exclude={"occurred_at"}
            ) != event.model_dump(exclude={"occurred_at"}):
                raise ValueError("lifecycle_event_id_conflict") from None
            record = persisted
            item = existing
        await self._publisher.publish(record.event, record.harness_id)
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


async def _connection_error(error: Exception) -> None:
    # Do not log connection URLs or exception text that could contain credentials.
    LOGGER.warning("lifecycle_bus_connection_error type=%s", type(error).__name__)


async def _reconnected() -> None:
    LOGGER.info("lifecycle_bus_reconnected")


async def connect_lifecycle_stream(
    url: str, *, startup_timeout: float = 10.0
) -> tuple[NatsClient, LifecyclePublisher]:
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("lifecycle_bus_startup_timeout_must_be_finite_positive")
    client = NatsClient()
    try:
        # Runtime outages must not permanently exhaust the client's retry budget.
        # Initial connection/stream setup still has a finite deadline.
        async with asyncio.timeout(startup_timeout):
            await client.connect(
                url,
                name="freechat-scheduler",
                connect_timeout=5,
                max_reconnect_attempts=-1,
                reconnect_time_wait=2,
                error_cb=_connection_error,
                reconnected_cb=_reconnected,
            )
            jetstream = client.jetstream(timeout=5)
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
    except BaseException:
        await client.close()
        raise


async def close_lifecycle_stream(client: NatsClient, *, timeout: float = 5.0) -> None:
    """Bound shutdown even if the bus is down; drain is not a delivery receipt."""
    try:
        async with asyncio.timeout(timeout):
            await client.drain()
    except (ConnectionClosedError, ConnectionReconnectingError, TimeoutError):
        LOGGER.warning("lifecycle_bus_drain_unavailable; unacknowledged delivery not confirmed")
    finally:
        await client.close()


def lifecycle_subject(tenant_id: str, harness_id: str, event_type: str) -> str:
    tenant, harness = _safe_segment(tenant_id), _safe_segment(harness_id)
    event_name = _safe_segment(event_type.replace(".", "_"))
    return f"freechat.lifecycle.{tenant}.{harness}.{event_name}"


def _safe_segment(value: str) -> str:
    if not _SEGMENT.fullmatch(value):
        raise ValueError(f"invalid JetStream subject segment: {value!r}")
    return value
