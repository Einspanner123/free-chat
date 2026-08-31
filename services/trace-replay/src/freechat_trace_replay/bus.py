from __future__ import annotations

import re
from typing import Any, Protocol

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


def _safe_segment(value: str) -> str:
    if not _SEGMENT.fullmatch(value):
        raise ValueError(f"invalid JetStream subject segment: {value!r}")
    return value
