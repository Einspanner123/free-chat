from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    tenant_id: str
    aggregate_id: str
    aggregate_generation: int = Field(ge=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    traceparent: str | None = None
    payload: dict[str, Any]


Handler = Callable[[EventEnvelope], Awaitable[None]]


class IdempotentEventConsumer:
    """Defines JetStream consumer semantics independent of the NATS client.

    Production persistence replaces the in-memory set with an etcd generation
    checkpoint. Ack occurs only after `consume` returns successfully.
    """

    def __init__(self) -> None:
        self._processed: set[str] = set()
        self._generation: dict[str, int] = {}

    async def consume(self, event: EventEnvelope, handler: Handler) -> bool:
        if event.event_id in self._processed:
            return False
        current = self._generation.get(event.aggregate_id, 0)
        if event.aggregate_generation < current:
            self._processed.add(event.event_id)
            return False
        await handler(event)
        self._processed.add(event.event_id)
        self._generation[event.aggregate_id] = max(current, event.aggregate_generation)
        return True
