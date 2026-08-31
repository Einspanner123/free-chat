from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class EvidenceState(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    BLOCKED = "BLOCKED"


class MetricPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    value: float


class ConsoleSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: EvidenceState
    generated_at: datetime | None = None
    summary: str
    metrics: tuple[MetricPoint, ...] = ()
    records: tuple[dict[str, str | int | float | bool | None], ...] = ()


class ConsoleReadModel(Protocol):
    async def snapshot(self, view: str, tenant_id: str) -> ConsoleSnapshot: ...


class EmptyConsoleReadModel:
    async def snapshot(self, view: str, tenant_id: str) -> ConsoleSnapshot:
        del tenant_id
        return ConsoleSnapshot(
            state=EvidenceState.UNVERIFIED,
            summary=f"No control-plane evidence has been ingested for {view}.",
        )
