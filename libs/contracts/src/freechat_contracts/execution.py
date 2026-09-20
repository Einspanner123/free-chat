"""Request observation contract. A terminal engine state alone is not a release proof."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExecutionAction(StrEnum):
    QUERY = "query"
    ABORT = "abort"


class ExecutionStatus(StrEnum):
    UNKNOWN = "unknown"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    NOT_ACCEPTED = "not_accepted"


TERMINAL_EXECUTION = {
    ExecutionStatus.COMPLETED,
    ExecutionStatus.ABORTED,
    ExecutionStatus.NOT_ACCEPTED,
}


class ExecutionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tenant_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    worker_generation: int = Field(ge=1)
    engine_instance_id: str = Field(min_length=1)
    action: ExecutionAction

    @property
    def operation_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


class ExecutionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: ExecutionCommand
    observation_sequence: int = Field(ge=0)
    observed_at: datetime
    status: ExecutionStatus
    quiescent: bool = False
    # Durable tombstone/admission fence: late data-plane retries cannot restart this call.
    admission_closed: bool = False

    @property
    def releasable(self) -> bool:
        return self.status in TERMINAL_EXECUTION and self.quiescent and self.admission_closed
