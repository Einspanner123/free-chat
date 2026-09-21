"""Request-bound cache control, independent of execution reservations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from freechat_contracts.execution import ExecutionAction, ExecutionCommand

MAX_CACHE_CONTROL_BYTES = 16 * 1024

CacheLifecycle = Literal["tool_wait", "resume", "terminal", "cancelled"]


class CacheLifecycleUpdate(BaseModel):
    """Caller intent only: tenant and engine identity must come from trusted routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    decision_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(strict=True, ge=1, le=2**63 - 1)
    lifecycle: CacheLifecycle
    expected_resume_ms: int | None = Field(default=None, strict=True, ge=0, le=300_000)

    @model_validator(mode="after")
    def validate_resume_horizon(self) -> CacheLifecycleUpdate:
        if (self.lifecycle == "tool_wait") != (self.expected_resume_ms is not None):
            raise ValueError("resume_horizon_only_for_tool_wait")
        return self


class CacheLifecycleCommand(BaseModel):
    """Original execution owner plus cache intent; never an admission or abort."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    owner: ExecutionCommand
    update: CacheLifecycleUpdate
    cache_generation: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def validate_owner_binding(self) -> CacheLifecycleCommand:
        if self.owner.action is not ExecutionAction.QUERY:
            raise ValueError("cache_control_is_not_execution_abort")
        if self.owner.decision_id != self.update.decision_id:
            raise ValueError("cache_decision_identity_mismatch")
        return self


class PrefixLifecycleReceipt(BaseModel):
    """Observed engine residency, not reserved bytes or proof of performance gain."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(min_length=1, max_length=512)
    sequence: int = Field(strict=True, ge=1)
    lifecycle: CacheLifecycle
    resident_blocks: int = Field(strict=True, ge=0)
    protected_blocks: int = Field(strict=True, ge=0)
    status: Literal["applied", "not_resident"]
    applied_at_ms: int = Field(strict=True, ge=0)
    replayed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_residency_counts(self) -> PrefixLifecycleReceipt:
        if self.protected_blocks > self.resident_blocks:
            raise ValueError("protection_exceeds_residency")
        if (self.status == "not_resident") != (self.resident_blocks == 0):
            raise ValueError("residency_status_mismatch")
        if self.lifecycle != "tool_wait" and self.protected_blocks:
            raise ValueError("closed_lifecycle_cannot_protect")
        return self


class CacheLifecycleReceipt(BaseModel):
    """Identity-bound engine results; observed_at is wall time, applied_at_ms is monotonic."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    command: CacheLifecycleCommand
    prefixes: tuple[PrefixLifecycleReceipt, ...] = Field(min_length=1, max_length=128)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_command_binding(self) -> CacheLifecycleReceipt:
        if self.observed_at.tzinfo is None:
            raise ValueError("cache_receipt_requires_timezone")
        if len({p.request_id for p in self.prefixes}) != len(self.prefixes):
            raise ValueError("duplicate_prefix_receipt")
        if any(
            p.sequence != self.command.update.sequence
            or p.lifecycle != self.command.update.lifecycle
            for p in self.prefixes
        ):
            raise ValueError("cache_receipt_command_mismatch")
        return self
