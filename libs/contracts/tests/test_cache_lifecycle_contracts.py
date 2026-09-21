"""CPU contract checks; these do not demonstrate physical KV residency."""

from datetime import UTC, datetime
from typing import Any

import pytest
from freechat_contracts.cache_lifecycle import (
    CacheLifecycleCommand,
    CacheLifecycleReceipt,
    CacheLifecycleUpdate,
    PrefixLifecycleReceipt,
)
from freechat_contracts.execution import ExecutionCommand
from pydantic import ValidationError


def update(**changes: Any) -> CacheLifecycleUpdate:
    return CacheLifecycleUpdate.model_validate(
        {
            "decision_id": "decision",
            "sequence": 1,
            "lifecycle": "tool_wait",
            "expected_resume_ms": 1000,
            **changes,
        }
    )


def command() -> CacheLifecycleCommand:
    return CacheLifecycleCommand(
        owner=ExecutionCommand(
            tenant_id="tenant",
            request_id="request",
            decision_id="decision",
            worker_id="worker",
            worker_generation=1,
            engine_instance_id="engine",
            action="query",
        ),
        update=update(),
        cache_generation=1,
    )


def prefix(**changes: Any) -> PrefixLifecycleReceipt:
    return PrefixLifecycleReceipt.model_validate(
        {
            "request_id": "internal",
            "sequence": 1,
            "lifecycle": "tool_wait",
            "resident_blocks": 4,
            "protected_blocks": 4,
            "status": "applied",
            "applied_at_ms": 100,
            "replayed": False,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": True},
        {"sequence": "1"},
        {"sequence": 0},
        {"sequence": 2**63},
        {"tenant_id": "forged"},
        {"worker_id": "forged"},
        {"block_ids": [1]},
        {"expected_resume_ms": None},
        {"expected_resume_ms": -1},
        {"expected_resume_ms": 300001},
        {"expected_resume_ms": True},
        {"lifecycle": "resume"},
        {"decision_id": ""},
    ],
)
def test_update_rejects_invalid_intent(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        update(**changes)


@pytest.mark.parametrize("state", ["resume", "terminal", "cancelled"])
def test_non_wait_has_no_resume_horizon(state: str) -> None:
    assert update(lifecycle=state, expected_resume_ms=None).expected_resume_ms is None


@pytest.mark.parametrize("change", [{"action": "abort"}, {"decision_id": "other"}])
def test_cache_command_cannot_change_execution_owner(change: dict[str, Any]) -> None:
    data = command().model_dump()
    data["owner"].update(change)
    with pytest.raises(ValidationError):
        CacheLifecycleCommand.model_validate(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"protected_blocks": 5},
        {"resident_blocks": -1},
        {"resident_blocks": True},
        {"status": "not_resident"},
        {"lifecycle": "resume"},
        {"replayed": 1},
    ],
)
def test_prefix_receipt_rejects_inconsistent_residency(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        prefix(**changes)


def test_receipt_roundtrip_and_no_resident_blocks() -> None:
    receipt = CacheLifecycleReceipt(
        command=command(),
        prefixes=(prefix(resident_blocks=0, protected_blocks=0, status="not_resident"),),
        observed_at=datetime.now(UTC),
    )
    assert CacheLifecycleReceipt.model_validate_json(receipt.model_dump_json()) == receipt


@pytest.mark.parametrize("case", ["sequence", "lifecycle", "duplicate", "timezone"])
def test_receipt_must_match_command(case: str) -> None:
    item = prefix(sequence=2) if case == "sequence" else prefix()
    if case == "lifecycle":
        item = prefix(lifecycle="resume", protected_blocks=0)
    with pytest.raises(ValidationError):
        CacheLifecycleReceipt(
            command=command(),
            prefixes=(item, item) if case == "duplicate" else (item,),
            observed_at=datetime(2026, 1, 1) if case == "timezone" else datetime.now(UTC),
        )
