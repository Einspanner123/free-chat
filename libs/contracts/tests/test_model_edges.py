"""Model validation boundaries; these are CPU contract checks."""

from datetime import UTC, datetime, timedelta
from itertools import product
from typing import Any

import pytest
from freechat_contracts.models import (
    AgentHints,
    CostCalibration,
    Lifecycle,
    ModelCapability,
    PredictiveOffloadDirective,
    RequestProfile,
    WorkerTelemetry,
    validate_lifecycle_transition,
)
from freechat_contracts.preparation import body_digest
from pydantic import ValidationError


def hints(**changes: Any) -> AgentHints:
    return AgentHints.model_validate(
        {"harness_id": "harness", "task_id": "task", "agent_id": "agent", **changes}
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"parent_agent_id": "agent"}, "own parent"),
        ({"parent_branch_id": "main"}, "own parent"),
        ({"metadata": {str(i): "value" for i in range(33)}}, "32 entries"),
        ({"lifecycle": "resume"}, "expected_resume_ms"),
        ({"source": "inferred", "confidence": 1}, "confidence below"),
    ],
)
def test_hint_semantic_boundaries(changes: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        hints(**changes)


def test_hints_accept_explicit_zero_horizon_and_bounded_metadata() -> None:
    item = hints(
        source="inferred",
        confidence=0.99,
        lifecycle="resume",
        expected_resume_ms=0,
        parent_agent_id="parent",
        parent_branch_id="root",
        metadata={str(i): "value" for i in range(32)},
    )
    assert item.expected_resume_ms == 0
    assert len(item.metadata) == 32


# Independent expected state machine, not the production transition table.
EXPECTED_TRANSITIONS: dict[str, set[str]] = {
    "spawn": {"active", "cancelled"},
    "active": {"active", "tool_wait", "terminal", "cancelled"},
    "tool_wait": {"resume", "cancelled"},
    "resume": {"active", "tool_wait", "terminal", "cancelled"},
    "terminal": set(),
    "cancelled": set(),
}


@pytest.mark.parametrize("current,target", list(product(Lifecycle, repeat=2)))
def test_lifecycle_transition_matrix(current: Lifecycle, target: Lifecycle) -> None:
    if target.value in EXPECTED_TRANSITIONS[current.value]:
        validate_lifecycle_transition(current, target)
    else:
        with pytest.raises(ValueError, match="invalid lifecycle transition"):
            validate_lifecycle_transition(current, target)


def calibration(**changes: Any) -> CostCalibration:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return CostCalibration.model_validate(
        {
            "calibration_id": "calibration",
            "worker_id": "worker",
            "worker_generation": 1,
            "engine_instance_id": "engine",
            "model": ModelCapability(
                model_id="model",
                revision="sha",
                tokenizer_revision="tokenizer",
                architecture="dense",
                attention="gqa",
                max_context_tokens=64,
                dtype="half",
            ),
            "image_identity": "sha256:image",
            "observed_at": now,
            "expires_at": now + timedelta(hours=1),
            "artifact_sha256": "a" * 64,
            "sample_count": 3,
            "input_tokens_min": 1,
            "input_tokens_max": 64,
            "output_tokens_min": 2,
            "output_tokens_max": 32,
            "prefill_tokens_per_second": 10,
            "decode_tokens_per_second": 5,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"observed_at": datetime(2026, 1, 1)}, "timezone"),
        ({"expires_at": datetime(2026, 1, 1)}, "timezone"),
        ({"expires_at": datetime(2026, 1, 1, tzinfo=UTC)}, "expiry"),
        ({"input_tokens_min": 65}, "input-token"),
        ({"output_tokens_min": 33}, "output-token"),
        ({"store_bytes_per_second": 1}, "both directions"),
        ({"load_bytes_per_second": 1}, "both directions"),
        ({"transfer_bytes_min": 1}, "both directions"),
        ({"transfer_bytes_max": 1}, "both directions"),
        (
            {
                "store_bytes_per_second": 1,
                "load_bytes_per_second": 1,
                "transfer_bytes_min": 2,
                "transfer_bytes_max": 1,
            },
            "transfer-byte",
        ),
    ],
)
def test_calibration_rejects_invalid_scope(changes: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        calibration(**changes)


def test_calibration_accepts_exact_transfer_range_and_rejects_infinity() -> None:
    assert calibration().transfer_bytes_min is None
    item = calibration(
        store_bytes_per_second=1,
        load_bytes_per_second=2,
        transfer_bytes_min=8,
        transfer_bytes_max=8,
    )
    assert item.transfer_bytes_min == item.transfer_bytes_max == 8
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError):
            calibration(prefill_tokens_per_second=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_cache_capacity_bytes": 4},
        {"kv_cache_free_bytes": 4},
        {"kv_cache_capacity_bytes": 4, "kv_cache_free_bytes": 5},
    ],
)
def test_telemetry_rejects_unpaired_or_impossible_capacity(changes: Any) -> None:
    with pytest.raises(ValidationError):
        WorkerTelemetry(worker_id="worker", generation=1, free_vram_bytes=100, **changes)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"native_protocol": "/v1/responses"}, "native payload required"),
        ({"native_body_sha256": "a" * 64}, "native payload required"),
        ({"native_protocol": "/unknown", "native_request_json": "{}"}, "invalid native protocol"),
        ({"native_request_json": "x" * (4 * 1024 * 1024 + 1)}, "invalid native protocol"),
        ({"native_request_json": "[]", "native_protocol": "/v1/messages"}, "model mismatch"),
        (
            {"native_request_json": '{"model":"other"}', "native_protocol": "/v1/messages"},
            "model mismatch",
        ),
        (
            {
                "native_protocol": "/v1/messages",
                "native_request_json": '{"model":"model"}',
                "native_body_sha256": "a" * 64,
            },
            "fingerprint mismatch",
        ),
    ],
)
def test_request_native_body_identity_cannot_be_forged(changes: Any, reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        RequestProfile(
            tenant_id="tenant",
            model_id="model",
            input_tokens=0,
            output_tokens=1,
            hints=hints(),
            **changes,
        )


@pytest.mark.parametrize("fingerprint", [None, body_digest({"model": "model"})])
def test_native_body_fingerprint_is_derived_and_prompt_excluded(fingerprint: str | None) -> None:
    item = RequestProfile(
        tenant_id="tenant",
        model_id="model",
        input_tokens=0,
        output_tokens=1,
        hints=hints(),
        native_protocol="/v1/responses",
        native_request_json='{"model":"model"}',
        native_body_sha256=fingerprint,
    )
    assert item.native_body_sha256 == body_digest({"model": "model"})
    assert "native_request_json" not in item.model_dump()
    assert "native_request_json" not in repr(item)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"enabled": True}, "applicable"),
        ({"enabled": True, "applicable": True}, "tokens and estimated"),
        ({"enabled": True, "applicable": True, "max_offload_tokens": 1}, "tokens and estimated"),
        ({"max_offload_tokens": 1}, "disabled offload"),
    ],
)
def test_offload_directive_requires_actionable_budget(changes: Any, reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        PredictiveOffloadDirective(reason="test", **changes)


def test_enabled_offload_retains_exact_budget() -> None:
    item = PredictiveOffloadDirective(
        reason="test", enabled=True, applicable=True, max_offload_tokens=2, estimated_kv_bytes=8
    )
    assert item.max_offload_tokens == 2 and item.estimated_kv_bytes == 8
