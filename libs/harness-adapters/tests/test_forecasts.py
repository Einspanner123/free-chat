from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from freechat_contracts import Lifecycle
from freechat_harness_adapters import ADAPTERS, HarnessCall, ReuseForecast, adapter_for


def forecast() -> ReuseForecast:
    now = datetime.now(UTC)
    return ReuseForecast(
        "call-1",
        0.7,
        5000,
        now,
        now + timedelta(minutes=5),
        "fixture-only",
        "task",
        "session",
        "agent",
    )


@pytest.mark.parametrize("harness", sorted(ADAPTERS))
@pytest.mark.parametrize("lifecycle", [Lifecycle.ACTIVE, Lifecycle.TOOL_WAIT, Lifecycle.RESUME])
def test_lifecycle_does_not_invent_future_reuse(harness: str, lifecycle: Lifecycle) -> None:
    call = HarnessCall("task", "session", "agent", lifecycle=lifecycle, expected_resume_ms=30000)
    hints = adapter_for(harness).hints(call)
    assert hints.expected_reuse_probability == 0
    assert hints.metadata["reuse_forecast_status"] == "unavailable"


@pytest.mark.parametrize("harness", sorted(ADAPTERS))
def test_explicit_forecast_is_call_scoped_and_traceable(harness: str) -> None:
    call = HarnessCall("task", "session", "agent", call_id="call-1", reuse_forecast=forecast())
    hints = adapter_for(harness).hints(call)
    assert hints.expected_reuse_probability == 0.7
    assert hints.expected_resume_ms == 5000
    assert hints.metadata["reuse_forecast_evidence_reference"] == "fixture-only"
    assert (
        adapter_for(harness).hints(replace(call, call_id="call-2")).expected_reuse_probability == 0
    )
    for terminal in (Lifecycle.TERMINAL, Lifecycle.CANCELLED):
        assert (
            adapter_for(harness).hints(replace(call, lifecycle=terminal)).expected_reuse_probability
            == 0
        )


def test_expired_and_future_forecast_are_unavailable() -> None:
    item = forecast()
    call = HarnessCall("task", "session", "agent", call_id="call-1")
    assert not item.applicable(call, item.expires_at)
    assert not item.applicable(call, item.observed_at - timedelta(seconds=1))
    assert item.applicable(call, item.observed_at)
    for other in (
        replace(call, task_id="another"),
        replace(call, session_id="another"),
        replace(call, agent_id="another"),
        replace(call, branch_id="another"),
    ):
        assert not item.applicable(other, item.observed_at)


@pytest.mark.parametrize("probability", [-1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        replace(forecast(), probability=probability)


def test_unbounded_or_naive_forecast_rejected() -> None:
    item = forecast()
    with pytest.raises(ValueError):
        replace(item, expires_at=item.observed_at + timedelta(hours=2))
    with pytest.raises(ValueError):
        replace(item, observed_at=item.observed_at.replace(tzinfo=None))
