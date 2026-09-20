from datetime import UTC, datetime, timedelta

from freechat_harness_adapters import HarnessCall, ReuseForecast, adapter_for
from freechat_scheduler.forecasts import forecast_rejection


def test_scheduler_revalidates_forecast_after_transport() -> None:
    now = datetime.now(UTC)
    forecast = ReuseForecast(
        "call", 0.7, 1000, now, now + timedelta(seconds=1), "fixture", "task", "session", "agent"
    )
    hints = adapter_for("langgraph").hints(
        HarnessCall(
            "task",
            "session",
            "agent",
            call_id="call",
            reuse_forecast=forecast,
        )
    )
    assert forecast_rejection(hints, now) is None
    assert forecast_rejection(hints, now + timedelta(seconds=2)) == "reuse_forecast_not_current"
    hints.task_id = "other"
    assert forecast_rejection(hints, now) == "reuse_forecast_scope_mismatch"


def test_unavailable_prediction_cannot_authorize_offload() -> None:
    hints = adapter_for("opencode").hints(HarnessCall("task", "session", "agent"))
    hints.expected_reuse_probability = 1
    assert forecast_rejection(hints, datetime.now(UTC)) == "reuse_forecast_unavailable"
