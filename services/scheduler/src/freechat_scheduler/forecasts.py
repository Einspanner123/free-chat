from datetime import datetime

from freechat_contracts import AgentHints


def forecast_rejection(hints: AgentHints, now: datetime) -> str | None:
    """Revalidate adapter forecasts after transport; provenance is not authentication."""
    status = hints.metadata.get("reuse_forecast_status")
    if status is None:
        return None  # Preserve the existing explicit-hints API contract.
    if status != "caller_supplied":
        return "reuse_forecast_unavailable"
    metadata = hints.metadata
    for name in ("task_id", "session_id", "agent_id", "branch_id", "call_id"):
        if metadata.get(f"reuse_forecast_{name}") != getattr(hints, name):
            return "reuse_forecast_scope_mismatch"
    if not metadata.get("reuse_forecast_evidence_reference", "").strip():
        return "reuse_forecast_missing_reference"
    try:
        observed = datetime.fromisoformat(metadata["reuse_forecast_observed_at"])
        expires = datetime.fromisoformat(metadata["reuse_forecast_expires_at"])
        if observed.tzinfo is None or expires.tzinfo is None:
            return "reuse_forecast_invalid_time"
        if not 0 < (expires - observed).total_seconds() <= 3600:
            return "reuse_forecast_invalid_time"
        if not observed <= now < expires:
            return "reuse_forecast_not_current"
    except (KeyError, ValueError):
        return "reuse_forecast_invalid_time"
    return None
