"""Native rendering budgets shared by scheduler and worker; no prompt storage."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def body_digest(body: dict[str, Any]) -> str:
    # Scheduling metadata is supplied after preparation and never rendered as text.
    return digest({key: value for key, value in body.items() if key != "agent_lifecycle"})


class TokenBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    protocol: str
    body_sha256: str
    prompt_sha256: str
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    expires_at: float = Field(gt=0, allow_inf_nan=False)

    def check_execution(self, prompt: Any, sampling_params: Any, now: float) -> None:
        if (
            now >= self.expires_at
            or not isinstance(prompt, dict)
            or digest(prompt.get("prompt_token_ids")) != self.prompt_sha256
            or getattr(sampling_params, "n", None) != 1
            or getattr(sampling_params, "max_tokens", 0) < 1
            or sampling_params.max_tokens > self.output_tokens
        ):
            raise ValueError("execution_does_not_match_prepared_budget")


class PreparedAdmission(BaseModel):
    """An opaque preparation belongs to exactly one live engine incarnation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preparation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_id: str = Field(min_length=1)
    generation: int = Field(gt=0)
    engine_instance_id: str = Field(min_length=1)
    budget: TokenBudget
