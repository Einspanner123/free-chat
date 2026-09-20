"""Internal development control authentication; not a replacement for mTLS."""

from __future__ import annotations

import hmac
from typing import Any

import grpc


async def authenticate_control(context: Any, token: str | None) -> None:
    # None is reserved for explicitly constructed CPU contract fixtures.
    if token is None:
        return
    values = [v for k, v in context.invocation_metadata() if k == "authorization"]
    if len(values) != 1 or not hmac.compare_digest(values[0], f"Bearer {token}"):
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "control_identity_required")
