"""Structured validation records for stdout and normal log collection, never result files."""

from __future__ import annotations

import hashlib
import json


def emit_record(name: str, payload: str) -> dict[str, str]:
    reference = {"name": name, "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    print(
        json.dumps({"record_type": "validation_artifact", **reference, "payload": payload}),
        flush=True,
    )
    return reference
