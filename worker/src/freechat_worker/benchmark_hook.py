"""Append vLLM cache lifecycle events to an audit JSONL file.

This hook is intentionally benchmark-only. The serving benchmark loads it in
the engine-core process through ``VLLM_AGENT_CACHE_HOOK`` so cache hits can be
joined to request measurements without inferring them from latency.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


class JsonlCacheEventHook:
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_cache_event(self, event: Any) -> None:
        metadata = getattr(event, "metadata", None)
        record = {
            "schema_version": 1,
            "observed_monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "event_type": str(getattr(event, "event_type", "")),
            "request_id": getattr(event, "request_id", None),
            "reason": str(getattr(event, "reason", "")),
            "block_ids": [list(group) for group in getattr(event, "block_ids", ())],
            "metadata": (
                {
                    "tenant_id": metadata.tenant_id,
                    "cache_key": metadata.cache_key,
                    "task_id": metadata.task_id,
                    "session_id": metadata.session_id,
                    "agent_id": metadata.agent_id,
                    "branch_id": metadata.branch_id,
                    "call_id": metadata.call_id,
                    "lifecycle": str(metadata.lifecycle),
                    "worker_generation": metadata.worker_generation,
                    "cache_generation": metadata.cache_generation,
                    "expected_resume_ms": metadata.expected_resume_ms,
                    "priority": metadata.priority,
                    "allow_kv_offload": metadata.allow_kv_offload,
                }
                if metadata is not None
                else None
            ),
        }
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        descriptor = os.open(
            self._output_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, payload)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def create_jsonl_cache_hook() -> JsonlCacheEventHook:
    value = os.getenv("FREECHAT_CACHE_EVENT_JSONL")
    if not value:
        raise RuntimeError("FREECHAT_CACHE_EVENT_JSONL is required")
    return JsonlCacheEventHook(Path(value))
