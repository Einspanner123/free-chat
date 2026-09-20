"""Send correlated vLLM cache lifecycle observations to the service logger."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

LOGGER = logging.getLogger("freechat.cache.events")


class LoggingCacheEventHook:
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
        LOGGER.info("FREECHAT_CACHE_EVENT %s", json.dumps(record, sort_keys=True))


def create_cache_log_hook() -> LoggingCacheEventHook:
    return LoggingCacheEventHook()
