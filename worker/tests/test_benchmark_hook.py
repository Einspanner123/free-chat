from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from freechat_worker.benchmark_hook import JsonlCacheEventHook


@dataclass(frozen=True)
class Metadata:
    tenant_id: str = "tenant-a"
    cache_key: str = "cache-a"
    task_id: str = "task-a"
    session_id: str = "session-a"
    agent_id: str = "agent-a"
    branch_id: str = "main"
    call_id: str = "call-a"
    lifecycle: str = "resume"
    worker_generation: int = 1
    cache_generation: int = 2
    expected_resume_ms: int = 100
    priority: int = 4
    allow_kv_offload: bool = True


@dataclass(frozen=True)
class Event:
    event_type: str = "hit"
    request_id: str = "request-a"
    reason: str = "prefix_cache_hit"
    block_ids: tuple[tuple[int, ...], ...] = ((1, 2),)
    metadata: Metadata | None = Metadata()


def test_jsonl_hook_records_correlated_cache_evidence(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    JsonlCacheEventHook(output).on_cache_event(Event())

    record = json.loads(output.read_text())
    assert record["event_type"] == "hit"
    assert record["block_ids"] == [[1, 2]]
    assert record["metadata"]["task_id"] == "task-a"
    assert record["metadata"]["lifecycle"] == "resume"
    assert output.stat().st_mode & 0o777 == 0o600
