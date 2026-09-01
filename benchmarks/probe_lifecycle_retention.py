"""Probe lifecycle cache retention under deterministic prefix pressure.

The probe may use vLLM's dummy load format to validate the serving mechanism.
Its output is permanently labelled ``MECHANISM_ONLY`` and is not admissible as
a model-quality or performance claim.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    call_id: str
    phase: str
    response_id: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float


def _metadata(
    *,
    task_id: str,
    call_id: str,
    lifecycle: str,
    expected_resume_ms: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "tenant_id": "mechanism-tenant",
        "cache_key": f"mechanism-cache:{task_id}",
        "task_id": task_id,
        "session_id": f"session:{task_id}",
        "agent_id": f"agent:{task_id}",
        "branch_id": "main",
        "call_id": call_id,
        "lifecycle": lifecycle,
        "worker_generation": 1,
        "cache_generation": 1,
        "priority": 100 if lifecycle in {"tool_wait", "resume"} else 0,
        "allow_kv_offload": True,
    }
    if expected_resume_ms is not None:
        value["expected_resume_ms"] = expected_resume_ms
    return value


def _request(
    client: httpx.Client,
    *,
    server_url: str,
    model: str,
    prompt: str,
    task_id: str,
    call_id: str,
    phase: str,
    expected_resume_ms: int | None = None,
) -> ProbeRequest:
    started = time.perf_counter_ns()
    response = client.post(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
            "agent_lifecycle": _metadata(
                task_id=task_id,
                call_id=call_id,
                lifecycle=phase,
                expected_resume_ms=expected_resume_ms,
            ),
        },
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if response.is_error:
        raise RuntimeError(f"serving request failed ({response.status_code}): {response.text}")
    payload = response.json()
    usage = payload["usage"]
    return ProbeRequest(
        call_id=call_id,
        phase=phase,
        response_id=payload["id"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        elapsed_ms=elapsed_ms,
    )


def _events_for_call(path: Path, call_id: str) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        metadata = event.get("metadata")
        if metadata is not None and metadata.get("call_id") == call_id:
            events.append(event)
    return events


def _hit_tokens(events: list[dict[str, Any]], block_tokens: int) -> int:
    hits = [event for event in events if event["event_type"] == "hit"]
    if len(hits) > 1:
        raise ValueError("a probe call emitted more than one cache-hit event")
    return (
        sum(len(group) for group in hits[0]["block_ids"]) * block_tokens
        if hits
        else 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="qwen-mechanism")
    parser.add_argument("--cache-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-repetitions", type=int, default=100)
    parser.add_argument("--pressure-requests", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.prefix_repetitions < 1 or args.pressure_requests < 1:
        parser.error("prefix repetitions and pressure requests must be positive")
    if args.block_tokens < 1:
        parser.error("block tokens must be positive")

    stable = "alpha beta gamma delta epsilon zeta eta theta " * args.prefix_repetitions
    requests = []
    with httpx.Client(timeout=120, trust_env=False) as client:
        requests.append(
            _request(
                client,
                server_url=args.server_url,
                model=args.model,
                prompt=stable,
                task_id="target",
                call_id="target:wait",
                phase="tool_wait",
                expected_resume_ms=120_000,
            )
        )
        for index in range(args.pressure_requests):
            requests.append(
                _request(
                    client,
                    server_url=args.server_url,
                    model=args.model,
                    prompt=(
                        f"pressure{index} alpha beta gamma delta epsilon zeta eta "
                        * args.prefix_repetitions
                    ),
                    task_id=f"pressure:{index}",
                    call_id=f"pressure:{index}:active",
                    phase="active",
                )
            )
        requests.append(
            _request(
                client,
                server_url=args.server_url,
                model=args.model,
                prompt=stable,
                task_id="target",
                call_id="target:resume",
                phase="resume",
                expected_resume_ms=120_000,
            )
        )

    resume_events = _events_for_call(args.cache_events, "target:resume")
    result = {
        "evidence_level": "MECHANISM_ONLY",
        "performance_claim_admissible": False,
        "reason": "probe does not constitute a real-model paired Harness benchmark",
        "requests": [asdict(item) for item in requests],
        "resume_hit_tokens": _hit_tokens(resume_events, args.block_tokens),
        "resume_events": resume_events,
        "resolved_config": {
            "model": args.model,
            "prefix_repetitions": args.prefix_repetitions,
            "pressure_requests": args.pressure_requests,
            "block_tokens": args.block_tokens,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
