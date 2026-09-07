from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COUNTER_PATTERN = re.compile(r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$")


@dataclass(slots=True)
class Endpoint:
    name: str
    url: str
    gpu_id: str
    active_requests: int = 0


@dataclass(frozen=True, slots=True)
class StreamResult:
    started_ms: float
    first_token_ms: float
    finished_ms: float
    input_tokens: int
    output_tokens: int
    content: str


def prometheus_counter(payload: str, metric: str) -> float:
    values: list[float] = []
    for line in payload.splitlines():
        match = COUNTER_PATTERN.match(line)
        if match is not None and match.group("name") == metric:
            values.append(float(match.group("value")))
    if not values:
        raise ValueError(f"metric {metric!r} is missing")
    return sum(values)


def select_endpoint(strategy: str, endpoints: list[Endpoint], turn: int) -> Endpoint:
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    if strategy == "direct-vllm":
        if len(endpoints) != 1:
            raise ValueError("direct-vllm requires exactly one endpoint")
        return endpoints[0]
    if strategy == "round-robin":
        return endpoints[turn % len(endpoints)]
    if strategy == "least-load":
        return min(endpoints, key=lambda item: (item.active_requests, item.name))
    raise ValueError(f"unsupported baseline strategy: {strategy}")


def get_text(url: str, timeout_seconds: float) -> str:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return str(response.read().decode("utf-8"))


def stream_chat(
    endpoint: Endpoint,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: float,
) -> StreamResult:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{endpoint.url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ms = time.perf_counter_ns() / 1_000_000
    first_token_ms: float | None = None
    input_tokens = 0
    output_tokens = 0
    fragments: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            encoded = line.removeprefix("data:").strip()
            if not encoded or encoded == "[DONE]":
                continue
            event = json.loads(encoded)
            usage = event.get("usage") or {}
            input_tokens = max(input_tokens, int(usage.get("prompt_tokens") or 0))
            output_tokens = max(output_tokens, int(usage.get("completion_tokens") or 0))
            choices = event.get("choices") or []
            for choice in choices:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if first_token_ms is None:
                        first_token_ms = time.perf_counter_ns() / 1_000_000
                    fragments.append(str(content))
    finished_ms = time.perf_counter_ns() / 1_000_000
    return StreamResult(
        started_ms=started_ms,
        first_token_ms=first_token_ms or finished_ms,
        finished_ms=finished_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        content="".join(fragments),
    )


class GpuSampler:
    def __init__(self, run_id: str, gpu_index: str, interval_seconds: float) -> None:
        self._run_id = run_id
        self._gpu_index = gpu_index
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.samples: list[dict[str, Any]] = []

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _sample(self) -> None:
        while not self._stop.is_set():
            observed_ms = time.perf_counter_ns() / 1_000_000
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-gpu=uuid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            if completed.returncode == 0:
                gpu_uuid, utilization = completed.stdout.strip().split(",", maxsplit=1)
                self.samples.append(
                    {
                        "run_id": self._run_id,
                        "gpu_id": gpu_uuid.strip(),
                        "observed_ms": observed_ms,
                        "sample_interval_ms": self._interval_seconds * 1_000,
                        "utilization_percent": float(utilization.strip()),
                    }
                )
            self._stop.wait(self._interval_seconds)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    endpoints = [
        Endpoint(name=name, url=url, gpu_id=gpu)
        for name, url, gpu in (item.split(",", maxsplit=2) for item in arguments.endpoint)
    ]
    if arguments.concurrency != 1:
        raise ValueError(
            "the baseline collector currently requires concurrency=1 so Prometheus "
            "cache-counter deltas remain attributable to individual requests"
        )
    for endpoint in endpoints:
        get_text(f"{endpoint.url.rstrip('/')}/health", arguments.timeout)

    shared_prefix = " ".join(
        [
            "You are a coding agent. Preserve requirements, cite tool outputs, and make no "
            "unsupported assumptions."
        ]
        * arguments.prefix_repetitions
    )
    records: list[dict[str, Any]] = []
    route_turn = 0
    with GpuSampler(arguments.run_id, arguments.gpu_index, arguments.sample_interval) as sampler:
        for task_index in range(arguments.tasks):
            task_id = f"task-{task_index:04d}"
            messages = [
                {"role": "system", "content": shared_prefix},
                {
                    "role": "user",
                    "content": f"Inspect file {task_index}. Return the next tool action only.",
                },
            ]
            for phase in ("active", "resume"):
                endpoint = select_endpoint(arguments.strategy, endpoints, route_turn)
                route_turn += 1
                before = get_text(f"{endpoint.url.rstrip('/')}/metrics", arguments.timeout)
                before_hits = prometheus_counter(before, "vllm:prefix_cache_hits_total")
                endpoint.active_requests += 1
                try:
                    result = stream_chat(
                        endpoint,
                        model=arguments.model,
                        messages=messages,
                        max_tokens=arguments.max_tokens,
                        timeout_seconds=arguments.timeout,
                    )
                finally:
                    endpoint.active_requests -= 1
                after = get_text(f"{endpoint.url.rstrip('/')}/metrics", arguments.timeout)
                after_hits = prometheus_counter(after, "vllm:prefix_cache_hits_total")
                cached_tokens = max(0, round(after_hits - before_hits))
                records.append(
                    {
                        "run_id": arguments.run_id,
                        "trial_id": arguments.trial_id,
                        "workload_id": "coding-agent-tool-resume-baseline",
                        "strategy": arguments.strategy,
                        "model_revision": arguments.model_revision,
                        "task_id": task_id,
                        "harness_id": "custom-loop",
                        "phase": phase,
                        "endpoint": endpoint.name,
                        "started_ms": result.started_ms,
                        "finished_ms": result.finished_ms,
                        "ttft_ms": result.first_token_ms - result.started_ms,
                        "input_tokens": result.input_tokens,
                        "cached_tokens": cached_tokens,
                        "output_tokens": result.output_tokens,
                        "success": True,
                    }
                )
                if phase == "active":
                    messages.extend(
                        [
                            {"role": "assistant", "content": result.content},
                            {
                                "role": "user",
                                "content": (
                                    f"Tool result for file {task_index}: no conflicts found."
                                ),
                            },
                        ]
                    )
        gpu_samples = list(sampler.samples)
    if not gpu_samples:
        raise RuntimeError("GPU sampler produced no observations")
    return {
        "schema": 1,
        # The collector may run on a GPU host whose control Python predates 3.11.
        "observed_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "run": {
            "run_id": arguments.run_id,
            "trial_id": arguments.trial_id,
            "strategy": arguments.strategy,
            "model": arguments.model,
            "model_revision": arguments.model_revision,
            "tasks": arguments.tasks,
            "concurrency": arguments.concurrency,
            "prefix_repetitions": arguments.prefix_repetitions,
            "max_tokens": arguments.max_tokens,
            "endpoints": [asdict(item) for item in endpoints],
        },
        "requests": records,
        "gpu_samples": gpu_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-matched direct serving baseline")
    parser.add_argument("--endpoint", action="append", required=True, help="name,url,gpu-id")
    parser.add_argument(
        "--strategy",
        choices=("direct-vllm", "round-robin", "least-load"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu-index", required=True)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prefix-repetitions", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["run"], indent=2))


if __name__ == "__main__":
    main()
