"""Exercise running Gateway/Scheduler/GPU Worker services; stdout and service logs only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Iterable
from typing import Any

import httpx

from tools.validate_native_http import has_token


def body_for(protocol: str, *, stream: bool, limit: int, pressure: bool = False) -> dict[str, Any]:
    prompt = (
        "Reply with the single word OK." if pressure else "Describe the solar system in detail."
    )
    body: dict[str, Any] = {"stream": stream}
    if protocol == "responses":
        body.update(input=prompt, max_output_tokens=limit)
    else:
        body.update(messages=[{"role": "user", "content": prompt}], max_tokens=limit)
    if pressure:
        body["stop"] = ["OK"]  # Full output budget, a deliberately short actual response.
    return body


async def validate(args: argparse.Namespace) -> None:
    worker_token = os.environ["FREECHAT_WORKER_TOKEN"]
    api_key = os.environ["FREECHAT_VALIDATION_API_KEY"]
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        async with asyncio.timeout(180):
            while True:
                try:
                    runtime_response = await client.get(
                        args.worker_url + "/freechat/runtime",
                        headers={"x-freechat-worker-token": worker_token},
                    )
                    runtime_response.raise_for_status()
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(1)

        runtime = runtime_response.json()
        capacity = runtime["capacity"]
        pool = (capacity["num_blocks"] - 1) * capacity["block_bytes"]
        headers = {"authorization": f"Bearer {api_key}"}
        unauthorized = await client.post(args.url + "/v1/responses", json={})
        assert unauthorized.status_code == 401
        # Worker startup may precede control startup; readiness must become real.
        async with asyncio.timeout(90):
            while True:
                warmup = await client.post(
                    args.url + "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": args.model,
                        **body_for("chat/completions", stream=False, limit=4),
                    },
                )
                if warmup.status_code == 200:
                    break
                if warmup.status_code != 503:
                    warmup.raise_for_status()
                await asyncio.sleep(1)
        if args.mode == "concurrent":
            await concurrent_probe(client, args, headers, runtime, pool)
            return
        for protocol in ("chat/completions", "responses", "messages"):
            for mode in ("json", "stream", "disconnect"):
                body = {
                    "model": args.model,
                    **body_for(
                        protocol,
                        stream=mode != "json",
                        limit=512 if mode == "disconnect" else 16,
                    ),
                }
                url = args.url + "/v1/" + protocol
                if mode == "json":
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    assert response.json().get("usage")
                else:
                    seen = False
                    async with client.stream("POST", url, headers=headers, json=body) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                seen |= has_token(json.loads(line[6:]))
                                if seen and mode == "disconnect":
                                    break
                    assert seen, (protocol, mode, "no generated text")
                assert int(response.headers["x-freechat-reserved-kv-bytes"]) > 0
                print(
                    json.dumps(
                        {
                            "scope": "SINGLE_NODE_GPU_INFERENCE_LOOP",
                            "worker": runtime["worker_id"],
                            "protocol": protocol,
                            "mode": mode,
                            "decision_id": response.headers["x-freechat-decision-id"],
                            "http_status": response.status_code,
                        }
                    ),
                    flush=True,
                )
        total, calls = 0, 0
        body = {
            "model": args.model,
            **body_for(
                "chat/completions",
                stream=False,
                limit=capacity["max_context_tokens"] - 128,
                pressure=True,
            ),
        }
        while total <= pool:
            if calls >= 1000:
                raise ValueError("capacity reuse probe exceeded bounded call count")
            response = await client.post(
                args.url + "/v1/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            reserved = int(response.headers["x-freechat-reserved-kv-bytes"])
            assert reserved > 0
            total += reserved
            calls += 1
        print(
            json.dumps(
                {
                    "scope": "SEQUENTIAL_CAPACITY_REUSE",
                    "worker": runtime["worker_id"],
                    "completed_calls": calls,
                    "cumulative_reserved_bytes": total,
                    "physical_usable_kv_bytes": pool,
                    "exceeds_one_pool": total > pool,
                    "claim_boundary": "not concurrent throughput, crash recovery, or HA",
                }
            ),
            flush=True,
        )


async def concurrent_probe(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    headers: dict[str, str],
    runtime: dict[str, Any],
    pool: int,
) -> None:
    capacity = runtime["capacity"]
    limit = min(800, capacity["max_context_tokens"] - 128)
    if (
        limit < 128
        or args.concurrent_requests
        * limit
        * (capacity["block_bytes"] // capacity["block_size_tokens"])
        <= pool
    ):
        raise ValueError("use a smaller real KV pool to exercise concurrent capacity rejection")
    routes, rejected = 1, 0  # Includes the successful readiness request.
    for round_index in range(args.rounds):
        body = {
            "model": args.model,
            **body_for("chat/completions", stream=False, limit=limit),
            "ignore_eos": True,
        }
        before_rejected = rejected
        # HTTP completion precedes confirmed release. Retry real backpressure,
        # count every rejection, and bound the wait instead of assuming instant reuse.
        async with asyncio.timeout(30):
            while True:
                replies = await asyncio.gather(
                    *(
                        client.post(args.url + "/v1/chat/completions", headers=headers, json=body)
                        for _ in range(args.concurrent_requests)
                    )
                )
                statuses = [response.status_code for response in replies]
                if not set(statuses) <= {200, 503} or 503 not in statuses:
                    raise ValueError(f"expected admission/backpressure: {statuses}")
                for response in replies:
                    if response.status_code == 200:
                        assert response.json()["usage"]["completion_tokens"] == limit
                        assert int(response.headers["x-freechat-reserved-kv-bytes"]) > 0
                        routes += 1
                    else:
                        assert "x-freechat-decision-id" not in response.headers
                        rejected += 1
                if 200 in statuses:
                    break
                await asyncio.sleep(0.2)
        # Completion must restore admission; do not require zero reconciliation latency.
        async with asyncio.timeout(20):
            while True:
                recovery = await client.post(
                    args.url + "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": args.model,
                        **body_for("chat/completions", stream=False, limit=4),
                    },
                )
                if recovery.status_code == 200:
                    routes += 1
                    break
                if recovery.status_code != 503:
                    recovery.raise_for_status()
                rejected += 1
                await asyncio.sleep(0.1)
        print(
            json.dumps(
                {
                    "scope": "CONCURRENT_CAPACITY",
                    "worker": runtime["worker_id"],
                    "round": round_index,
                    "accepted": statuses.count(200),
                    "backpressure_including_retries": rejected - before_rejected,
                    "recovery": 200,
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "scope": "AUDIT_REQUIRED",
                "worker": runtime["worker_id"],
                "physical_usable_kv_bytes": pool,
                "expected_routes": routes,
                "expected_cancels": 0,
                "expected_capacity_rejections": rejected,
                "claim_boundary": "audit Scheduler logs; not throughput or recovery-time evidence",
            }
        ),
        flush=True,
    )


def audit_log(
    lines: Iterable[str],
    *,
    pool: int,
    worker: str,
    expected_routes: int,
    expected_cancels: int,
    expected_rejections: int,
) -> dict[str, Any]:
    """Check a complete single-incarnation Scheduler log, without writing artifacts."""
    if pool <= 0 or expected_routes <= 0:
        raise ValueError("positive pool and expected route count required")
    seen: dict[str, dict[str, Any]] = {}
    routes: dict[str, dict[str, Any]] = {}
    outstanding: dict[str, int] = {}
    released: set[str] = set()
    cancelled: set[str] = set()
    rejection_ids: set[str] = set()
    peak = 0
    for line in lines:
        if "admission_rejected " in line:
            rejection = json.loads(line.split("admission_rejected ", 1)[1])
            if rejection["rejected"].get(worker) == ["vram_capacity"]:
                rejection_ids.add(rejection["request_id"])
        if "lifecycle_event " not in line:
            continue
        event = json.loads(line.split("lifecycle_event ", 1)[1])
        payload = event["payload"]
        if payload["worker_id"] != worker:
            raise ValueError("audit requires an isolated single-Worker log")
        if event["event_id"] in seen:
            if event != seen[event["event_id"]]:
                raise ValueError("conflicting duplicate event")
            continue  # Identical transport redelivery is not duplicate accounting.
        seen[event["event_id"]] = event
        key, kind = event["aggregate_id"], event["event_type"]
        if kind == "route.decided":
            if key in routes:
                raise ValueError("duplicate route identity")
            routes[key] = event
            amount = payload["reserved_kv_bytes_per_rank"]
            if not isinstance(amount, int) or amount <= 0:
                raise ValueError("invalid reservation")
            outstanding[key] = amount
            peak = max(peak, sum(outstanding.values()))
            if peak > pool:
                raise ValueError("physical KV pool oversubscribed")
        elif kind == "lease.cancel_requested":
            if key not in outstanding or key in cancelled:
                raise ValueError("cancel does not match an outstanding route")
            cancelled.add(key)
        elif kind == "lease.released":
            if key not in outstanding or key in released:
                raise ValueError("duplicate or unowned release")
            receipt = payload["execution_receipt"]
            command = receipt["command"]
            original = routes[key]
            for field, wanted in {
                "decision_id": key,
                "request_id": original["payload"]["request_id"],
                "tenant_id": original["tenant_id"],
                "worker_id": worker,
                "worker_generation": original["aggregate_generation"],
                "engine_instance_id": original["payload"]["decision"]["engine_instance_id"],
            }.items():
                if command[field] != wanted:
                    raise ValueError("receipt identity mismatch")
            if not (receipt["quiescent"] is True and receipt["admission_closed"] is True):
                raise ValueError("release without execution fence")
            if receipt["status"] != ("aborted" if key in cancelled else "completed"):
                raise ValueError("unexpected terminal status")
            if key in cancelled and command["action"] != "abort":
                raise ValueError("cancel lacks Worker abort confirmation")
            del outstanding[key]
            released.add(key)
    if (len(routes), len(cancelled), len(rejection_ids)) != (
        expected_routes,
        expected_cancels,
        expected_rejections,
    ):
        raise ValueError("observed route/cancel/capacity-rejection counts differ from probe")
    incarnations = {e["aggregate_generation"] for e in routes.values()}
    if len(incarnations) != 1 or outstanding or len(released) != len(routes):
        raise ValueError("incomplete closure or mixed Worker incarnations")
    return {
        "scope": "SINGLE_WORKER_LIFECYCLE_LOG_AUDIT",
        "worker": worker,
        "routes": len(routes),
        "releases": len(released),
        "cancel_intents": len(cancelled),
        "capacity_rejections": len(rejection_ids),
        "peak_reserved_bytes": peak,
        "physical_usable_kv_bytes": pool,
        "unreleased": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen")
    parser.add_argument(
        "--mode", choices=("sequential", "concurrent", "audit-log"), default="sequential"
    )
    parser.add_argument("--concurrent-requests", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--worker")
    parser.add_argument("--pool-bytes", type=int, default=0)
    parser.add_argument("--expected-routes", type=int, default=0)
    parser.add_argument("--expected-cancels", type=int, default=0)
    parser.add_argument("--expected-rejections", type=int, default=0)
    args = parser.parse_args()
    if not 2 <= args.concurrent_requests <= 64 or not 1 <= args.rounds <= 20:
        parser.error("concurrent requests must be 2..64 and rounds 1..20")
    if args.mode == "audit-log":
        if not args.worker:
            parser.error("--worker is required for log audit")
        print(
            json.dumps(
                audit_log(
                    sys.stdin,
                    pool=args.pool_bytes,
                    worker=args.worker,
                    expected_routes=args.expected_routes,
                    expected_cancels=args.expected_cancels,
                    expected_rejections=args.expected_rejections,
                )
            ),
            flush=True,
        )
    else:
        asyncio.run(validate(args))


if __name__ == "__main__":
    main()
