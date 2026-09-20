"""Exercise running Gateway/Scheduler/GPU Worker services; stdout and service logs only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen")
    asyncio.run(validate(parser.parse_args()))


if __name__ == "__main__":
    main()
