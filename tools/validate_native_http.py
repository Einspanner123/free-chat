"""Exercise managed native HTTP protocols and execution RPC; stdout only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import grpc
import httpx
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionReceipt
from freechat_worker.capacity import EngineCapacity


def has_token(event: dict[str, Any]) -> bool:
    if event.get("choices"):
        return bool(event["choices"][0].get("delta", {}).get("content"))
    if event.get("type") == "response.output_text.delta":
        return bool(event.get("delta"))
    delta = event.get("delta")
    return isinstance(delta, dict) and bool(delta.get("text"))


async def validate(url: str, control: str, model: str, token: str) -> None:
    if len(token) < 32:
        raise ValueError("FREECHAT_WORKER_TOKEN is required")
    async with httpx.AsyncClient(base_url=url, timeout=120, trust_env=False) as client:
        auth = {"x-freechat-worker-token": token}
        identity_response = await client.get("/freechat/runtime", headers=auth)
        identity_response.raise_for_status()
        identity = identity_response.json()
        capacity = EngineCapacity.model_validate(identity["capacity"])
        assert capacity.usable_bytes > 0 and capacity.bytes_per_token > 0
        print(json.dumps({"scope": "ENGINE_KV_CAPACITY", **identity}), flush=True)
        async with grpc.aio.insecure_channel(control) as channel:
            stub = control_pb2_grpc.RequestExecutionServiceStub(channel)  # type: ignore[no-untyped-call]

            async def observe(command: ExecutionCommand) -> ExecutionReceipt:
                response = await stub.Observe(
                    control_pb2.RequestExecutionCommand(command_json=command.model_dump_json()),
                    metadata=(("authorization", f"Bearer {token}"),),
                    timeout=10,
                )
                return ExecutionReceipt.model_validate_json(response.receipt_json)

            async def terminal(command: ExecutionCommand) -> ExecutionReceipt:
                async with asyncio.timeout(15):
                    while True:
                        receipt = await observe(command)
                        if receipt.releasable:
                            return receipt
                        await asyncio.sleep(0.1)

            for protocol in ("chat/completions", "responses", "messages"):
                for mode in ("json", "stream", "cancel"):
                    request_id = str(uuid4())
                    command = ExecutionCommand(
                        tenant_id="validation",
                        request_id=request_id,
                        decision_id=request_id,
                        worker_id=identity["worker_id"],
                        worker_generation=identity["generation"],
                        engine_instance_id=identity["engine_instance_id"],
                        action=ExecutionAction.QUERY,
                    )
                    headers = {
                        **auth,
                        "x-freechat-internal-tenant": command.tenant_id,
                        "x-freechat-internal-request-id": request_id,
                        "x-freechat-internal-decision-id": request_id,
                        "x-freechat-internal-worker-generation": str(command.worker_generation),
                        "x-freechat-internal-engine-instance-id": command.engine_instance_id,
                    }
                    limit = 512 if mode == "cancel" else 16
                    body: dict[str, Any] = {"model": model, "stream": mode != "json"}
                    if protocol == "responses":
                        body.update(
                            input="Describe the solar system in detail.", max_output_tokens=limit
                        )
                    else:
                        body.update(
                            messages=[
                                {"role": "user", "content": "Describe the solar system in detail."}
                            ],
                            max_tokens=limit,
                        )
                    endpoint = f"/v1/{protocol}"
                    prepared = await client.post(
                        "/freechat/prepare", headers=headers,
                        json={"protocol": endpoint, "request": body},
                    )
                    prepared.raise_for_status()
                    preparation = prepared.json()
                    headers["x-freechat-internal-preparation"] = preparation["preparation_id"]
                    budget = preparation["budget"]
                    total_tokens = budget["input_tokens"] + budget["output_tokens"]
                    headers["x-freechat-internal-reserved-kv-bytes"] = str(
                        (total_tokens + capacity.block_size_tokens - 1)
                        // capacity.block_size_tokens * capacity.block_bytes
                    )
                    seen = False
                    if mode == "json":
                        response = await client.post(endpoint, headers=headers, json=body)
                        response.raise_for_status()
                        result = response.json()
                        usage = result["usage"]
                        assert usage.get("prompt_tokens", usage.get("input_tokens")) == (
                            preparation["budget"]["input_tokens"]
                        ), "native usage differs from prepared token count"
                    else:
                        async with client.stream(
                            "POST", endpoint, headers=headers, json=body
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line.startswith("data: ") or line == "data: [DONE]":
                                    continue
                                event = json.loads(line[6:])
                                seen |= has_token(event)
                                if mode == "cancel" and seen:
                                    await observe(
                                        command.model_copy(update={"action": ExecutionAction.ABORT})
                                    )
                                    break
                        assert seen, "no generated text received"
                    receipt = await terminal(command)
                    duplicate = await client.post(endpoint, headers=headers, json=body)
                    assert duplicate.status_code == 409, duplicate.text
                    print(
                        json.dumps(
                            {
                                "scope": "NATIVE_HTTP_AND_EXECUTION_RPC",
                                "protocol": protocol,
                                "mode": mode,
                                "status": receipt.status.value,
                                "releasable": receipt.releasable,
                                "prepared_input_tokens": preparation["budget"]["input_tokens"],
                                "duplicate_status": duplicate.status_code,
                                "worker": identity,
                            }
                        ),
                        flush=True,
                    )
        denied = await client.post("/v1/chat/completions", json={"model": model})
        assert denied.status_code == 401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--control", default="127.0.0.1:50052")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(
        validate(args.url, args.control, args.model, os.environ.get("FREECHAT_WORKER_TOKEN", ""))
    )


if __name__ == "__main__":
    main()
