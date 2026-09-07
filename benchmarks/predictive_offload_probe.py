from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import grpc
import httpx
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import ModelCapability, WorkerCapabilities, WorkerTelemetry

METRIC_PATTERN = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>\S+)$")
STORE_BYTES = "vllm:kv_offload_store_size_sum"
LOAD_BYTES = "vllm:kv_offload_load_size_sum"


def metric(payload: str, name: str) -> float:
    values = [
        float(match.group("value"))
        for line in payload.splitlines()
        if (match := METRIC_PATTERN.match(line)) is not None
        and match.group("name") == name
    ]
    return sum(values)


def context(request_id: str, tenant_id: str = "system") -> Any:
    return control_pb2.RequestContext(
        request_id=request_id,
        idempotency_key=request_id,
        tenant_id=tenant_id,
        schema_version=1,
    )


async def register_worker(arguments: argparse.Namespace) -> dict[str, Any]:
    capabilities = WorkerCapabilities(
        worker_id=arguments.worker_id,
        generation=arguments.worker_generation,
        endpoint=arguments.worker_endpoint,
        node_id=arguments.node_id,
        gpu_id=arguments.gpu_id,
        gpu_name=arguments.gpu_name,
        compute_capability=arguments.compute_capability,
        total_vram_bytes=arguments.total_vram_bytes,
        p2p_domain=arguments.node_id,
        network_domain="lan",
        models=(
            ModelCapability(
                model_id=arguments.model,
                revision=arguments.model_revision,
                tokenizer_revision=arguments.model_revision,
                architecture="dense",
                attention="gqa",
                max_context_tokens=arguments.max_model_len,
                dtype="float16",
                supports_kv_offload=True,
                kv_bytes_per_token=arguments.kv_bytes_per_token,
            ),
        ),
    )
    telemetry = WorkerTelemetry(
        worker_id=arguments.worker_id,
        generation=arguments.worker_generation,
        free_vram_bytes=arguments.free_vram_bytes,
        kv_cache_capacity_bytes=arguments.kv_cache_capacity_bytes,
        kv_cache_free_bytes=arguments.kv_cache_free_bytes,
        estimated_prefill_tokens_per_second=arguments.prefill_tokens_per_second,
        estimated_decode_tokens_per_second=arguments.decode_tokens_per_second,
        cache_load_bytes_per_second=arguments.cache_load_bytes_per_second,
        cache_store_bytes_per_second=arguments.cache_store_bytes_per_second,
    )
    channel = grpc.aio.insecure_channel(arguments.scheduler)
    try:
        worker = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
        registered = await worker.Register(
            control_pb2.WorkerRegistration(
                context=context("register-predictive-probe"),
                worker_id=capabilities.worker_id,
                generation=capabilities.generation,
                endpoint=capabilities.endpoint,
                capabilities_json=capabilities.model_dump_json(),
            )
        )

        async def heartbeats() -> AsyncIterator[control_pb2.WorkerHeartbeat]:
            yield control_pb2.WorkerHeartbeat(
                context=context("heartbeat-predictive-probe"),
                worker_id=telemetry.worker_id,
                generation=telemetry.generation,
                telemetry_json=telemetry.model_dump_json(),
            )

        heartbeat = await worker.Heartbeat(heartbeats()).read()
        return {
            "registration": registered.status,
            "heartbeat": heartbeat.status,
            "capabilities": capabilities.model_dump(mode="json"),
            "telemetry": telemetry.model_dump(mode="json"),
        }
    finally:
        await channel.close()


async def preview_decision(
    arguments: argparse.Namespace,
    *,
    estimated_input_tokens: int,
) -> dict[str, Any]:
    channel = grpc.aio.insecure_channel(arguments.scheduler)
    try:
        scheduler = control_pb2_grpc.SchedulerServiceStub(channel)  # type: ignore[no-untyped-call]
        request_id = f"preview-{arguments.run_id}"
        route = await scheduler.Route(
            control_pb2.RouteRequest(
                context=context(request_id, tenant_id="predictive-probe"),
                hints=control_pb2.AgentHints(
                    harness_id="predictive-offload-probe",
                    task_id="target-preview",
                    agent_id="probe-agent",
                    branch_id="main",
                    lifecycle="active",
                    prefix_scope="task",
                    reuse_class="growing_history",
                    expected_reuse_probability=1.0,
                    allow_preemption=True,
                    allow_kv_offload=True,
                    allow_remote_worker=True,
                    confidence=1.0,
                    source="explicit",
                ),
                model=arguments.model,
                input_tokens=estimated_input_tokens,
                output_tokens=1,
            )
        )
        directive = route.kv_transfer
        released = await scheduler.Release(
            control_pb2.LeaseRequest(
                context=context(f"{request_id}-release", tenant_id="predictive-probe"),
                decision_id=route.decision_id,
                worker_id=route.worker_id,
                worker_generation=route.worker_generation,
            )
        )
        return {
            "decision_id": route.decision_id,
            "worker_id": route.worker_id,
            "worker_generation": route.worker_generation,
            "applicable": directive.applicable,
            "enabled": directive.enabled,
            "max_offload_tokens": directive.max_offload_tokens,
            "estimated_kv_bytes": directive.estimated_kv_bytes,
            "predicted_reuse_probability": directive.predicted_reuse_probability,
            "predicted_eviction_probability": directive.predicted_eviction_probability,
            "estimated_recompute_ms": directive.estimated_recompute_ms,
            "estimated_store_ms": directive.estimated_store_ms,
            "estimated_restore_ms": directive.estimated_restore_ms,
            "expected_net_benefit_ms": directive.expected_net_benefit_ms,
            "reason": directive.reason,
            "release_status": released.status,
        }
    finally:
        await channel.close()


async def request(
    client: httpx.AsyncClient,
    arguments: argparse.Namespace,
    *,
    task_id: str,
    content: str,
    lifecycle: str,
    expected_reuse_probability: float,
    allow_kv_offload: bool,
    expected_resume_ms: int | None = None,
) -> dict[str, Any]:
    hints: dict[str, Any] = {
        "harness_id": "predictive-offload-probe",
        "task_id": task_id,
        "agent_id": "probe-agent",
        "branch_id": "main",
        "lifecycle": lifecycle,
        "prefix_scope": "task",
        "reuse_class": "growing_history",
        "expected_reuse_probability": expected_reuse_probability,
        "allow_kv_offload": allow_kv_offload,
    }
    if expected_resume_ms is not None:
        hints["expected_resume_ms"] = expected_resume_ms
    started = datetime.now(UTC)
    response = await client.post(
        f"{arguments.gateway.rstrip('/')}/v1/chat/completions",
        headers={"x-api-key": arguments.api_key, "x-request-id": str(uuid4())},
        json={
            "model": arguments.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 1,
            "freechat": {"agent_hints": hints},
        },
    )
    response.raise_for_status()
    return {
        "task_id": task_id,
        "lifecycle": lifecycle,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status_code": response.status_code,
        "decision_id": response.headers.get("x-freechat-decision-id"),
        "offload": response.headers.get("x-freechat-kv-offload"),
        "response_id": response.json().get("id"),
    }


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    control = await register_worker(arguments)
    target = " ".join(
        [f"agent tool context {arguments.run_id} must survive waiting"]
        * arguments.target_repetitions
    )
    control["preview_decision"] = await preview_decision(
        arguments,
        estimated_input_tokens=max(1, len(target) // 2),
    )
    async with httpx.AsyncClient(timeout=arguments.timeout, trust_env=False) as client:
        before = await client.get(f"{arguments.metrics_endpoint.rstrip('/')}/metrics")
        before.raise_for_status()
        requests = [
            await request(
                client,
                arguments,
                task_id="target",
                content=target,
                lifecycle="active",
                expected_reuse_probability=1.0,
                allow_kv_offload=True,
            )
        ]
        for index in range(arguments.pressure_requests):
            pressure = " ".join(
                [f"unrelated pressure prefix {arguments.run_id} {index}"]
                * arguments.pressure_repetitions
            )
            requests.append(
                await request(
                    client,
                    arguments,
                    task_id=f"pressure-{index}",
                    content=pressure,
                    lifecycle="active",
                    expected_reuse_probability=0.0,
                    allow_kv_offload=False,
                )
            )
        requests.append(
            await request(
                client,
                arguments,
                task_id="target",
                content=target,
                lifecycle="resume",
                expected_reuse_probability=1.0,
                allow_kv_offload=True,
                expected_resume_ms=60_000,
            )
        )
        after = await client.get(f"{arguments.metrics_endpoint.rstrip('/')}/metrics")
        after.raise_for_status()
    store_delta = metric(after.text, STORE_BYTES) - metric(before.text, STORE_BYTES)
    load_delta = metric(after.text, LOAD_BYTES) - metric(before.text, LOAD_BYTES)
    return {
        "schema": 1,
        "evidence_level": "MECHANISM_ONLY",
        "performance_claim_admissible": False,
        "observed_at": datetime.now(UTC).isoformat(),
        "run_id": arguments.run_id,
        "control": control,
        "requests": requests,
        "observed_counter_deltas": {
            "gpu_to_cpu_bytes": store_delta,
            "cpu_to_gpu_bytes": load_delta,
        },
        "assertions": {
            "directive_applicable": control["preview_decision"]["applicable"] is True,
            "directive_enabled": control["preview_decision"]["enabled"] is True,
            "target_offload_enabled": requests[0]["offload"] == "enabled",
            "pressure_offload_disabled": all(
                item["offload"] == "disabled" for item in requests[1:-1]
            ),
            "resume_offload_enabled": requests[-1]["offload"] == "enabled",
            "gpu_to_cpu_observed": store_delta > 0,
            "cpu_to_gpu_observed": load_delta > 0,
        },
        "limitations": [
            "This probe validates scheduler authority through Gateway to native vLLM "
            "transfer counters.",
            "It is not a paired Harness benchmark and cannot support an end-to-end "
            "performance claim.",
            "Predictive waste thresholds remain unfrozen until the real Harness baseline "
            "is collected.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive KV offload control-chain probe")
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--worker-endpoint", required=True)
    parser.add_argument("--metrics-endpoint", required=True)
    parser.add_argument("--worker-id", default="workstation-a5000-probe")
    parser.add_argument("--worker-generation", type=int, default=1)
    parser.add_argument("--node-id", default="workstation")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--gpu-name", default="NVIDIA RTX A5000")
    parser.add_argument("--compute-capability", default="8.6")
    parser.add_argument("--total-vram-bytes", type=int, default=24 * 1024**3)
    parser.add_argument("--free-vram-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--kv-cache-capacity-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--kv-cache-free-bytes", type=int, default=2 * 1024**2)
    parser.add_argument("--cache-load-bytes-per-second", type=float, default=5e9)
    parser.add_argument("--cache-store-bytes-per-second", type=float, default=5e9)
    parser.add_argument("--prefill-tokens-per-second", type=float, default=3_000)
    parser.add_argument("--decode-tokens-per-second", type=float, default=100)
    parser.add_argument("--kv-bytes-per-token", type=int, default=12_288)
    parser.add_argument("--max-model-len", type=int, default=4_096)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--api-key", default="predictive-probe-key")
    parser.add_argument("--target-repetitions", type=int, default=120)
    parser.add_argument("--pressure-repetitions", type=int, default=140)
    parser.add_argument("--pressure-requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = asyncio.run(run(arguments))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not all(payload["assertions"].values()):
        raise RuntimeError(f"probe assertions failed: {payload['assertions']}")
    print(json.dumps(payload["assertions"], indent=2))


if __name__ == "__main__":
    main()
