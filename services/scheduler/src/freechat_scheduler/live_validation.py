from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import grpc
import nats
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import ModelCapability, RouteDecision, WorkerCapabilities
from freechat_control_store import EtcdHttpStore

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


async def register_and_route(target: str, state_path: Path, nats_url: str) -> dict[str, Any]:
    channel = grpc.aio.insecure_channel(target)
    try:
        worker = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
        scheduler = control_pb2_grpc.SchedulerServiceStub(channel)  # type: ignore[no-untyped-call]
        capabilities = WorkerCapabilities(
            worker_id="workstation-a5000",
            generation=17,
            endpoint="http://workstation-worker:8000",
            node_id="workstation",
            gpu_id="0",
            gpu_name="NVIDIA RTX A5000",
            compute_capability="8.6",
            total_vram_bytes=24 * 1024**3,
            p2p_domain="workstation",
            network_domain="weak-lan",
            models=(
                ModelCapability(
                    model_id=MODEL,
                    revision="validation-model-sha",
                    tokenizer_revision="validation-tokenizer-sha",
                    architecture="dense",
                    attention="gqa",
                    max_context_tokens=32_768,
                    dtype="bfloat16",
                ),
            ),
        )
        await worker.Register(
            control_pb2.WorkerRegistration(
                context=_context("register-request", "register-validation"),
                worker_id=capabilities.worker_id,
                generation=capabilities.generation,
                endpoint=capabilities.endpoint,
                capabilities_json=capabilities.model_dump_json(),
            )
        )
        route = await scheduler.Route(
            control_pb2.RouteRequest(
                context=_context("route-request", "route-validation", tenant="tenant-a"),
                hints=control_pb2.AgentHints(
                    harness_id="langgraph",
                    task_id="validation-task",
                    agent_id="validation-agent",
                    lifecycle="active",
                    prefix_scope="agent",
                    reuse_class="growing_history",
                    confidence=1,
                    source="explicit",
                    allow_preemption=True,
                    allow_kv_offload=True,
                    allow_remote_worker=True,
                ),
                model=MODEL,
                input_tokens=128,
                output_tokens=32,
            )
        )
        state = {
            "decision_id": route.decision_id,
            "worker_id": route.worker_id,
            "worker_generation": route.worker_generation,
        }
        state_path.write_text(json.dumps(state))
        state["jetstream_messages"] = await _stream_messages(nats_url)
        return state
    finally:
        await channel.close()


async def recover_and_release(
    target: str,
    state_path: Path,
    nats_url: str,
    etcd_url: str,
) -> dict[str, Any]:
    state = await _load_state(state_path, etcd_url)
    channel = grpc.aio.insecure_channel(target)
    try:
        scheduler = control_pb2_grpc.SchedulerServiceStub(channel)  # type: ignore[no-untyped-call]
        lease = control_pb2.LeaseRequest(
            context=_context("recover-request", "release-validation", tenant="tenant-a"),
            decision_id=state["decision_id"],
            worker_id=state["worker_id"],
            worker_generation=state["worker_generation"],
        )
        decision = await scheduler.ExplainDecision(lease)
        release = await scheduler.Release(lease)
        return {
            "restored_worker": decision.worker_id,
            "restored_generation": decision.worker_generation,
            "release_status": release.status,
            "jetstream_messages": await _stream_messages(nats_url),
        }
    finally:
        await channel.close()


def _context(request_id: str, idempotency_key: str, *, tenant: str = "system") -> Any:
    return control_pb2.RequestContext(
        request_id=request_id,
        idempotency_key=idempotency_key,
        tenant_id=tenant,
    )


async def _stream_messages(nats_url: str) -> int:
    client = await nats.connect(nats_url, connect_timeout=5)
    try:
        info = await client.jetstream().stream_info("FREECHAT_LIFECYCLE")
        return info.state.messages
    finally:
        await client.drain()


async def _load_state(state_path: Path, etcd_url: str) -> dict[str, Any]:
    if state_path.exists():
        return dict(json.loads(state_path.read_text()))
    store = EtcdHttpStore(etcd_url)
    try:
        leases = await store.list_prefix("/freechat/leases/")
        if not leases:
            raise RuntimeError("no persisted lease is available for recovery validation")
        decision = RouteDecision.model_validate_json(leases[0].value)
        return {
            "decision_id": decision.decision_id,
            "worker_id": decision.worker_id,
            "worker_generation": decision.worker_generation,
        }
    finally:
        await store.close()


async def _run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register", "recover"))
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--state", type=Path, default=Path("/tmp/freechat-live-validation.json"))
    parser.add_argument("--nats-url", default="nats://nats:4222")
    parser.add_argument("--etcd-url", default="http://etcd:2379")
    args = parser.parse_args()
    if args.phase == "register":
        result = await register_and_route(args.target, args.state, args.nats_url)
    else:
        result = await recover_and_release(args.target, args.state, args.nats_url, args.etcd_url)
    print(json.dumps(result, sort_keys=True))


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
