"""Real worker scrape through an isolated gRPC scheduler; no performance claim."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import grpc
import httpx
from freechat.control.v1 import control_pb2_grpc
from freechat_contracts import AgentHints, RequestProfile, WorkerCapabilities, WorkerTelemetry
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.scheduler import Scheduler
from freechat_worker.telemetry import TelemetryCollector, heartbeat


async def run(args: argparse.Namespace) -> None:
    caps = WorkerCapabilities.model_validate_json(args.capabilities.read_text())
    if args.output.exists():
        raise ValueError("output directory already exists; use a unique run path")
    registry = InMemoryWorkerRegistry()
    await registry.register(
        caps,
        WorkerTelemetry(
            worker_id=caps.worker_id,
            generation=caps.generation,
            free_vram_bytes=0,
        ),
    )
    server = grpc.aio.server()
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry),
        server,
    )
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(Scheduler(registry), LeaseBook()),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    worker = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
    router = GrpcSchedulerClient(f"127.0.0.1:{port}")
    collector = TelemetryCollector(caps, args.engine_instance_id)
    records = []
    artifacts = []
    args.output.mkdir(parents=True)
    try:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            health = await client.get(f"{caps.endpoint}/health")
            health.raise_for_status()
            for phase in ("before", "after"):
                if phase == "after":
                    reply = await client.post(
                        f"{caps.endpoint}/v1/chat/completions",
                        json={
                            "model": caps.models[0].model_id,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"{uuid4()} "
                                    + "Read the supplied tool result carefully. " * 100,
                                }
                            ],
                            "max_tokens": 1,
                            "temperature": 0,
                            "kv_transfer_params": {"max_offload_tokens": 4096},
                        },
                    )
                    reply.raise_for_status()
                    # Allow asynchronous engine counters to publish before the second scrape.
                    await asyncio.sleep(2)
                metrics = await client.get(f"{caps.endpoint}/metrics")
                metrics.raise_for_status()
                path = args.output / f"{phase}.prom"
                path.write_text(metrics.text)
                artifacts.append(
                    {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
                gpu = await asyncio.create_subprocess_exec(
                    "ssh",
                    args.gpu_host,
                    "nvidia-smi",
                    f"--id={caps.gpu_id}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await gpu.communicate()
                if gpu.returncode:
                    raise RuntimeError("remote GPU observation failed")
                telemetry = collector.collect(
                    metrics.text,
                    free_vram_bytes=int(stdout.decode().strip()) * 1024**2,
                    observed_at=datetime.now(UTC),
                )
                status = await heartbeat(worker, telemetry)
                records.append(
                    {
                        "phase": phase,
                        "status": status,
                        "telemetry": telemetry.model_dump(mode="json"),
                    }
                )
            profile = RequestProfile(
                tenant_id="probe",
                model_id=caps.models[0].model_id,
                input_tokens=100,
                output_tokens=1,
                hints=AgentHints(
                    harness_id="probe",
                    task_id="task",
                    agent_id="agent",
                    expected_reuse_probability=1,
                ),
            )
            decision = await router.route(profile)
            await router.release(profile, decision)
            result = {
                "evidence_level": "MECHANISM_ONLY",
                "performance_claim_admissible": False,
                "records": records,
                "decision": decision.model_dump(mode="json"),
                "artifacts": artifacts,
                "limitations": [
                    "Isolated in-memory scheduler, real GPU worker.",
                    "Direct request explicitly enables offload for measurement.",
                    "No prefill calibration or Harness performance acceptance.",
                ],
            }
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            assert records[-1]["status"] == "heartbeat_accepted"
            assert telemetry.cache_store_bytes_per_second is not None
            assert decision.kv_transfer.reason == "prefill_calibration_required"
            print(json.dumps(result, indent=2))
    finally:
        await router.aclose()
        await channel.close()
        await server.stop(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--gpu-host", required=True)
    parser.add_argument("--engine-instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
