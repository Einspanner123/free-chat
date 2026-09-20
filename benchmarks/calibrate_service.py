"""Calibrate isolated engine service intervals and validate gRPC routing scope."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import grpc
import httpx
from freechat.control.v1 import control_pb2_grpc
from freechat_contracts import AgentHints, CostCalibration, RequestProfile, WorkerCapabilities
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService, WorkerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.scheduler import Scheduler
from freechat_worker.calibration import ServiceObservation, fit_service_profile, observe_request
from freechat_worker.telemetry import TelemetryCollector, heartbeat


def save(path: Path, payload: str) -> dict[str, str]:
    path.write_text(payload)
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


async def validate_routes(
    caps: WorkerCapabilities,
    profiles: tuple[CostCalibration, ...],
    engine_id: str,
    metrics: str,
    free_vram_bytes: int,
) -> list[dict[str, Any]]:
    telemetry = TelemetryCollector(caps, engine_id, profiles).collect(
        metrics,
        free_vram_bytes=free_vram_bytes,
        observed_at=datetime.now(UTC),
    )
    registry = InMemoryWorkerRegistry()
    await registry.register(caps, telemetry)
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(Scheduler(registry), LeaseBook()),
        server,
    )
    control_pb2_grpc.add_WorkerControlServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerGrpcService(registry),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    client = GrpcSchedulerClient(f"127.0.0.1:{port}")
    worker = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
    results = []
    try:
        assert await heartbeat(worker, telemetry) == "heartbeat_accepted"
        for profile in profiles:
            request = RequestProfile(
                tenant_id="calibration-probe",
                model_id=profile.model.model_id,
                input_tokens=profile.input_tokens_min,
                output_tokens=profile.output_tokens_min,
                hints=AgentHints(harness_id="calibration", task_id="task", agent_id="agent"),
            )
            decision = await client.route(request)
            await client.release(request, decision)
            assert decision.selected.estimate_available
            assert decision.selected.calibration_id == profile.calibration_id
            assert decision.strategy == "lifecycle-aware"
            assert decision.kv_transfer.reason == "transfer_calibration_required"
            results.append({"case": "in-scope", "decision": decision.model_dump(mode="json")})
        request = request.model_copy(
            update={"input_tokens": max(item.input_tokens_max for item in profiles) + 100}
        )
        decision = await client.route(request)
        await client.release(request, decision)
        assert not decision.selected.estimate_available
        assert decision.strategy == "least-load"
        assert decision.fallback_reason == "candidate_cost_unavailable"
        results.append({"case": "out-of-scope", "decision": decision.model_dump(mode="json")})
        return results
    finally:
        await client.aclose()
        await channel.close()
        await server.stop(None)


async def run(args: argparse.Namespace) -> None:
    caps = WorkerCapabilities.model_validate_json(args.capabilities.read_text())
    if args.output.exists():
        raise ValueError("output exists; choose a unique run directory")
    args.output.mkdir(parents=True)
    artifacts = []
    profiles = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:

        async def metrics() -> str:
            response = await client.get(f"{caps.endpoint}/metrics")
            response.raise_for_status()
            return response.text

        async def infer(repetitions: int) -> dict[str, Any]:
            response = await client.post(
                f"{caps.endpoint}/v1/chat/completions",
                json={
                    "model": caps.models[0].model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Read the supplied tool result carefully. " * repetitions,
                        }
                    ],
                    "cache_salt": str(uuid4()),
                    "temperature": 0,
                    "max_tokens": args.output_tokens,
                    "ignore_eos": True,
                    "kv_transfer_params": {"max_offload_tokens": 0},
                },
            )
            response.raise_for_status()
            return dict(response.json())

        health = await client.get(f"{caps.endpoint}/health")
        health.raise_for_status()
        for repetitions in args.repetitions:
            # Warm kernels and initialize lazily emitted histograms; never fit this sample.
            await infer(repetitions)
            await asyncio.sleep(1)
            observations: list[ServiceObservation] = []
            records = []
            for index in range(args.trials):
                before = await metrics()
                response = await infer(repetitions)
                await asyncio.sleep(1)
                after = await metrics()
                stem = f"r{repetitions}-trial{index}"
                artifacts.append(save(args.output / f"{stem}-before.prom", before))
                artifacts.append(save(args.output / f"{stem}-after.prom", after))
                observation = observe_request(before, after, caps.models[0].model_id)
                if observation.input_tokens != response["usage"]["prompt_tokens"]:
                    raise ValueError("response and engine prompt token counts disagree")
                if observation.output_tokens != response["usage"]["completion_tokens"]:
                    raise ValueError("response and engine output token counts disagree")
                observations.append(observation)
                records.append({"sample": observation.model_dump(), "response": response})
            artifact = save(
                args.output / f"r{repetitions}-observations.json",
                json.dumps(records, indent=2) + "\n",
            )
            artifacts.append(artifact)
            profile = fit_service_profile(
                observations,
                capabilities=caps,
                engine_instance_id=args.engine_instance_id,
                image_identity=args.image_identity,
                artifact_sha256=artifact["sha256"],
            )
            profiles.append(profile)
            artifacts.append(
                save(
                    args.output / f"r{repetitions}-profile.json",
                    profile.model_dump_json(indent=2) + "\n",
                )
            )
        process = await asyncio.create_subprocess_exec(
            "ssh",
            args.gpu_host,
            "nvidia-smi",
            f"--id={caps.gpu_id}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode:
            raise RuntimeError("GPU observation failed")
        routes = await validate_routes(
            caps,
            tuple(profiles),
            args.engine_instance_id,
            await metrics(),
            int(stdout.decode().strip()) * 1024**2,
        )
    artifacts.append(save(args.output / "routes.json", json.dumps(routes, indent=2) + "\n"))
    manifest = {
        "evidence_level": "MECHANISM_ONLY",
        "performance_claim_admissible": False,
        "observed_at": datetime.now(UTC).isoformat(),
        "capabilities": caps.model_dump(mode="json"),
        "engine_instance_id": args.engine_instance_id,
        "image_identity": args.image_identity,
        "artifacts": artifacts,
        "limitations": [
            "Single worker, isolated requests, uncached prompts, warm engine.",
            "No concurrent or Harness performance acceptance.",
            "Transfer calibration absent; predictive offload remains disabled.",
            "Image identity is a local image ID, not a registry digest.",
        ],
    }
    save(args.output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps([profile.model_dump(mode="json") for profile in profiles], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--engine-instance-id", required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--gpu-host", required=True)
    parser.add_argument("--repetitions", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 3 or args.output_tokens < 2 or min(args.repetitions) < 1:
        parser.error("requires at least three trials, two output tokens and positive repetitions")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
