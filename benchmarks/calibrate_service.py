"""Calibrate isolated engine service intervals and validate gRPC routing scope."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from freechat_contracts import AgentHints, CostCalibration, RequestProfile, WorkerCapabilities
from freechat_worker.calibration import ServiceObservation, fit_service_profile, observe_request
from freechat_worker.telemetry import TelemetryCollector

from benchmarks.probe_runtime import ProbeSession, RuntimeObservation, observe_free_vram, route_once
from benchmarks.records import emit_record


async def validate_routes(
    caps: WorkerCapabilities,
    profiles: tuple[CostCalibration, ...],
    observation: RuntimeObservation,
    metrics: str,
    free_vram_bytes: int,
) -> list[dict[str, Any]]:
    if not profiles:
        raise ValueError("service_profiles_required")
    telemetry = observation.budgeted(
        TelemetryCollector(caps, observation.engine_instance_id, profiles).collect(
            metrics,
            free_vram_bytes=free_vram_bytes,
            observed_at=datetime.now(UTC),
        )
    )
    results = []
    for profile in profiles:
        request = RequestProfile(
            tenant_id="calibration-probe",
            local_node_id=caps.node_id,
            model_id=profile.model.model_id,
            input_tokens=profile.input_tokens_min,
            output_tokens=profile.output_tokens_min,
            hints=AgentHints(harness_id="calibration", task_id="task", agent_id="agent"),
        )
        decision = await route_once(caps, telemetry, request)
        assert decision.selected.estimate_available
        assert decision.selected.calibration_id == profile.calibration_id
        assert decision.strategy == "lifecycle-aware"
        assert not decision.kv_transfer.enabled
        results.append({"case": "in-scope", "decision": decision.model_dump(mode="json")})
    request = request.model_copy(
        update={
            "request_id": str(uuid4()),
            "input_tokens": max(item.input_tokens_max for item in profiles) + 100,
        }
    )
    decision = await route_once(caps, telemetry, request)
    assert not decision.selected.estimate_available
    assert decision.strategy == "least-load"
    assert decision.fallback_reason == "candidate_cost_unavailable"
    results.append({"case": "out-of-scope", "decision": decision.model_dump(mode="json")})
    return results


async def run(args: argparse.Namespace) -> None:
    caps = WorkerCapabilities.model_validate_json(args.capabilities.read_text())
    artifacts = []
    profiles = []
    token = os.environ.get("FREECHAT_WORKER_TOKEN", "")
    if len(token) < 32:
        raise ValueError("FREECHAT_WORKER_TOKEN requires at least 32 characters")
    session = ProbeSession(caps, args.engine_instance_id, token)
    async with httpx.AsyncClient(
        timeout=120,
        trust_env=False,
        headers={"x-freechat-worker-token": token},
    ) as client:

        async def metrics() -> str:
            response = await client.get(f"{caps.endpoint}/metrics")
            response.raise_for_status()
            return response.text

        async def infer(repetitions: int) -> dict[str, Any]:
            return await session.infer(
                client,
                {
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

        health = await client.get(f"{caps.endpoint}/health")
        health.raise_for_status()
        initial = await session.observe(client)
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
                artifacts.append(emit_record(f"{stem}-before.prom", before))
                artifacts.append(emit_record(f"{stem}-after.prom", after))
                observation = observe_request(before, after, caps.models[0].model_id)
                if observation.input_tokens != response["usage"]["prompt_tokens"]:
                    raise ValueError("response and engine prompt token counts disagree")
                if observation.output_tokens != response["usage"]["completion_tokens"]:
                    raise ValueError("response and engine output token counts disagree")
                observations.append(observation)
                records.append({"sample": observation.model_dump(), "response": response})
            artifact = emit_record(
                f"r{repetitions}-observations.json",
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
                emit_record(
                    f"r{repetitions}-profile.json",
                    profile.model_dump_json(indent=2) + "\n",
                )
            )
        routes = await validate_routes(
            caps,
            tuple(profiles),
            await session.observe(client),
            await metrics(),
            await observe_free_vram(caps, args.gpu_host),
        )
    artifacts.append(emit_record("routes.json", json.dumps(routes, indent=2) + "\n"))
    manifest = {
        "evidence_level": "MECHANISM_ONLY",
        "performance_claim_admissible": False,
        "observed_at": datetime.now(UTC).isoformat(),
        "capabilities": caps.model_dump(mode="json"),
        "runtime_observation": initial.model_dump(mode="json"),
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
    emit_record("manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps([profile.model_dump(mode="json") for profile in profiles], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--engine-instance-id", required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--gpu-host", help="Optional SSH host; default queries local nvidia-smi")
    parser.add_argument("--repetitions", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.trials < 3 or args.output_tokens < 2 or min(args.repetitions) < 1:
        parser.error("requires at least three trials, two output tokens and positive repetitions")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
