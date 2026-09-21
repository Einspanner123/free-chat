"""Real worker scrape through an isolated gRPC scheduler; no performance claim."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from freechat_contracts import AgentHints, RequestProfile, WorkerCapabilities
from freechat_worker.telemetry import TelemetryCollector

from benchmarks.probe_runtime import ProbeSession, observe_free_vram, route_once
from benchmarks.records import emit_record


async def run(args: argparse.Namespace) -> None:
    caps = WorkerCapabilities.model_validate_json(args.capabilities.read_text())
    collector = TelemetryCollector(caps, args.engine_instance_id)
    records = []
    artifacts = []
    token = os.environ.get("FREECHAT_WORKER_TOKEN", "")
    if len(token) < 32:
        raise ValueError("FREECHAT_WORKER_TOKEN requires at least 32 characters")
    session = ProbeSession(caps, args.engine_instance_id, token)
    async with httpx.AsyncClient(
        timeout=120,
        trust_env=False,
        headers={"x-freechat-worker-token": token},
    ) as client:
        health = await client.get(f"{caps.endpoint}/health")
        health.raise_for_status()
        await session.observe(client)
        for phase in ("before", "after"):
            if phase == "after":
                await session.infer(
                    client,
                    {
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
                        "kv_transfer_params": {"max_offload_tokens": 0},
                    },
                )
                # Allow asynchronous engine counters to publish before the second scrape.
                await asyncio.sleep(2)
            metrics = await client.get(f"{caps.endpoint}/metrics")
            metrics.raise_for_status()
            artifacts.append(emit_record(f"{phase}.prom", metrics.text))
            telemetry = collector.collect(
                metrics.text,
                free_vram_bytes=await observe_free_vram(caps, args.gpu_host),
                observed_at=datetime.now(UTC),
            )
            observation = await session.observe(client)
            telemetry = observation.budgeted(telemetry)
            records.append(
                {
                    "phase": phase,
                    "status": "runtime_observed",
                    "runtime_observation": observation.model_dump(mode="json"),
                    "telemetry": telemetry.model_dump(mode="json"),
                }
            )
        profile = RequestProfile(
            tenant_id="probe",
            local_node_id=caps.node_id,
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
        decision = await route_once(caps, telemetry, profile)
        result = {
            "evidence_level": "MECHANISM_ONLY",
            "performance_claim_admissible": False,
            "records": records,
            "decision": decision.model_dump(mode="json"),
            "artifacts": artifacts,
            "limitations": [
                "Isolated in-memory scheduler, real GPU worker.",
                "Serial isolated observation; offload is never enabled by the probe.",
                "No prefill calibration or Harness performance acceptance.",
            ],
        }
        assert not decision.kv_transfer.enabled
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--gpu-host", help="Optional SSH host; default queries local nvidia-smi")
    parser.add_argument("--engine-instance-id", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
