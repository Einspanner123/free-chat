"""Observe native Store -> pressure -> Load on a dedicated, already-running worker."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from freechat_contracts import WorkerCapabilities
from freechat_worker.transfer_calibration import observe_transfer, summarize_transfers

from benchmarks.records import emit_record


async def run(args: argparse.Namespace) -> None:
    caps = WorkerCapabilities.model_validate_json(args.capabilities.read_text())
    model = caps.models[0].model_id
    artifacts = []
    summaries = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:

        async def metrics() -> str:
            result = await client.get(f"{caps.endpoint}/metrics")
            result.raise_for_status()
            return result.text

        async def infer(repetitions: int, salt: str, offload: bool) -> dict[str, Any]:
            result = await client.post(
                f"{caps.endpoint}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Read the supplied tool result carefully. " * repetitions,
                        }
                    ],
                    "cache_salt": salt,
                    "max_tokens": 16,
                    "ignore_eos": True,
                    "temperature": 0,
                    "kv_transfer_params": {"max_offload_tokens": 4096 if offload else 0},
                },
            )
            result.raise_for_status()
            await asyncio.sleep(1)
            return dict(result.json())

        for repetitions in args.repetitions:
            stores = []
            loads = []
            for trial in range(-1, args.trials):
                salt = str(uuid4())
                before = await metrics()
                target_response = await infer(repetitions, salt, True)
                stored = await metrics()
                # 8 * ~2k tokens exceeds this probe worker's 64 MiB HBM KV pool.
                # Pressure requests have different salts and cannot spill to CPU.
                for _ in range(8):
                    await infer(280, str(uuid4()), False)
                pressure = await metrics()
                resume_response = await infer(repetitions, salt, False)
                resumed = await metrics()
                stem = f"r{repetitions}-trial{trial}"
                artifacts.append(
                    emit_record(
                        f"{stem}-responses.json",
                        json.dumps(
                            {
                                "cache_salt": salt,
                                "target": target_response,
                                "resume": resume_response,
                            },
                            indent=2,
                        ),
                    )
                )
                if (
                    target_response["usage"]["prompt_tokens"]
                    != resume_response["usage"]["prompt_tokens"]
                ):
                    raise ValueError("target and resume token counts disagree")
                for phase, payload in (
                    ("before", before),
                    ("stored", stored),
                    ("pressure", pressure),
                    ("resumed", resumed),
                ):
                    artifacts.append(emit_record(f"{stem}-{phase}.prom", payload))
                if trial < 0:
                    continue  # Initializes lazy counters; retained, never fitted.
                stores.append(observe_transfer(before, stored, model, "store"))
                loads.append(observe_transfer(pressure, resumed, model, "load"))
                if loads[-1].transferred_bytes != (
                    loads[-1].external_hit_tokens * caps.models[0].kv_bytes_per_token
                ):
                    raise ValueError("load bytes do not match this model's KV token layout")
            record = {
                "repetitions": repetitions,
                "store": [item.model_dump() for item in stores],
                "load": [item.model_dump() for item in loads],
                "summaries": [summarize_transfers(stores), summarize_transfers(loads)],
            }
            summaries.append(record)
        artifacts.append(emit_record("observations.json", json.dumps(summaries, indent=2)))
    manifest = {
        "observed_at": datetime.now(UTC).isoformat(),
        "engine_instance_id": args.engine_instance_id,
        "image_identity": args.image_identity,
        "capabilities": caps.model_dump(mode="json"),
        "artifacts": artifacts,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence_level": "MECHANISM_ONLY",
        "performance_claim_admissible": False,
        "limitations": [
            "Single isolated worker, synthetic pressure, not a Harness trial.",
            "Transfer service durations are not request latency or PCIe saturation.",
            "External hit correlation is not a live per-block residency catalog.",
            "Not automatically installed into scheduler profiles.",
        ],
    }
    emit_record("manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--engine-instance-id", required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--repetitions", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 3 or min(args.repetitions) < 1:
        parser.error("requires three trials and positive repetitions")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
