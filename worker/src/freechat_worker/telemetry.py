"""Single-engine vLLM metrics to fenced Worker heartbeats.

Usage ratio is reported separately: it is not a prefix residency inventory.
Counter deltas measure transfer service time, not wall-clock PCIe saturation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import grpc
import httpx
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import CostCalibration, WorkerCapabilities, WorkerTelemetry
from prometheus_client.parser import text_string_to_metric_families

METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:kv_offload_store_size_sum",
    "vllm:kv_offload_store_time_total",
    "vllm:kv_offload_load_size_sum",
    "vllm:kv_offload_load_time_total",
}


def samples(
    payload: str, model: str, engine: str, *, names: set[str] | None = None,
) -> dict[str, float]:
    selected_names = METRICS if names is None else names
    result: dict[str, float] = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            if sample.name not in selected_names:
                continue
            if sample.labels.get("model_name") != model or sample.labels.get("engine") != engine:
                continue
            if "le" in sample.labels or "reason" in sample.labels:
                continue
            if sample.name in result:
                raise ValueError(f"ambiguous metric series: {sample.name}")
            if not math.isfinite(sample.value) or sample.value < 0:
                raise ValueError(f"invalid metric value: {sample.name}")
            result[sample.name] = sample.value
    return result


class TelemetryCollector:
    def __init__(
        self, capabilities: WorkerCapabilities, engine_instance_id: str,
        calibrations: tuple[CostCalibration, ...] = (),
    ) -> None:
        if len(capabilities.models) != 1:
            raise ValueError("collector requires one served model per worker")
        self.capabilities = capabilities
        self.engine_instance_id = engine_instance_id
        self.previous: dict[str, float] | None = None
        self.calibrations = calibrations

    def collect(
        self, payload: str, *, free_vram_bytes: int, observed_at: datetime
    ) -> WorkerTelemetry:
        current = samples(payload, self.capabilities.models[0].model_id, "0")
        required = (
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc",
        )
        if any(name not in current for name in required):
            raise ValueError("missing required single-engine metrics")
        for name in required[:2]:
            if not current[name].is_integer():
                raise ValueError("non-integral request count")
        rates: dict[str, float] = {}
        reset = self.previous is not None and any(
            value < self.previous[name]
            for name, value in current.items()
            if name in self.previous and (name.endswith("_sum") or name.endswith("_total"))
        )
        if reset:
            self.previous = None
            raise ValueError("engine counters reset; re-register with a new worker generation")
        if self.previous is not None:
            for direction in ("store", "load"):
                size = f"vllm:kv_offload_{direction}_size_sum"
                duration = f"vllm:kv_offload_{direction}_time_total"
                if all(key in current and key in self.previous for key in (size, duration)):
                    byte_delta = current[size] - self.previous[size]
                    time_delta = current[duration] - self.previous[duration]
                    if byte_delta > 0 and time_delta > 0:
                        rates[direction] = byte_delta / time_delta
        self.previous = current
        return WorkerTelemetry(
            worker_id=self.capabilities.worker_id,
            generation=self.capabilities.generation,
            observed_at=observed_at,
            free_vram_bytes=free_vram_bytes,
            active_requests=int(current[required[0]]),
            queue_depth=int(current[required[1]]),
            kv_cache_usage_ratio=current[required[2]],
            cache_store_bytes_per_second=rates.get("store"),
            cache_load_bytes_per_second=rates.get("load", 1.0),
            telemetry_source="vllm-prometheus-window",
            engine_instance_id=self.engine_instance_id,
            transfer_observed_at=observed_at if rates else None,
            calibrations=self.calibrations,
        )


async def heartbeat(stub: Any, telemetry: WorkerTelemetry) -> str:
    async def stream() -> AsyncIterator[control_pb2.WorkerHeartbeat]:
        yield control_pb2.WorkerHeartbeat(
            context=control_pb2.RequestContext(
                request_id=telemetry.observed_at.isoformat(),
                idempotency_key=telemetry.observed_at.isoformat(),
                tenant_id="system",
            ),
            worker_id=telemetry.worker_id,
            generation=telemetry.generation,
            telemetry_json=telemetry.model_dump_json(),
        )

    response = await stub.Heartbeat(stream()).read()
    return str(response.status)


async def run(args: argparse.Namespace) -> None:
    caps = WorkerCapabilities.model_validate_json(args.capabilities.read_text())
    calibrations = tuple(
        CostCalibration.model_validate_json(path.read_text()) for path in args.calibration
    )
    collector = TelemetryCollector(caps, args.engine_instance_id, calibrations)
    channel = grpc.aio.insecure_channel(args.scheduler)
    worker = control_pb2_grpc.WorkerControlServiceStub(channel)  # type: ignore[no-untyped-call]
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            for _ in range(args.samples):
                response = await client.get(f"{caps.endpoint.rstrip('/')}/health")
                response.raise_for_status()
                response = await client.get(f"{caps.endpoint.rstrip('/')}/metrics")
                response.raise_for_status()
                process = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    f"--id={caps.gpu_id}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()
                if process.returncode:
                    raise RuntimeError("GPU memory observation failed")
                telemetry = collector.collect(
                    response.text,
                    free_vram_bytes=int(stdout.decode().strip()) * 1024**2,
                    observed_at=datetime.now(UTC),
                )
                status = await heartbeat(worker, telemetry)
                print(
                    json.dumps({"status": status, "telemetry": telemetry.model_dump(mode="json")}),
                    flush=True,
                )
                await asyncio.sleep(args.interval)
    finally:
        await channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--engine-instance-id", required=True)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.samples < 1 or not 0 < args.interval <= 15:
        parser.error("samples must be positive; interval must be in (0, 15]")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
