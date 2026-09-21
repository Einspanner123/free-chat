"""CPU probe orchestration only: HTTP/SSH observations are deterministic fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import grpc
import httpx
import pytest
import respx
from freechat_contracts import AgentHints, ModelCapability, RequestProfile, WorkerCapabilities
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.scheduler import Scheduler
from freechat_worker.calibration import HISTOGRAMS, ServiceObservation, fit_service_profile
from freechat_worker.telemetry import TelemetryCollector

from benchmarks import calibrate_service, calibrate_transfer, telemetry_probe


@pytest.fixture
def capabilities(tmp_path: Path) -> Path:
    caps = WorkerCapabilities(
        worker_id="fixture-worker",
        generation=1,
        endpoint="http://probe.invalid:8000",
        node_id="fixture-node",
        gpu_id="0",
        gpu_name="fixture-not-gpu-evidence",
        compute_capability="8.6",
        total_vram_bytes=8 * 1024**3,
        p2p_domain="fixture",
        network_domain="fixture",
        models=(
            ModelCapability(
                model_id="model",
                revision="fixture",
                tokenizer_revision="fixture",
                architecture="dense",
                attention="gqa",
                max_context_tokens=4096,
                dtype="half",
                supports_kv_offload=True,
                kv_bytes_per_token=256,
                kv_admission_bytes_per_token_per_rank=256,
                kv_block_size_tokens=16,
            ),
        ),
    )
    path = tmp_path / "capabilities.json"
    path.write_text(caps.model_dump_json())
    return path


def arguments(capabilities: Path) -> argparse.Namespace:
    return argparse.Namespace(
        capabilities=capabilities,
        engine_instance_id="fixture-engine",
        image_identity="fixture-image",
        gpu_host="fixture-host",
        repetitions=[1],
        trials=3,
        output_tokens=16,
    )


class ObservationFixture:
    """Counters are test inputs, not observations of a running GPU worker."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stored_salts: set[str] = set()
        self.stores = 0
        self.loads = 0
        self.mismatch: str | None = None
        self.transfer_bytes = 4096
        self.inference_status = 200

    def infer(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        salt = payload.get("cache_salt", "")
        resumed = salt in self.stored_salts
        if payload["kv_transfer_params"]["max_offload_tokens"]:
            self.stores += 1
            self.stored_salts.add(salt)
        elif resumed:
            self.loads += 1
        prompt = (
            101 if self.mismatch == "prompt" or (resumed and self.mismatch == "resume") else 100
        )
        completion = 17 if self.mismatch == "output" else 16
        return httpx.Response(
            self.inference_status,
            json={"usage": {"prompt_tokens": prompt, "completion_tokens": completion}},
        )

    def metrics(self, request: httpx.Request) -> httpx.Response:
        del request
        count = len(self.requests)
        values: dict[str, float] = {
            "num_requests_running": 0,
            "num_requests_waiting": 0,
            "num_preemptions_total": 0,
            "kv_cache_usage_perc": 0.25,
            "external_prefix_cache_hits_total": self.loads * 16,
        }
        for name, total in zip(HISTOGRAMS, [0.01, 0.03, 100, 100, 16], strict=True):
            values[f"{name}_count"] = count
            values[f"{name}_sum"] = total * count
        for direction, operations in (("store", self.stores), ("load", self.loads)):
            values[f"kv_offload_{direction}_size_sum"] = operations * self.transfer_bytes
            values[f"kv_offload_{direction}_size_count"] = operations
            values[f"kv_offload_{direction}_time_total"] = operations * 0.01
        return httpx.Response(
            200,
            text="\n".join(
                f'vllm:{name}{{model_name="model",engine="0"}} {value}'
                for name, value in values.items()
            ),
        )


@pytest.fixture
def observations(monkeypatch: pytest.MonkeyPatch) -> Any:
    engine = ObservationFixture()
    process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"4096\n", b"")))
    subprocess = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", subprocess)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with respx.mock(assert_all_called=False) as router:
        router.get("http://probe.invalid:8000/health").respond(200)
        router.get("http://probe.invalid:8000/metrics").mock(side_effect=engine.metrics)
        router.post("http://probe.invalid:8000/v1/chat/completions").mock(side_effect=engine.infer)
        yield engine, process, subprocess, router


def artifacts(output: str) -> dict[str, Any]:
    records = {}
    for line in output.splitlines():
        if line.startswith('{"record_type":'):
            record = json.loads(line)
            records[record["name"]] = record
    return records


async def test_service_probe_missing_allocator_budget_fails_closed(
    capabilities: Path, observations: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    engine, _, subprocess, _ = observations
    # This probe predates allocator-backed admission. Metrics and free VRAM
    # cannot authorize a route; retain this rejection until real capacity is wired.
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await calibrate_service.run(arguments(capabilities))
    assert error.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "no worker satisfies" in (error.value.details() or "")
    records = artifacts(capsys.readouterr().out)
    assert "manifest.json" not in records and "routes.json" not in records
    profile = json.loads(records["r1-profile.json"]["payload"])
    assert profile["sample_count"] == 3
    assert profile["prefill_tokens_per_second"] == pytest.approx(10_000)
    assert profile["decode_tokens_per_second"] == pytest.approx(500)
    assert profile["artifact_sha256"] == records["r1-observations.json"]["sha256"]
    assert len(engine.requests) == 4  # Warmup is not a fitted observation.
    assert len({item["cache_salt"] for item in engine.requests}) == 4
    assert all(item["kv_transfer_params"]["max_offload_tokens"] == 0 for item in engine.requests)
    assert subprocess.await_args.args[:3] == ("ssh", "fixture-host", "nvidia-smi")


@pytest.mark.parametrize("mismatch", ["prompt", "output"])
async def test_service_probe_rejects_response_metric_disagreement(
    capabilities: Path, observations: Any, mismatch: str
) -> None:
    engine, _, subprocess, _ = observations
    engine.mismatch = mismatch
    with pytest.raises(ValueError, match=f"{mismatch} token counts disagree"):
        await calibrate_service.run(arguments(capabilities))
    subprocess.assert_not_awaited()


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
async def test_failed_gpu_observation_is_not_success(
    capabilities: Path, observations: Any, module: ModuleType
) -> None:
    _, process, subprocess, _ = observations
    process.returncode = 1
    with pytest.raises(RuntimeError, match="GPU observation failed"):
        await module.run(arguments(capabilities))
    assert subprocess.await_count == 1


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
async def test_unhealthy_worker_is_rejected_before_ssh(
    capabilities: Path, observations: Any, module: ModuleType
) -> None:
    _, _, subprocess, router = observations
    router.get("http://probe.invalid:8000/health").respond(503)
    with pytest.raises(httpx.HTTPStatusError):
        await module.run(arguments(capabilities))
    subprocess.assert_not_awaited()


async def test_transfer_probe_preserves_salt_and_excludes_warmup(
    capabilities: Path, observations: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    engine, _, subprocess, _ = observations
    await calibrate_transfer.run(arguments(capabilities))
    records = artifacts(capsys.readouterr().out)
    manifest = json.loads(records["manifest.json"]["payload"])
    measured = json.loads(records["observations.json"]["payload"])
    assert not manifest["performance_claim_admissible"]
    assert len(measured[0]["store"]) == len(measured[0]["load"]) == 3
    assert [item["sample_count"] for item in measured[0]["summaries"]] == [3, 3]
    assert measured[0]["load"][0]["external_hit_tokens"] == 16
    assert measured[0]["load"][0]["transferred_bytes"] == 4096
    assert "r1-trial-1-responses.json" in records
    assert len(engine.requests) == 40
    for offset in range(0, 40, 10):
        target, resume = engine.requests[offset], engine.requests[offset + 9]
        assert target["cache_salt"] == resume["cache_salt"]
        assert target["kv_transfer_params"]["max_offload_tokens"] == 4096
        assert resume["kv_transfer_params"]["max_offload_tokens"] == 0
        pressure = engine.requests[offset + 1 : offset + 9]
        assert len({item["cache_salt"] for item in pressure}) == 8
        assert all(item["cache_salt"] != target["cache_salt"] for item in pressure)
    subprocess.assert_not_awaited()


@pytest.mark.parametrize("kind", ["resume", "layout"])
async def test_transfer_probe_rejects_unmatched_resume_or_kv_layout(
    capabilities: Path, observations: Any, kind: str
) -> None:
    engine, _, _, _ = observations
    if kind == "resume":
        engine.mismatch = "resume"
        reason = "target and resume token counts disagree"
    else:
        engine.transfer_bytes = 8192
        reason = "load bytes do not match"
    with pytest.raises(ValueError, match=reason):
        await calibrate_transfer.run(arguments(capabilities))


@pytest.mark.parametrize("module", [calibrate_service, calibrate_transfer, telemetry_probe])
async def test_inference_http_error_is_propagated(
    capabilities: Path, observations: Any, module: ModuleType
) -> None:
    engine, _, _, _ = observations
    engine.inference_status = 500
    with pytest.raises(httpx.HTTPStatusError):
        await module.run(arguments(capabilities))


async def test_telemetry_probe_missing_allocator_budget_fails_closed(
    capabilities: Path, observations: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    engine, _, subprocess, _ = observations
    # Real loopback gRPC rejects the unsupported probe instead of weakening
    # admission or pretending the fixture's free VRAM is allocator capacity.
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await telemetry_probe.run(arguments(capabilities))
    assert error.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "no worker satisfies" in (error.value.details() or "")
    output = capsys.readouterr().out
    assert set(artifacts(output)) == {"before.prom", "after.prom"}
    assert '"evidence_level"' not in output
    assert subprocess.await_count == 2
    assert len(engine.requests) == 1


@pytest.mark.parametrize("module", [calibrate_service, calibrate_transfer, telemetry_probe])
def test_cli_dispatches_requested_identity(
    capabilities: Path, monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    invoke = AsyncMock()
    monkeypatch.setattr(module, "run", invoke)
    args = ["probe", "--capabilities", str(capabilities), "--engine-instance-id", "chosen-engine"]
    if module is not calibrate_transfer:
        args += ["--gpu-host", "chosen-host"]
    if module is not telemetry_probe:
        args += ["--image-identity", "chosen-image"]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    invoke.assert_awaited_once()
    assert invoke.await_args is not None
    assert invoke.await_args.args[0].engine_instance_id == "chosen-engine"


@pytest.mark.parametrize(
    "module,extra",
    [
        (calibrate_service, ["--trials", "2"]),
        (calibrate_service, ["--output-tokens", "1"]),
        (calibrate_service, ["--repetitions", "0"]),
        (calibrate_transfer, ["--trials", "2"]),
        (calibrate_transfer, ["--repetitions", "0"]),
    ],
)
def test_cli_rejects_invalid_calibration_sample_settings(
    capabilities: Path, monkeypatch: pytest.MonkeyPatch, module: ModuleType, extra: list[str]
) -> None:
    invoke = AsyncMock()
    monkeypatch.setattr(module, "run", invoke)
    args = [
        "probe",
        "--capabilities",
        str(capabilities),
        "--engine-instance-id",
        "fixture-engine",
        "--image-identity",
        "fixture-image",
    ]
    if module is calibrate_service:
        args += ["--gpu-host", "fixture-host"]
    monkeypatch.setattr(sys, "argv", args + extra)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    invoke.assert_not_called()


async def test_probe_scrape_cannot_substitute_free_vram_for_allocator_budget(
    capabilities: Path, observations: Any
) -> None:
    from datetime import UTC, datetime

    from freechat_scheduler.scheduler import NoEligibleWorker

    engine, _, _, _ = observations
    caps = WorkerCapabilities.model_validate_json(capabilities.read_text())
    observed = TelemetryCollector(caps, "fixture-engine").collect(
        engine.metrics(httpx.Request("GET", "http://probe.invalid/metrics")).text,
        free_vram_bytes=4 * 1024**3,
        observed_at=datetime.now(UTC),
    )
    assert observed.kv_admission_available_bytes_per_rank is None
    registry = InMemoryWorkerRegistry()
    await registry.register(caps, observed)
    request = RequestProfile(
        tenant_id="fixture",
        local_node_id=caps.node_id,
        model_id="model",
        input_tokens=100,
        output_tokens=16,
        hints=AgentHints(harness_id="fixture", task_id="fixture", agent_id="fixture"),
    )
    with pytest.raises(NoEligibleWorker) as error:
        Scheduler(registry).route(request)
    assert error.value.rejected == {"fixture-worker": ("kv_admission_budget_unknown",)}


@pytest.mark.parametrize("insufficient_samples", [True, False])
def test_calibration_fit_rejects_insufficient_samples_or_multiple_models(
    capabilities: Path, insufficient_samples: bool
) -> None:
    caps = WorkerCapabilities.model_validate_json(capabilities.read_text())
    if not insufficient_samples:
        caps = caps.model_copy(update={"models": caps.models + caps.models})
    observation = ServiceObservation(
        input_tokens=100, output_tokens=16, prefill_seconds=0.01, decode_seconds=0.03
    )
    with pytest.raises(ValueError, match="at least three samples of one served model"):
        fit_service_profile(
            [observation] * (2 if insufficient_samples else 3),
            capabilities=caps,
            engine_instance_id="fixture-engine",
            image_identity="fixture-image",
            artifact_sha256="a" * 64,
        )
