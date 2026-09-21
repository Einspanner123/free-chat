"""CPU probe orchestration only: HTTP/SSH observations are deterministic fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import grpc
import httpx
import pytest
import respx
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import AgentHints, ModelCapability, RequestProfile, WorkerCapabilities
from freechat_contracts.execution import ExecutionCommand, ExecutionReceipt, ExecutionStatus
from freechat_contracts.preparation import body_digest
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
        execution_endpoint="127.0.0.1:50052",
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
                supports_kv_offload=False,
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
        self.runtime: dict[str, Any] = {}
        self.request_headers: list[dict[str, str]] = []
        self.prepare_fault: str | None = None
        self.executions: list[ExecutionCommand] = []
        self.receipt_status = ExecutionStatus.COMPLETED
        self.running = 0

    def infer(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        self.request_headers.append(dict(request.headers))
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

    def prepare(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        result = {
            "preparation_id": "a" * 64,
            "worker_id": self.runtime["worker_id"],
            "generation": self.runtime["generation"],
            "engine_instance_id": self.runtime["engine_instance_id"],
            "budget": {
                "tenant_id": request.headers["x-freechat-internal-tenant"],
                "protocol": payload["protocol"],
                "body_sha256": body_digest(payload["request"]),
                "prompt_sha256": "b" * 64,
                "input_tokens": 100,
                "output_tokens": payload["request"]["max_tokens"],
                "expires_at": time.time() + 60,
            },
        }
        if self.prepare_fault in {"identity", "hash"}:
            if self.prepare_fault == "identity":
                result["engine_instance_id"] = "replacement"
            else:
                result["budget"]["body_sha256"] = "c" * 64
        elif self.prepare_fault == "capacity":
            result["budget"]["input_tokens"] = 4096
        elif self.prepare_fault == "expired":
            result["budget"]["expires_at"] = time.time() - 1
        return httpx.Response(200, json=result)

    def metrics(self, request: httpx.Request) -> httpx.Response:
        del request
        count = len(self.requests)
        values: dict[str, float] = {
            "num_requests_running": self.running,
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
async def observations(monkeypatch: pytest.MonkeyPatch, capabilities: Path) -> Any:
    engine = ObservationFixture()
    monkeypatch.setenv("FREECHAT_WORKER_TOKEN", "fixture-token-" * 4)
    caps = WorkerCapabilities.model_validate_json(capabilities.read_text())

    class ReceiptService(control_pb2_grpc.RequestExecutionServiceServicer):
        async def Observe(self, message: Any, context: Any) -> Any:
            assert (
                "authorization",
                "Bearer " + "fixture-token-" * 4,
            ) in context.invocation_metadata()
            command = ExecutionCommand.model_validate_json(message.command_json)
            engine.executions.append(command)
            receipt = ExecutionReceipt(
                command=command,
                observation_sequence=1,
                observed_at=datetime.now(UTC),
                status=engine.receipt_status,
                quiescent=True,
                admission_closed=True,
            )
            return control_pb2.RequestExecutionReceipt(receipt_json=receipt.model_dump_json())

    server = grpc.aio.server()
    control_pb2_grpc.add_RequestExecutionServiceServicer_to_server(  # type: ignore[no-untyped-call]
        ReceiptService(),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    caps = caps.model_copy(update={"execution_endpoint": f"127.0.0.1:{port}"})
    capabilities.write_text(caps.model_dump_json())
    engine.runtime = {
        "worker_id": caps.worker_id,
        "generation": caps.generation,
        "engine_instance_id": "fixture-engine",
        "capabilities": caps.model_dump(mode="json"),
        "capacity": {
            "num_blocks": 256,
            "block_size_tokens": 16,
            "block_bytes": 4096,
            "allocated_bytes": 256 * 4096,
            "max_context_tokens": 4096,
            "gpu_name": caps.gpu_name,
            "gpu_uuid": caps.gpu_id,
            "total_vram_bytes": caps.total_vram_bytes,
            "compute_capability": caps.compute_capability,
        },
    }
    process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"4096\n", b"")))
    subprocess = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", subprocess)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with respx.mock(assert_all_called=False) as router:
        router.get("http://probe.invalid:8000/health").respond(200)
        router.get("http://probe.invalid:8000/freechat/runtime").mock(
            side_effect=lambda _: httpx.Response(
                200, json={"observed_at": datetime.now(UTC).isoformat(), **engine.runtime}
            )
        )
        router.post("http://probe.invalid:8000/freechat/prepare").mock(side_effect=engine.prepare)
        router.get("http://probe.invalid:8000/metrics").mock(side_effect=engine.metrics)
        router.post("http://probe.invalid:8000/v1/chat/completions").mock(side_effect=engine.infer)
        try:
            yield engine, process, subprocess, router
        finally:
            await server.stop(None)


def artifacts(output: str) -> dict[str, Any]:
    records = {}
    for line in output.splitlines():
        if line.startswith('{"record_type":'):
            record = json.loads(line)
            records[record["name"]] = record
    return records


async def test_service_probe_uses_measured_budget_and_locality(
    capabilities: Path, observations: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    engine, _, subprocess, _ = observations
    await calibrate_service.run(arguments(capabilities))
    records = artifacts(capsys.readouterr().out)
    manifest = json.loads(records["manifest.json"]["payload"])
    routes = json.loads(records["routes.json"]["payload"])
    assert manifest["performance_claim_admissible"] is False
    assert [item["case"] for item in routes] == ["in-scope", "out-of-scope"]
    assert routes[0]["decision"]["selected"]["estimate_available"]
    assert routes[0]["decision"]["strategy"] == "lifecycle-aware"
    assert routes[1]["decision"]["fallback_reason"] == "candidate_cost_unavailable"
    assert routes[0]["decision"]["reserved_kv_bytes_per_rank"] == 32768
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


async def test_telemetry_probe_routes_without_requiring_offload(
    capabilities: Path, observations: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    engine, _, subprocess, _ = observations
    await telemetry_probe.run(arguments(capabilities))
    output = capsys.readouterr().out
    assert set(artifacts(output)) >= {"before.prom", "after.prom"}
    result = json.loads(output[output.index("{\n") :])
    assert result["performance_claim_admissible"] is False
    assert result["decision"]["kv_transfer"]["enabled"] is False
    assert result["records"][-1]["telemetry"]["kv_admission_available_bytes_per_rank"] == 255 * 4096
    assert result["records"][-1]["telemetry"]["cache_store_bytes_per_second"] is None
    assert engine.requests[0].get("kv_transfer_params", {}).get("max_offload_tokens", 0) == 0
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


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "worker",
        "generation",
        "engine",
        "node",
        "model",
        "gpu",
        "geometry",
        "stale",
        "future",
    ],
)
async def test_runtime_mismatch_is_rejected_before_inference(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
    case: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine, _, _, _ = observations
    runtime = engine.runtime
    if case == "missing":
        runtime["capacity"] = None
    elif case in {"worker", "generation", "engine"}:
        key = {"worker": "worker_id", "generation": "generation", "engine": "engine_instance_id"}[
            case
        ]
        runtime[key] = 2 if case == "generation" else "other"
    elif case == "node":
        runtime["capabilities"]["node_id"] = "other-node"
    elif case == "model":
        runtime["capabilities"]["models"][0]["revision"] = "other-revision"
    elif case == "gpu":
        runtime["capacity"]["gpu_uuid"] = "other-gpu"
    elif case == "geometry":
        runtime["capacity"]["block_size_tokens"] = 32
    else:
        runtime["observed_at"] = (
            datetime.now(UTC) + timedelta(seconds=60 if case == "future" else -60)
        ).isoformat()
    with pytest.raises(ValueError, match="runtime_"):
        await module.run(arguments(capabilities))
    assert engine.requests == []
    assert "manifest.json" not in artifacts(capsys.readouterr().out)


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
async def test_probe_prepares_and_confirms_every_direct_request(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
) -> None:
    engine, _, _, router = observations
    await module.run(arguments(capabilities))
    assert len(engine.executions) == len(engine.requests) > 0
    assert len({command.request_id for command in engine.executions}) == len(engine.executions)
    for headers, command in zip(engine.request_headers, engine.executions, strict=True):
        assert headers["x-freechat-internal-preparation"] == "a" * 64
        assert int(headers["x-freechat-internal-reserved-kv-bytes"]) == (
            28672 if module is telemetry_probe else 32768
        )
        assert headers["x-freechat-internal-decision-id"] == command.decision_id
        assert command.action.value == "query"
    for call in router.calls:
        assert call.request.headers.get("x-freechat-worker-token") == "fixture-token-" * 4


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
@pytest.mark.parametrize("fault", ["identity", "hash", "capacity", "expired"])
async def test_probe_rejects_bad_preparation_before_dispatch(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
    fault: str,
) -> None:
    engine, _, _, _ = observations
    engine.prepare_fault = fault
    with pytest.raises(ValueError, match="probe_preparation"):
        await module.run(arguments(capabilities))
    assert engine.requests == []


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
async def test_aborted_execution_cannot_produce_probe_success(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine, _, _, _ = observations
    engine.receipt_status = ExecutionStatus.ABORTED
    with pytest.raises(ValueError, match="probe_execution_not_completed"):
        await module.run(arguments(capabilities))
    assert "manifest.json" not in artifacts(capsys.readouterr().out)


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
async def test_probe_can_measure_gpu_in_same_container_namespace(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
) -> None:
    _, _, process, _ = observations
    args = arguments(capabilities)
    args.gpu_host = None
    await module.run(args)
    assert process.await_args is not None
    assert process.await_args.args[:2] == ("nvidia-smi", "--id=0")


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
async def test_busy_worker_is_rejected_before_probe_inference(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
) -> None:
    engine, _, _, _ = observations
    engine.running = 1
    with pytest.raises(ValueError, match="runtime_requires_matching_idle_observation"):
        await module.run(arguments(capabilities))
    assert engine.requests == []


@pytest.mark.parametrize("module", [calibrate_service, telemetry_probe])
@pytest.mark.parametrize("change", ["replacement", "pool"])
async def test_runtime_change_during_inference_cannot_produce_calibration(
    capabilities: Path,
    observations: Any,
    module: ModuleType,
    change: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine, _, _, router = observations
    original = engine.infer

    def replace_runtime(request: httpx.Request) -> httpx.Response:
        result: httpx.Response = original(request)
        if change == "replacement":
            engine.runtime["engine_instance_id"] = "replacement"
        else:
            engine.runtime["capacity"]["num_blocks"] += 1
            engine.runtime["capacity"]["allocated_bytes"] += 4096
        return result

    router.post("http://probe.invalid:8000/v1/chat/completions").mock(side_effect=replace_runtime)
    with pytest.raises(ValueError, match="runtime_"):
        await module.run(arguments(capabilities))
    assert "manifest.json" not in artifacts(capsys.readouterr().out)


@pytest.mark.parametrize(
    "gpu_id",
    [
        "712545e0-1701-6651-37f5-824cdb4368e9",
        "GPU-712545e0-1701-6651-37f5-824cdb4368e9",
    ],
)
async def test_nvidia_smi_uses_prefixed_physical_uuid(
    capabilities: Path,
    observations: Any,
    gpu_id: str,
) -> None:
    from benchmarks.probe_runtime import observe_free_vram

    _, _, process, _ = observations
    caps = WorkerCapabilities.model_validate_json(capabilities.read_text())
    caps = caps.model_copy(update={"gpu_id": gpu_id})
    assert await observe_free_vram(caps, None) == 4 * 1024**3
    assert process.await_args is not None
    assert process.await_args.args[1] == "--id=GPU-712545e0-1701-6651-37f5-824cdb4368e9"
