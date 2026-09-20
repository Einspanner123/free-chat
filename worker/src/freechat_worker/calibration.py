"""Validate isolated uncached request observations before fitting service rates."""

from datetime import UTC, datetime, timedelta

from freechat_contracts import CostCalibration, WorkerCapabilities
from pydantic import BaseModel, ConfigDict, Field

from freechat_worker.telemetry import samples

HISTOGRAMS = (
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "request_prefill_kv_computed_tokens",
    "request_prompt_tokens",
    "request_generation_tokens",
)
CALIBRATION_METRICS = {
    f"vllm:{name}_{suffix}" for name in HISTOGRAMS for suffix in ("count", "sum")
} | {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_preemptions_total"}


class ServiceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(ge=2)
    prefill_seconds: float = Field(gt=0)
    decode_seconds: float = Field(gt=0)


def observe_request(before: str, after: str, model: str) -> ServiceObservation:
    start = samples(before, model, "0", names=CALIBRATION_METRICS)
    end = samples(after, model, "0", names=CALIBRATION_METRICS)
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        if start.get(name) != 0 or end.get(name) != 0:
            raise ValueError("calibration requires idle boundaries")
    preempt = "vllm:num_preemptions_total"
    if start.get(preempt) is None or end.get(preempt) != start[preempt]:
        raise ValueError("missing or changed preemption counter")
    deltas: dict[str, float] = {}
    for name in HISTOGRAMS:
        count = f"vllm:{name}_count"
        total = f"vllm:{name}_sum"
        if any(key not in start or key not in end for key in (count, total)):
            raise ValueError("warm-up required to initialize request histograms")
        if end[count] - start[count] != 1:
            raise ValueError("request cohort is not isolated")
        delta = end[total] - start[total]
        if delta <= 0:
            raise ValueError("invalid counter delta or engine reset")
        deltas[name] = delta
    prompt = deltas["request_prompt_tokens"]
    computed = deltas["request_prefill_kv_computed_tokens"]
    output = deltas["request_generation_tokens"]
    if not prompt.is_integer() or not output.is_integer() or computed != prompt:
        raise ValueError("calibration requires uncached integral token observations")
    return ServiceObservation(
        input_tokens=int(prompt),
        output_tokens=int(output),
        prefill_seconds=deltas["request_prefill_time_seconds"],
        decode_seconds=deltas["request_decode_time_seconds"],
    )


def fit_service_profile(
    observations: list[ServiceObservation],
    *,
    capabilities: WorkerCapabilities,
    engine_instance_id: str,
    image_identity: str,
    artifact_sha256: str,
    observed_at: datetime | None = None,
) -> CostCalibration:
    if len(observations) < 3 or len(capabilities.models) != 1:
        raise ValueError("at least three samples of one served model are required")
    now = observed_at or datetime.now(UTC)
    return CostCalibration(
        calibration_id=artifact_sha256,
        worker_id=capabilities.worker_id,
        worker_generation=capabilities.generation,
        engine_instance_id=engine_instance_id,
        model=capabilities.models[0],
        image_identity=image_identity,
        observed_at=now,
        expires_at=now + timedelta(hours=1),
        artifact_sha256=artifact_sha256,
        sample_count=len(observations),
        input_tokens_min=min(item.input_tokens for item in observations),
        input_tokens_max=max(item.input_tokens for item in observations),
        output_tokens_min=min(item.output_tokens for item in observations),
        output_tokens_max=max(item.output_tokens for item in observations),
        max_concurrent_requests=1,
        prefill_tokens_per_second=(
            sum(item.input_tokens for item in observations)
            / sum(item.prefill_seconds for item in observations)
        ),
        decode_tokens_per_second=(
            sum(item.output_tokens - 1 for item in observations)
            / sum(item.decode_seconds for item in observations)
        ),
    )
