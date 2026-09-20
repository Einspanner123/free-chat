"""Size-scoped native KV transfer observations, never inferred from free VRAM."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from freechat_worker.telemetry import samples

Direction = Literal["store", "load"]
TRANSFER_METRICS = {
    f"vllm:kv_offload_{direction}_{suffix}"
    for direction in ("store", "load")
    for suffix in ("size_sum", "size_count", "time_total")
} | {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:request_prompt_tokens_count",
    "vllm:external_prefix_cache_hits_total",
}


class TransferObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    direction: Direction
    transferred_bytes: int = Field(gt=0)
    service_seconds: float = Field(gt=0)
    operations: int = Field(gt=0)
    external_hit_tokens: int = Field(ge=0)


def observe_transfer(
    before: str, after: str, model: str, direction: Direction
) -> TransferObservation:
    start = samples(before, model, "0", names=TRANSFER_METRICS)
    end = samples(after, model, "0", names=TRANSFER_METRICS)
    for name in ("num_requests_running", "num_requests_waiting"):
        if start.get(f"vllm:{name}") != 0 or end.get(f"vllm:{name}") != 0:
            raise ValueError("transfer calibration requires idle boundaries")
    for key, value in start.items():
        if key not in end or end[key] < value:
            raise ValueError("counter disappeared or reset")
    for name in (
        "num_preemptions_total",
        "request_prompt_tokens_count",
        "external_prefix_cache_hits_total",
    ):
        if f"vllm:{name}" not in start or f"vllm:{name}" not in end:
            raise ValueError("missing request accounting")
    if end["vllm:num_preemptions_total"] != start["vllm:num_preemptions_total"]:
        raise ValueError("preempted request cannot calibrate isolated transfer")
    if end["vllm:request_prompt_tokens_count"] - start["vllm:request_prompt_tokens_count"] != 1:
        raise ValueError("transfer request cohort is not isolated")
    keys = [
        f"vllm:kv_offload_{direction}_{suffix}"
        for suffix in ("size_sum", "time_total", "size_count")
    ]
    if any(key not in start or key not in end for key in keys):
        raise ValueError("warm-up required for transfer counters")
    size, duration, count = (end[key] - start[key] for key in keys)
    hits = (
        end["vllm:external_prefix_cache_hits_total"]
        - start["vllm:external_prefix_cache_hits_total"]
    )
    if not all(value.is_integer() for value in (size, count, hits)):
        raise ValueError("non-integral transfer accounting")
    if direction == "load" and hits <= 0:
        raise ValueError("load requires correlated external prefix hits")
    return TransferObservation(
        direction=direction,
        transferred_bytes=int(size),
        service_seconds=duration,
        operations=int(count),
        external_hit_tokens=int(hits),
    )


def summarize_transfers(observations: list[TransferObservation]) -> dict[str, float | int | str]:
    if len(observations) < 3 or len({item.direction for item in observations}) != 1:
        raise ValueError("requires at least three observations of one direction")
    return {
        "direction": observations[0].direction,
        "sample_count": len(observations),
        "bytes_min": min(item.transferred_bytes for item in observations),
        "bytes_max": max(item.transferred_bytes for item in observations),
        "bytes_per_service_second": sum(item.transferred_bytes for item in observations)
        / sum(item.service_seconds for item in observations),
    }
