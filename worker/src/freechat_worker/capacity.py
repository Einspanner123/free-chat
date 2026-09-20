"""Measure allocated engine KV geometry; never infer it from free CUDA memory."""

from __future__ import annotations

import importlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EngineCapacity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    basis: Literal["gross_engine_pool_before_scheduler_reservations"] = (
        "gross_engine_pool_before_scheduler_reservations"
    )
    num_blocks: int = Field(gt=1)
    block_size_tokens: int = Field(gt=0)
    block_bytes: int = Field(gt=0)
    allocated_bytes: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)
    gpu_name: str = Field(min_length=1)
    gpu_uuid: str | None = None
    total_vram_bytes: int = Field(gt=0)
    compute_capability: str = Field(min_length=1)

    @model_validator(mode="after")
    def physical_layout(self) -> EngineCapacity:
        if self.allocated_bytes != self.num_blocks * self.block_bytes:
            raise ValueError("KV allocation does not match measured block geometry")
        if self.block_bytes % self.block_size_tokens:
            raise ValueError("KV geometry requires integral bytes per token")
        if self.allocated_bytes > self.total_vram_bytes:
            raise ValueError("KV allocation exceeds physical GPU memory")
        return self

    @property
    def bytes_per_token(self) -> int:
        return self.block_bytes // self.block_size_tokens

    @property
    def usable_bytes(self) -> int:
        # The pinned BlockPool removes one null block from its free queue.
        return (self.num_blocks - 1) * self.block_bytes


def collect_rank_layout(worker: Any) -> dict[str, Any]:
    """Executed by vLLM's native collective RPC after cache allocation."""
    torch = importlib.import_module("torch")
    config = worker.model_runner.kv_cache_config
    groups = []
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        inner = getattr(spec, "kv_cache_specs", None)
        groups.append(
            {
                "kind": type(spec).__name__,
                "block_size": spec.block_size,
                "page_bytes": spec.page_size_bytes,
                "layer_count": len(group.layer_names),
                "eagle": group.is_eagle_group,
                "inner_kinds": [] if inner is None else [type(s).__name__ for s in inner.values()],
                "inner_blocks": [] if inner is None else [s.block_size for s in inner.values()],
                "inner_layers": [] if inner is None else list(inner),
                "layers": list(group.layer_names),
            }
        )
    props = torch.cuda.get_device_properties(worker.device)
    return {
        "num_blocks": config.num_blocks,
        "groups": groups,
        "allocated_bytes": sum(tensor.size for tensor in config.kv_cache_tensors),
        "max_context_tokens": worker.model_config.max_model_len,
        "gpu_name": props.name,
        "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "total_vram_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def capacity_from_rank(raw: dict[str, Any]) -> EngineCapacity:
    groups = raw["groups"]
    if len(groups) != 1 or groups[0]["eagle"]:
        raise ValueError("KV measurement requires one non-speculative full-attention group")
    group = groups[0]
    supported = {"FullAttentionSpec", "MLAAttentionSpec", "TQFullAttentionSpec"}
    multiplier = group["layer_count"]
    if group["kind"] == "UniformTypeKVCacheSpecs":
        if (
            not group["inner_kinds"]
            or not set(group["inner_kinds"]) <= supported
            or set(group["inner_blocks"]) != {group["block_size"]}
            or set(group["inner_layers"]) != set(group["layers"])
        ):
            raise ValueError("unsupported mixed KV layout")
        multiplier = 1  # Upstream page_size_bytes already sums all inner layers.
    elif group["kind"] not in supported:
        raise ValueError("unsupported KV layout")
    if group["layer_count"] < 1:
        raise ValueError("empty KV layer group")
    return EngineCapacity(
        num_blocks=raw["num_blocks"],
        block_size_tokens=group["block_size"],
        block_bytes=group["page_bytes"] * multiplier,
        allocated_bytes=raw["allocated_bytes"],
        max_context_tokens=raw["max_context_tokens"],
        gpu_name=raw["gpu_name"],
        gpu_uuid=raw.get("gpu_uuid"),
        total_vram_bytes=raw["total_vram_bytes"],
        compute_capability=raw["compute_capability"],
    )


class CapacityExtension:
    """Named vLLM Worker extension; no callable serialization across processes."""

    device: Any  # Provided by vLLM when it composes the Worker extension.

    def freechat_free_memory(self) -> int:
        torch = importlib.import_module("torch")
        return int(torch.cuda.mem_get_info(self.device)[0])

    def freechat_capacity(self) -> dict[str, Any]:
        return collect_rank_layout(self)


async def measure_capacity(engine: Any) -> EngineCapacity:
    parallel = engine.vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
    ):
        raise ValueError("rank aggregation requires a verified parallel planner")
    ranks = await engine.collective_rpc("freechat_capacity", timeout=15)
    if not isinstance(ranks, list) or len(ranks) != 1:
        raise ValueError("expected exactly one physical rank report")
    return capacity_from_rank(ranks[0])
