from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from freechat_worker.capacity import capacity_from_rank, measure_capacity


def layout() -> dict[str, Any]:
    return {
        "num_blocks": 11,
        "groups": [{
            "kind": "FullAttentionSpec", "block_size": 16, "page_bytes": 512,
            "layer_count": 24, "eagle": False, "inner_kinds": [], "inner_blocks": [],
            "inner_layers": [], "layers": [f"layer-{i}" for i in range(24)],
        }],
        "allocated_bytes": 11 * 512 * 24, "max_context_tokens": 2048,
        "gpu_name": "test", "total_vram_bytes": 1024**3, "compute_capability": "8.6",
    }


@pytest.mark.parametrize("kind", ["FullAttentionSpec", "MLAAttentionSpec"])
def test_measured_pages_include_every_layer_and_exclude_null_block(kind: str) -> None:
    raw = layout()
    raw["groups"][0]["kind"] = kind
    capacity = capacity_from_rank(raw)
    assert capacity.bytes_per_token == 768
    assert capacity.usable_bytes == 10 * 512 * 24
    assert capacity.allocated_bytes == 11 * 512 * 24
    assert capacity.basis == "gross_engine_pool_before_scheduler_reservations"


def test_uniform_spec_is_already_aggregated_not_multiplied_twice() -> None:
    raw = layout()
    group = raw["groups"][0]
    group.update(
        kind="UniformTypeKVCacheSpecs", page_bytes=512 * 24,
        inner_kinds=["FullAttentionSpec"] * 24, inner_blocks=[16] * 24,
        inner_layers=group["layers"],
    )
    assert capacity_from_rank(raw).bytes_per_token == 768


@pytest.mark.parametrize(
    "failure", ["allocation", "null_only", "hybrid", "sliding", "eagle", "mixed", "fractional"]
)
def test_unsupported_or_inconsistent_geometry_is_not_advertised(failure: str) -> None:
    raw = layout()
    group = raw["groups"][0]
    if failure == "allocation":
        raw["allocated_bytes"] += 1
    elif failure == "null_only":
        raw["num_blocks"] = 1
    elif failure == "hybrid":
        raw["groups"].append(deepcopy(group))
    elif failure == "sliding":
        group["kind"] = "SlidingWindowSpec"
    elif failure == "eagle":
        group["eagle"] = True
    elif failure == "mixed":
        group.update(kind="UniformTypeKVCacheSpecs", inner_kinds=["MambaSpec"])
    else:
        group["block_size"] = 17
    with pytest.raises(ValueError):
        capacity_from_rank(raw)


async def test_measurement_uses_rank_report_and_rejects_missing_rank() -> None:
    reports = [layout()]

    async def rpc(method: Any, timeout: int) -> Any:
        assert method == "freechat_capacity" and timeout == 15
        return reports

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
            ),
        ),
        collective_rpc=rpc,
    )
    assert (await measure_capacity(engine)).bytes_per_token == 768
    reports.clear()
    with pytest.raises(ValueError, match="exactly one"):
        await measure_capacity(engine)
    engine.vllm_config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="parallel planner"):
        await measure_capacity(engine)
