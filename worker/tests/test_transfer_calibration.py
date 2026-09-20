import pytest
from freechat_worker.transfer_calibration import observe_transfer, summarize_transfers


def payload(count: int, *, hits: int = 0, size: int = 4096) -> str:
    values = {
        "num_requests_running": 0,
        "num_requests_waiting": 0,
        "num_preemptions_total": 0,
        "request_prompt_tokens_count": count,
        "external_prefix_cache_hits_total": hits,
        "kv_offload_load_size_sum": size * count,
        "kv_offload_load_size_count": count,
        "kv_offload_load_time_total": count * 0.01,
    }
    return "\n".join(
        f'vllm:{name}{{model_name="model",engine="0"}} {value}' for name, value in values.items()
    )


def test_measured_load_requires_external_hit_correlation() -> None:
    observation = observe_transfer(payload(1), payload(2, hits=16), "model", "load")
    assert observation.transferred_bytes == 4096
    assert observation.operations == 1
    summary = summarize_transfers([observation] * 3)
    assert summary["bytes_per_service_second"] == pytest.approx(409600)


@pytest.mark.parametrize("kind", ["no_hit", "concurrent", "reset", "missing", "no_bytes"])
def test_invalid_window_cannot_become_transfer_rate(kind: str) -> None:
    after = payload(2, hits=16)
    if kind == "no_hit":
        after = payload(2)
    elif kind == "concurrent":
        after = payload(3, hits=16)
    elif kind == "reset":
        after = payload(0)
    elif kind == "missing":
        after = ""
    elif kind == "no_bytes":
        after = payload(2, hits=16, size=2048)
    with pytest.raises(ValueError):
        observe_transfer(payload(1), after, "model", "load")
