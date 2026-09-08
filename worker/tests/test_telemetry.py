from datetime import UTC, datetime

import pytest
from freechat_contracts import ModelCapability, WorkerCapabilities, WorkerTelemetry
from freechat_worker.telemetry import TelemetryCollector


def collector() -> TelemetryCollector:
    return TelemetryCollector(
        WorkerCapabilities(
            worker_id="worker", generation=1, endpoint="http://worker:8000",
            node_id="node", gpu_id="0", gpu_name="test", compute_capability="8.6",
            total_vram_bytes=1000, p2p_domain="node", network_domain="lan",
            models=(ModelCapability(
                model_id="model", revision="sha", tokenizer_revision="sha",
                architecture="dense", attention="gqa", max_context_tokens=4096, dtype="half",
            ),),
        ),
        "engine-instance",
    )


def payload(stored: int, seconds: float) -> str:
    values = {
        "num_requests_running": 2,
        "num_requests_waiting": 3,
        "kv_cache_usage_perc": 0.25,
        "kv_offload_store_size_sum": stored,
        "kv_offload_store_time_total": seconds,
    }
    return "\n".join(
        f'vllm:{name}{{model_name="model",engine="0"}} {value}'
        for name, value in values.items()
    )


def test_transfer_rate_uses_window_delta_and_idle_window_expires_estimate() -> None:
    probe = collector()
    def observe(data: str) -> WorkerTelemetry:
        return probe.collect(data, free_vram_bytes=500, observed_at=datetime.now(UTC))
    first = observe(payload(1000, 2))
    assert first.cache_store_bytes_per_second is None
    second = observe(payload(3000, 3))
    assert second.cache_store_bytes_per_second == 2000
    assert second.kv_cache_usage_ratio == 0.25
    assert second.kv_cache_free_bytes is None
    assert second.active_requests == 2
    assert second.queue_depth == 3
    assert observe(payload(3000, 3)).cache_store_bytes_per_second is None


def test_counter_reset_requires_new_generation() -> None:
    probe = collector()
    probe.collect(payload(3000, 3), free_vram_bytes=500, observed_at=datetime.now(UTC))
    with pytest.raises(ValueError, match="re-register"):
        probe.collect(payload(0, 0), free_vram_bytes=500, observed_at=datetime.now(UTC))


def test_missing_metrics_cannot_create_healthy_snapshot() -> None:
    with pytest.raises(ValueError, match="missing"):
        collector().collect("", free_vram_bytes=500, observed_at=datetime.now(UTC))


def test_unrelated_multidimensional_metrics_do_not_corrupt_snapshot() -> None:
    data = payload(0, 0) + '\nvllm:other{engine="0",model_name="model",tier="cpu"} 1\n'
    data += 'vllm:other{engine="0",model_name="model",tier="gpu"} 2\n'
    snapshot = collector().collect(data, free_vram_bytes=500, observed_at=datetime.now(UTC))
    assert snapshot.queue_depth == 3
