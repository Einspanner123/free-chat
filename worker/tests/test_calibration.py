import pytest
from freechat_worker.calibration import HISTOGRAMS, observe_request


def metrics(*, count: int, cached: bool = False, busy: bool = False) -> str:
    values: dict[str, float] = {
        "num_requests_running": int(busy),
        "num_requests_waiting": 0,
        "num_preemptions_total": 0,
    }
    totals = [0.01, 0.03, 90 if cached else 100, 100, 16]
    for name, total in zip(HISTOGRAMS, totals, strict=True):
        values[f"{name}_count"] = count
        values[f"{name}_sum"] = total * count
    return "\n".join(
        f'vllm:{name}{{model_name="model",engine="0"}} {value}' for name, value in values.items()
    )


def test_isolated_uncached_observation() -> None:
    observation = observe_request(metrics(count=1), metrics(count=2), "model")
    assert observation.input_tokens == 100
    assert observation.output_tokens == 16
    assert observation.prefill_seconds == pytest.approx(0.01)


@pytest.mark.parametrize("kind", ["cached", "concurrent", "reset", "busy", "missing"])
def test_invalid_observations_are_not_calibrations(kind: str) -> None:
    after = metrics(count=2)
    if kind == "cached":
        after = metrics(count=2, cached=True)
    elif kind == "concurrent":
        after = metrics(count=3)
    elif kind == "reset":
        after = metrics(count=0)
    elif kind == "busy":
        after = metrics(count=2, busy=True)
    elif kind == "missing":
        after = ""
    with pytest.raises(ValueError):
        observe_request(metrics(count=1), after, "model")
