import pytest

from benchmarks.analyze_offload_boundary import paired_p95, percentile


def test_linear_percentile() -> None:
    assert percentile([10.0, 0.0], .95) == pytest.approx(9.5)


def test_paired_interval_keeps_constant_difference() -> None:
    result = paired_p95([10.0, 20.0, 30.0], [8.0, 18.0, 28.0])
    assert result["paired_bootstrap_95_interval"] == pytest.approx([-2.0, -2.0])
    assert result["difference_candidate_minus_baseline"] == pytest.approx(-2.0)


def test_unpaired_samples_rejected() -> None:
    with pytest.raises(ValueError):
        paired_p95([1.0, 2.0], [1.0])
