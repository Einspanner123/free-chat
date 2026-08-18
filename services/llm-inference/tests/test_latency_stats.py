"""Pure-unit tests for latency_stats (no torch / no model)."""

import pytest

from optimization.latency_stats import (
    percentile,
    summarize_latencies,
    summarize_speedups,
)


def test_percentile_endpoints():
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 0) == 10
    assert percentile(vals, 50) == 30
    assert percentile(vals, 100) == 50


def test_percentile_interpolates():
    vals = [0, 10, 20, 30, 40]
    # k = (5-1)*(25/100) = 1 -> s[1] = 10
    assert percentile(vals, 25) == 10
    # q=40 -> k=1.6 -> 10 + (20-10)*0.6 = 16
    assert abs(percentile(vals, 40) - 16.0) < 1e-9
    # q=50 -> k=2 -> s[2] = 20
    assert percentile(vals, 50) == 20


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_summarize_latencies_tail_ordering():
    times = [100, 200, 150, 1000, 120, 130, 110, 900, 140, 160]
    s = summarize_latencies(times)
    assert s["n"] == 10
    # tail percentiles are monotonic and bracket the median
    assert s["p99_ms"] >= s["p95_ms"] >= s["p50_ms"]
    assert s["min_ms"] <= s["p50_ms"] <= s["max_ms"]
    assert s["max_ms"] == 1000
    assert s["min_ms"] == 100
    # mean of the 10 values (right-skewed: mean can exceed median, so not ordered vs p50)
    assert abs(s["mean_ms"] - (sum(times) / 10)) < 1e-6


def test_summarize_latencies_empty():
    assert summarize_latencies([]) == {"n": 0}


def test_summarize_speedups_tail_ordering():
    sp = [1.0, 2.97, 2.78, 2.5, 3.1]
    s = summarize_speedups(sp)
    assert s["n"] == 5
    assert s["p99_x"] >= s["p95_x"] >= s["p50_x"]
    assert s["max_x"] == 3.1
    assert s["min_x"] == 1.0
