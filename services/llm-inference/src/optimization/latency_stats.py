"""
Latency statistics helpers for inference benchmarks.

Production serving cares about tail latency (p95/p99), not just mean
throughput. These pure functions compute percentiles and a compact
summary so benchmark scripts report tail latency alongside means.

No torch / model dependency — safe to unit-test on CPU.
"""

import math
from typing import Dict, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy's default method).

    ``q`` is in [0, 100]. Raises ValueError on empty input or out-of-range q.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    s = sorted(float(v) for v in values)
    if q == 0.0:
        return s[0]
    if q == 100.0:
        return s[-1]
    k = (len(s) - 1) * (q / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def summarize_latencies(times_ms: Sequence[float]) -> Dict:
    """Compact latency summary (milliseconds) with tail percentiles."""
    if not times_ms:
        return {"n": 0}
    s = sorted(float(v) for v in times_ms)
    mean = sum(s) / len(s)
    return {
        "n": len(s),
        "mean_ms": round(mean, 2),
        "p50_ms": round(percentile(s, 50), 2),
        "p95_ms": round(percentile(s, 95), 2),
        "p99_ms": round(percentile(s, 99), 2),
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
    }


def summarize_speedups(speedups: Sequence[float]) -> Dict:
    """Compact speedup summary (x) with tail percentiles."""
    if not speedups:
        return {"n": 0}
    s = sorted(float(v) for v in speedups)
    return {
        "n": len(s),
        "mean_x": round(sum(s) / len(s), 2),
        "p50_x": round(percentile(s, 50), 2),
        "p95_x": round(percentile(s, 95), 2),
        "p99_x": round(percentile(s, 99), 2),
        "min_x": round(s[0], 2),
        "max_x": round(s[-1], 2),
    }
