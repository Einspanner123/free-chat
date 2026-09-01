from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from freechat_trace_replay.artifacts import ArtifactManifest, ParquetTraceArchive


class RequestMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    trial_id: int = Field(ge=0)
    workload_id: str
    strategy: str
    model_revision: str
    task_id: str
    harness_id: str
    phase: str
    started_ms: float = Field(ge=0)
    finished_ms: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    success: bool

    @model_validator(mode="after")
    def validate_measurement(self) -> RequestMeasurement:
        if self.finished_ms < self.started_ms:
            raise ValueError("finished_ms must not precede started_ms")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached_tokens must not exceed input_tokens")
        return self


class GpuUtilizationSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    gpu_id: str
    observed_ms: float = Field(ge=0)
    sample_interval_ms: float = Field(gt=0)
    utilization_percent: float = Field(ge=0, le=100)


class FaultMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task_id: str
    fault_id: str
    injected_ms: float = Field(ge=0)
    recovered_ms: float | None = Field(default=None, ge=0)
    unique_events: int = Field(default=1, ge=0)
    duplicate_events: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_recovery(self) -> FaultMeasurement:
        if self.recovered_ms is not None and self.recovered_ms < self.injected_ms:
            raise ValueError("recovered_ms must not precede injected_ms")
        if self.unique_events + self.duplicate_events == 0:
            raise ValueError("fault event delivery denominator must be positive")
        return self


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    trial_id: int
    workload_id: str
    strategy: str
    model_revision: str
    harnesses: tuple[str, ...]
    completed_tasks: int
    failed_tasks: int
    task_latency_p95_ms: float
    prefix_cache_token_hit_rate: float
    resume_repeated_prefill_tokens: int
    tasks_per_second_per_gpu: float
    mean_gpu_utilization_percent: float
    fault_recovery_success_rate: float | None
    recovery_time_p95_ms: float | None
    duplicate_event_rate: float | None


@dataclass(frozen=True, slots=True)
class PairedLatencyComparison:
    task_count: int
    p95_reduction_percent: float
    mean_reduction_percent: float
    mean_reduction_ci95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EstimateWithCI:
    estimate: float
    ci95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PairedRunComparison:
    trial_count: int
    task_latency_p95_reduction_percent: EstimateWithCI
    prefix_cache_hit_lift_points: EstimateWithCI
    resume_repeated_prefill_reduction_percent: EstimateWithCI
    tasks_per_second_per_gpu_improvement_percent: EstimateWithCI
    gpu_utilization_lift_points: EstimateWithCI
    fault_recovery_lift_points: EstimateWithCI | None
    recovery_time_p95_reduction_percent: EstimateWithCI | None


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactSet:
    requests: ArtifactManifest
    gpu_samples: ArtifactManifest
    faults: ArtifactManifest | None
    summary: ArtifactManifest
    resolved_config: ArtifactManifest


def archive_benchmark_run(
    archive: ParquetTraceArchive,
    requests: list[RequestMeasurement],
    gpu_samples: list[GpuUtilizationSample],
    faults: list[FaultMeasurement],
    resolved_config: dict[str, str | int | float | bool],
) -> BenchmarkArtifactSet:
    """Persist the complete metric inputs before returning a derived summary."""
    summary = summarize_run(requests, gpu_samples, faults)
    run_id = summary.run_id
    return BenchmarkArtifactSet(
        requests=archive.write(
            f"benchmarks/{run_id}/requests",
            [item.model_dump(mode="json") for item in requests],
        ),
        gpu_samples=archive.write(
            f"benchmarks/{run_id}/gpu-samples",
            [item.model_dump(mode="json") for item in gpu_samples],
        ),
        faults=(
            archive.write(
                f"benchmarks/{run_id}/faults",
                [item.model_dump(mode="json") for item in faults],
            )
            if faults
            else None
        ),
        summary=archive.write(
            f"benchmarks/{run_id}/summary",
            [summary.model_dump(mode="json")],
        ),
        resolved_config=archive.write(
            f"benchmarks/{run_id}/resolved-config",
            [{"run_id": run_id, **resolved_config}],
        ),
    )


def summarize_run(
    requests: list[RequestMeasurement],
    gpu_samples: list[GpuUtilizationSample],
    faults: list[FaultMeasurement],
) -> BenchmarkSummary:
    if not requests:
        raise ValueError("at least one request measurement is required")
    identity = {
        (item.run_id, item.trial_id, item.workload_id, item.strategy, item.model_revision)
        for item in requests
    }
    if len(identity) != 1:
        raise ValueError("a summary cannot mix run, workload, strategy, or model identity")
    run_id, trial_id, workload_id, strategy, model_revision = next(iter(identity))
    if any(item.run_id != run_id for item in gpu_samples):
        raise ValueError("GPU samples belong to a different run")
    if any(item.run_id != run_id for item in faults):
        raise ValueError("fault measurements belong to a different run")

    by_task: dict[str, list[RequestMeasurement]] = defaultdict(list)
    for item in requests:
        by_task[item.task_id].append(item)
    successful_latencies: list[float] = []
    failed_tasks = 0
    for task_requests in by_task.values():
        if all(item.success for item in task_requests):
            successful_latencies.append(
                max(item.finished_ms for item in task_requests)
                - min(item.started_ms for item in task_requests)
            )
        else:
            failed_tasks += 1
    if not successful_latencies:
        raise ValueError("the run has no completed tasks")

    total_input = sum(item.input_tokens for item in requests)
    cached_tokens = sum(item.cached_tokens for item in requests)
    resume_repeated = sum(
        item.input_tokens - item.cached_tokens for item in requests if item.phase == "resume"
    )
    wall_ms = max(item.finished_ms for item in requests) - min(item.started_ms for item in requests)
    gpu_count = len({item.gpu_id for item in gpu_samples})
    if wall_ms <= 0 or gpu_count == 0:
        raise ValueError("throughput requires positive wall time and GPU samples")
    sampled_ms = sum(item.sample_interval_ms for item in gpu_samples)
    utilization = (
        sum(item.utilization_percent * item.sample_interval_ms for item in gpu_samples) / sampled_ms
    )

    recovered = [item for item in faults if item.recovered_ms is not None]
    recovery_rate = len(recovered) / len(faults) if faults else None
    recovery_p95 = (
        _percentile(
            [
                recovered_ms - item.injected_ms
                for item in recovered
                if (recovered_ms := item.recovered_ms) is not None
            ],
            0.95,
        )
        if recovered
        else None
    )
    delivered_events = sum(item.unique_events + item.duplicate_events for item in faults)
    duplicate_rate = (
        sum(item.duplicate_events for item in faults) / delivered_events if faults else None
    )
    return BenchmarkSummary(
        run_id=run_id,
        trial_id=trial_id,
        workload_id=workload_id,
        strategy=strategy,
        model_revision=model_revision,
        harnesses=tuple(sorted({item.harness_id for item in requests})),
        completed_tasks=len(successful_latencies),
        failed_tasks=failed_tasks,
        task_latency_p95_ms=_percentile(successful_latencies, 0.95),
        prefix_cache_token_hit_rate=(cached_tokens / total_input if total_input else 0.0),
        resume_repeated_prefill_tokens=resume_repeated,
        tasks_per_second_per_gpu=(len(successful_latencies) / (wall_ms / 1000) / gpu_count),
        mean_gpu_utilization_percent=utilization,
        fault_recovery_success_rate=recovery_rate,
        recovery_time_p95_ms=recovery_p95,
        duplicate_event_rate=duplicate_rate,
    )


def compare_paired_runs(
    baseline: list[BenchmarkSummary],
    candidate: list[BenchmarkSummary],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20260901,
) -> PairedRunComparison:
    if len(baseline) < 3 or len(candidate) < 3:
        raise ValueError("run comparison requires at least three trials per strategy")
    baseline_by_trial = {item.trial_id: item for item in baseline}
    candidate_by_trial = {item.trial_id: item for item in candidate}
    if len(baseline_by_trial) != len(baseline) or len(candidate_by_trial) != len(candidate):
        raise ValueError("trial IDs must be unique within each strategy")
    if baseline_by_trial.keys() != candidate_by_trial.keys():
        raise ValueError("baseline and candidate must contain the same trial IDs")
    identities = {
        (item.workload_id, item.model_revision, item.harnesses) for item in baseline + candidate
    }
    if len(identities) != 1:
        raise ValueError("paired trials must share workload, model revision, and Harnesses")
    if (
        len({item.strategy for item in baseline}) != 1
        or len({item.strategy for item in candidate}) != 1
    ):
        raise ValueError("each side must contain exactly one strategy")

    pairs = [(baseline_by_trial[index], candidate_by_trial[index]) for index in baseline_by_trial]
    latency = [
        _relative_reduction(left.task_latency_p95_ms, right.task_latency_p95_ms)
        for left, right in pairs
    ]
    hit_lift = [
        (right.prefix_cache_token_hit_rate - left.prefix_cache_token_hit_rate) * 100
        for left, right in pairs
    ]
    prefill = [
        _relative_reduction(
            float(left.resume_repeated_prefill_tokens),
            float(right.resume_repeated_prefill_tokens),
        )
        for left, right in pairs
    ]
    throughput = [
        _relative_improvement(
            left.tasks_per_second_per_gpu,
            right.tasks_per_second_per_gpu,
        )
        for left, right in pairs
    ]
    utilization = [
        right.mean_gpu_utilization_percent - left.mean_gpu_utilization_percent
        for left, right in pairs
    ]
    recovery_lift = _optional_pair_values(
        pairs,
        lambda item: item.fault_recovery_success_rate,
        lambda left, right: (right - left) * 100,
    )
    recovery_time = _optional_pair_values(
        pairs,
        lambda item: item.recovery_time_p95_ms,
        _relative_reduction,
    )
    rng = random.Random(seed)
    return PairedRunComparison(
        trial_count=len(pairs),
        task_latency_p95_reduction_percent=_estimate_ci(latency, bootstrap_repetitions, rng),
        prefix_cache_hit_lift_points=_estimate_ci(hit_lift, bootstrap_repetitions, rng),
        resume_repeated_prefill_reduction_percent=_estimate_ci(prefill, bootstrap_repetitions, rng),
        tasks_per_second_per_gpu_improvement_percent=_estimate_ci(
            throughput, bootstrap_repetitions, rng
        ),
        gpu_utilization_lift_points=_estimate_ci(utilization, bootstrap_repetitions, rng),
        fault_recovery_lift_points=(
            _estimate_ci(recovery_lift, bootstrap_repetitions, rng)
            if recovery_lift is not None
            else None
        ),
        recovery_time_p95_reduction_percent=(
            _estimate_ci(recovery_time, bootstrap_repetitions, rng)
            if recovery_time is not None
            else None
        ),
    )


def compare_paired_task_latency(
    baseline: list[RequestMeasurement],
    candidate: list[RequestMeasurement],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20260901,
) -> PairedLatencyComparison:
    _require_comparable(baseline, candidate)
    baseline_latency = _successful_task_latencies(baseline)
    candidate_latency = _successful_task_latencies(candidate)
    task_ids = sorted(set(baseline_latency) & set(candidate_latency))
    if len(task_ids) < 2:
        raise ValueError("paired comparison requires at least two shared completed tasks")
    reductions = [
        (baseline_latency[task] - candidate_latency[task]) / baseline_latency[task] * 100
        for task in task_ids
    ]
    rng = random.Random(seed)
    bootstrapped = [
        sum(rng.choice(reductions) for _ in reductions) / len(reductions)
        for _ in range(bootstrap_repetitions)
    ]
    baseline_p95 = _percentile(list(baseline_latency.values()), 0.95)
    candidate_p95 = _percentile(list(candidate_latency.values()), 0.95)
    return PairedLatencyComparison(
        task_count=len(task_ids),
        p95_reduction_percent=(baseline_p95 - candidate_p95) / baseline_p95 * 100,
        mean_reduction_percent=sum(reductions) / len(reductions),
        mean_reduction_ci95=(
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ),
    )


def _require_comparable(
    baseline: list[RequestMeasurement], candidate: list[RequestMeasurement]
) -> None:
    if not baseline or not candidate:
        raise ValueError("both runs require measurements")
    baseline_identity = {(item.workload_id, item.model_revision) for item in baseline}
    candidate_identity = {(item.workload_id, item.model_revision) for item in candidate}
    if len(baseline_identity) != 1 or baseline_identity != candidate_identity:
        raise ValueError("paired runs must share workload and model revision")
    baseline_shape = sorted(
        (item.task_id, item.harness_id, item.phase, item.input_tokens, item.output_tokens)
        for item in baseline
    )
    candidate_shape = sorted(
        (item.task_id, item.harness_id, item.phase, item.input_tokens, item.output_tokens)
        for item in candidate
    )
    if baseline_shape != candidate_shape:
        raise ValueError("paired runs must contain the same task/request shape")


def _successful_task_latencies(
    requests: list[RequestMeasurement],
) -> dict[str, float]:
    by_task: dict[str, list[RequestMeasurement]] = defaultdict(list)
    for item in requests:
        by_task[item.task_id].append(item)
    return {
        task_id: max(item.finished_ms for item in task) - min(item.started_ms for item in task)
        for task_id, task in by_task.items()
        if all(item.success for item in task)
    }


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("relative reduction requires a positive baseline")
    return (baseline - candidate) / baseline * 100


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("relative improvement requires a positive baseline")
    return (candidate - baseline) / baseline * 100


def _optional_pair_values(
    pairs: list[tuple[BenchmarkSummary, BenchmarkSummary]],
    getter: Callable[[BenchmarkSummary], float | None],
    transform: Callable[[float, float], float],
) -> list[float] | None:
    values: list[float] = []
    for left, right in pairs:
        left_value = getter(left)
        right_value = getter(right)
        if left_value is None or right_value is None:
            return None
        values.append(transform(left_value, right_value))
    return values


def _estimate_ci(values: list[float], repetitions: int, rng: random.Random) -> EstimateWithCI:
    if not values or repetitions < 100:
        raise ValueError("confidence interval requires values and at least 100 resamples")
    bootstrapped = [
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions)
    ]
    return EstimateWithCI(
        estimate=sum(values) / len(values),
        ci95=(_percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
