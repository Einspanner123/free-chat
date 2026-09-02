import pytest
from freechat_trace_replay.artifacts import ParquetTraceArchive
from freechat_trace_replay.benchmark_metrics import (
    BenchmarkSummary,
    FaultMeasurement,
    GpuUtilizationSample,
    RequestMeasurement,
    ResolvedBenchmarkConfig,
    archive_benchmark_run,
    compare_paired_evidence,
    compare_paired_runs,
    compare_paired_task_latency,
    summarize_run,
)


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}

    def put(
        self,
        bucket: str,
        key: str,
        payload: bytes,
        metadata: dict[str, str],
    ) -> None:
        self.objects[(bucket, key)] = payload, metadata


def request(
    task: str,
    *,
    strategy: str = "lifecycle-aware",
    started: float = 0,
    finished: float = 100,
    cached: int = 80,
) -> RequestMeasurement:
    return RequestMeasurement(
        run_id=f"run-{strategy}",
        trial_id=1,
        workload_id="coding-agent-fixture-1",
        strategy=strategy,
        model_revision="model-sha",
        task_id=task,
        harness_id="opencode",
        phase="resume",
        started_ms=started,
        finished_ms=finished,
        ttft_ms=20,
        input_tokens=100,
        cached_tokens=cached,
        output_tokens=16,
        success=True,
    )


def test_summary_uses_task_grain_token_denominators_and_fault_rto() -> None:
    requests = [
        request("a", started=0, finished=100, cached=80),
        request("b", started=20, finished=220, cached=60),
    ]
    summary = summarize_run(
        requests,
        [
            GpuUtilizationSample(
                run_id="run-lifecycle-aware",
                gpu_id="0",
                observed_ms=50,
                sample_interval_ms=50,
                utilization_percent=40,
            ),
            GpuUtilizationSample(
                run_id="run-lifecycle-aware",
                gpu_id="0",
                observed_ms=150,
                sample_interval_ms=150,
                utilization_percent=60,
            ),
        ],
        [
            FaultMeasurement(
                run_id="run-lifecycle-aware",
                task_id="a",
                fault_id="restart",
                injected_ms=50,
                recovered_ms=90,
            )
        ],
    )
    assert summary.completed_tasks == 2
    assert summary.prefix_cache_token_hit_rate == pytest.approx(0.7)
    assert summary.resume_repeated_prefill_tokens == 60
    assert summary.mean_gpu_utilization_percent == 55
    assert summary.fault_recovery_success_rate == 1
    assert summary.recovery_time_p95_ms == 40
    assert summary.duplicate_event_rate == 0


def test_summary_uses_delivered_events_as_duplicate_rate_denominator() -> None:
    summary = summarize_run(
        [request("a")],
        [
            GpuUtilizationSample(
                run_id="run-lifecycle-aware",
                gpu_id="0",
                observed_ms=50,
                sample_interval_ms=50,
                utilization_percent=40,
            )
        ],
        [
            FaultMeasurement(
                run_id="run-lifecycle-aware",
                task_id="a",
                fault_id="redelivery",
                injected_ms=50,
                recovered_ms=90,
                unique_events=9,
                duplicate_events=1,
            )
        ],
    )
    assert summary.duplicate_event_rate == pytest.approx(0.1)


def test_paired_comparison_rejects_workload_shape_drift() -> None:
    baseline = [request("a", strategy="round-robin"), request("b", strategy="round-robin")]
    candidate = [request("a"), request("different")]
    with pytest.raises(ValueError, match="same task/request shape"):
        compare_paired_task_latency(baseline, candidate)


def test_paired_comparison_reports_seeded_confidence_interval() -> None:
    baseline = [
        request("a", strategy="round-robin", finished=200),
        request("b", strategy="round-robin", finished=240),
        request("c", strategy="round-robin", finished=300),
    ]
    candidate = [
        request("a", finished=150),
        request("b", finished=180),
        request("c", finished=210),
    ]
    result = compare_paired_task_latency(baseline, candidate, bootstrap_repetitions=1_000)
    assert result.task_count == 3
    assert result.p95_reduction_percent > 0
    assert result.mean_reduction_ci95[0] > 0


def test_archive_contains_raw_inputs_summary_and_resolved_config() -> None:
    store = MemoryObjectStore()
    artifacts = archive_benchmark_run(
        ParquetTraceArchive(store, "evidence"),
        [request("a")],
        [
            GpuUtilizationSample(
                run_id="run-lifecycle-aware",
                gpu_id="0",
                observed_ms=50,
                sample_interval_ms=50,
                utilization_percent=40,
            )
        ],
        [],
        {
            "git_sha": "freechat-sha",
            "vllm_sha": "fork-sha",
            "image_digest": "sha256:image",
            "model_revision": "model-sha",
        },
    )
    assert artifacts.faults is None
    assert artifacts.requests.sha256
    assert artifacts.gpu_samples.sha256
    assert artifacts.summary.sha256
    assert artifacts.resolved_config.sha256
    assert len(store.objects) == 4


def summary(trial: int, strategy: str, factor: float = 1.0) -> BenchmarkSummary:
    return BenchmarkSummary(
        run_id=f"{strategy}-{trial}",
        trial_id=trial,
        workload_id="coding-agent-fixture-1",
        strategy=strategy,
        model_revision="model-sha",
        harnesses=("opencode",),
        completed_tasks=20,
        failed_tasks=0,
        task_latency_p95_ms=1_000 * factor,
        prefix_cache_token_hit_rate=0.8 if factor < 1 else 0.5,
        resume_repeated_prefill_tokens=int(1_000 * factor),
        tasks_per_second_per_gpu=2 / factor,
        mean_gpu_utilization_percent=70 if factor < 1 else 60,
        fault_recovery_success_rate=1.0 if factor < 1 else 0.8,
        recovery_time_p95_ms=500 * factor,
        duplicate_event_rate=0.0,
    )


def resolved_config(trial: int, strategy: str) -> ResolvedBenchmarkConfig:
    character = str(trial + 1)
    return ResolvedBenchmarkConfig(
        run_id=f"{strategy}-{trial}",
        trial_id=trial,
        seed=20260910 + trial,
        strategy=strategy,
        workload_revision="coding-agent-fixture-1",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        prompt_template_sha256=character * 64,
        freechat_git_sha="a" * 40,
        vllm_git_sha="b" * 40,
        worker_image_digest=f"sha256:{'c' * 64}",
        topology_sha256="d" * 64,
        harness_versions=("opencode=0.1.0",),
        concurrency_schedule=(1, 4, 8),
        warmup_tasks=4,
        fault_schedule_sha256="e" * 64,
        collector_version="dcgm-exporter-4.4.1",
    )


def test_run_comparison_requires_three_trials_and_reports_metric_intervals() -> None:
    baseline = [summary(trial, "round-robin") for trial in range(3)]
    candidate = [summary(trial, "lifecycle-aware", 0.8) for trial in range(3)]
    result = compare_paired_runs(
        baseline,
        candidate,
        bootstrap_repetitions=1_000,
    )
    assert result.trial_count == 3
    assert result.task_latency_p95_reduction_percent.ci95[0] > 0
    assert result.prefix_cache_hit_lift_points.estimate == pytest.approx(30)
    assert result.resume_repeated_prefill_reduction_percent.estimate == pytest.approx(20)
    assert result.tasks_per_second_per_gpu_improvement_percent.estimate == pytest.approx(25)
    assert result.gpu_utilization_lift_points.estimate == pytest.approx(10)
    assert result.fault_recovery_lift_points is not None
    assert result.recovery_time_p95_reduction_percent is not None

    with pytest.raises(ValueError, match="at least three trials"):
        compare_paired_runs(baseline[:2], candidate[:2])


def test_evidence_comparison_rejects_image_or_schedule_drift() -> None:
    baseline = [summary(trial, "round-robin") for trial in range(3)]
    candidate = [summary(trial, "lifecycle-aware", 0.8) for trial in range(3)]
    baseline_configs = [resolved_config(trial, "round-robin") for trial in range(3)]
    candidate_configs = [resolved_config(trial, "lifecycle-aware") for trial in range(3)]

    result = compare_paired_evidence(
        baseline,
        candidate,
        baseline_configs,
        candidate_configs,
        bootstrap_repetitions=1_000,
    )
    assert result.task_latency_p95_reduction_percent.ci95[0] > 0

    drifted = candidate_configs.copy()
    drifted[1] = drifted[1].model_copy(
        update={"worker_image_digest": f"sha256:{'f' * 64}"}
    )
    with pytest.raises(ValueError, match="worker_image_digest"):
        compare_paired_evidence(
            baseline,
            candidate,
            baseline_configs,
            drifted,
            bootstrap_repetitions=1_000,
        )


def test_evidence_comparison_requires_independent_seeds() -> None:
    baseline = [summary(trial, "round-robin") for trial in range(3)]
    candidate = [summary(trial, "lifecycle-aware", 0.8) for trial in range(3)]
    baseline_configs = [
        resolved_config(trial, "round-robin").model_copy(update={"seed": 7})
        for trial in range(3)
    ]
    candidate_configs = [
        resolved_config(trial, "lifecycle-aware").model_copy(update={"seed": 7})
        for trial in range(3)
    ]

    with pytest.raises(ValueError, match="three independent seeds"):
        compare_paired_evidence(
            baseline,
            candidate,
            baseline_configs,
            candidate_configs,
            bootstrap_repetitions=1_000,
        )
