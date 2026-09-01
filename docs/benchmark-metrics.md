# Benchmark metrics and evidence contract

Performance claims are admitted only when the candidate and baseline use the same
workload revision, model revision, tokenizer, prompt/template, output constraint,
hardware allocation, concurrency schedule, warm-up policy and fault schedule. Every
run records its resolved configuration and environment digest with the raw samples.

## Measurement unit

An Agent task is the unit used for end-to-end latency and throughput. A task may
contain multiple model calls, tool waits and resumptions. A model request is not
counted as a task. Failed or cancelled tasks are reported separately and are never
silently removed from a success-rate denominator.

## Frozen definitions

| Metric | Definition | Required raw fields |
|---|---|---|
| Agent task latency P95 | P95 of `max(request.finished_ms) - min(request.started_ms)` over completed tasks. | task/request identity, timestamps, success state |
| Prefix-cache token hit rate | `sum(cached_tokens) / sum(input_tokens)` across all measured requests. This is token weighted, not request weighted. | input and cached tokens per request |
| Resume repeated-prefill tokens | `sum(input_tokens - cached_tokens)` for requests whose phase is `resume`. | lifecycle phase, input and cached tokens |
| Tasks/s/GPU | completed Agent tasks divided by measured wall time and the number of allocated GPUs. | task completion, wall-time boundaries, GPU identity |
| Mean GPU utilization | Time-weighted mean of per-GPU utilization samples. Sampling cadence and collector version are fixed within a comparison. | utilization, sample interval, GPU identity |
| Fault recovery success rate | recovered injected faults divided by all injected faults. | fault identity, injection and recovery state |
| Recovery time P95 | P95 of `recovered_ms - injected_ms` over recovered faults, with unrecovered count reported beside it. | injection and recovery timestamps |
| Duplicate-event rate | duplicate deliveries divided by unique plus duplicate deliveries observed during fault trials. | unique and duplicate delivery counts |

The resume prefill claim is reported both as an absolute token count and as relative
reduction against the paired baseline. Cache hit rate is reported as both absolute
rates and percentage-point lift; using relative percent alone is prohibited.

## Baselines

- Routing: direct vLLM, round-robin and least-load are mandatory. Prefix-affinity,
  cost-aware and lifecycle-aware strategies are compared only after each preceding
  baseline is functional.
- Cache: native vLLM prefix caching and native eviction are the baseline. A policy
  that merely emits lifecycle events is not a lifecycle-aware cache candidate.
- Faults: no-fault steady state and the same deterministic fault schedule are both
  retained. Recovery claims require a real serving process and in-flight tasks.
- Kernel: PyTorch reference and the unfused implementation run in the same process,
  dtype, shapes and device power state as the candidate.

## Statistical gate

Runs use paired task fixtures and a recorded seed. The primary latency comparison
uses a seeded bootstrap over per-task relative reductions. A positive claim requires
the 95% confidence interval to exclude zero, at least three independent run seeds,
and a non-target workload guardrail. P95, failure count and sample size are always
reported together. Kernel microbenchmarks follow the stricter candidate gate in
`docs/kernel-candidates.md`.

## Artifact gate

Raw request records, GPU samples and fault records are stored as Parquet. Each
artifact is SHA-256 identified and accompanied by the resolved run configuration,
Git SHAs, image digest, model/tokenizer revision, GPU topology, driver, CUDA,
PyTorch, Triton, vLLM and Harness versions. Claims Ledger evidence must point to
these immutable artifacts. Summaries without the raw records are not evidence.

## Resume publication gate

The resume placeholders remain blank until all relevant Claims Ledger rows are
`VERIFIED`. Isolated kernel speedup is listed only as a kernel result and cannot be
used as proof of Agent-task latency, cache reuse, throughput, utilization or fault
recovery.
