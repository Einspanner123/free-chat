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
`agent-aware-inference-infra-plan.md`.

`ResolvedBenchmarkConfig` enforces the comparison boundary in code. Paired trials
must match workload, model and tokenizer revisions, prompt-template hash, FreeChat
and vLLM commits, Worker image digest, topology hash, Harness versions, concurrency,
warm-up, fault schedule and collector version. Only strategy and run identity may
differ. A mismatch fails before a confidence interval is calculated.

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

## 原生预卸载的测量边界

当前采用条件式 admission-time native pre-offload，不宣称优于其他路线。
对照必须区分 admission eligibility、实际 offload/load、prefix hit 和 task 成功。
仅适配器输入、SSE transport 完成或 block free 都不代表推理资源已经释放。
统计协议见下；既有试验结论只在 Claims Ledger 与原始 evidence 中保留。

## Offload route decision evidence protocol

Frozen before the measured run `evidence/harness-boundary/20260908-paired01/`.

Question: does request-admission pre-offload have an observable opportunity on this
worker, and what waste does it introduce when a tool returns without HBM eviction?
This is an architectural boundary test, not the final A predictor or a B implementation.

Population: OpenAI Agents SDK and LangGraph, one real repository-heading tool and
two real Qwen2.5-0.5B model calls per task, workstation A5000, 64 MiB HBM KV pool.
Forced tool selection and a forced final answer bound the workflow equally in both
arms. Context padding is synthetic. Native LRU with offload disabled is compared
with an admission pre-offload envelope (all model calls permitted to offload).
The envelope does not use a learned predictor, Scheduler, Gateway or lifecycle hints.

For each Harness and pressure/no-pressure scenario: one unmeasured warm-up pair,
then 20 measured pairs. Alternate treatment order; use a fresh salt per task and
reuse it only inside that task. No filtering of wrong answers or slow samples.
Before/after metric windows must contain exactly one completed model call and no
counter regression or preemption. Preserve every failure and raw sample.

Metrics: whole-task raw elapsed time and elapsed time minus explicitly timed probe
instrumentation; model request elapsed time; server TTFT on the resume call; resume
new-KV tokens; stored and loaded bytes; terminal-horizon stored-minus-loaded bytes
(only meaningful for this exactly-one-resume workflow); local/external cache hits.
Server TTFT is not client-observed streaming TTFT. Report per-Harness/scenario, never
pool unlike task distributions. Paired bootstrap, fixed seed 42, 10000 draws, linear
P95 interpolation; report difference intervals, not only point improvements.

The smoke run is excluded from estimation, but retained. It failed exact-output
formatting: the model returned the right heading with explanatory text. Preserve
`correct` as strict character-for-character grading. Add a separately named semantic
heading-presence diagnostic; never substitute it for strict task acceptance.

Decision boundaries:

- Reduced recomputation with correlated external Load establishes opportunity for A,
  not a verified performance win.
- An interval crossing zero cannot support a task-latency improvement claim.
- CPU copies with no Load expose prediction waste, but do not prove B would win:
  even a known Tool Wait does not guarantee HBM eviction or eventual reuse.
- B means a new exact post-tool-wait API. Its implementation, resource retention
  costs and head-to-head benchmark are absent; no number is assigned to it.
- A final project-wide A/B acceptance requires the previously agreed four Harnesses,
  coding and general-tool workloads, real predictor calibration and owner confirmation
  for any reversal. This narrow experiment cannot silently lower those requirements.

Engineering recommendation may prefer continuing A when native transfers demonstrably
avoid recomputation and no evidence establishes that B is necessary. Such a recommendation
must remain distinct from a final accepted A/B decision or production performance claim.
