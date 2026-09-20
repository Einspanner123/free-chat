# Offload route decision evidence protocol

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
