# Scoped inference service calibration

The Scheduler consumes measured service profiles instead of interpreting telemetry's
legacy default rates as measured inference costs. Profiles bind worker generation,
engine instance, model capability, observation/expiry timestamps, image identity and
an observation artifact hash. Matching requires the measured input/output scope and
concurrency scope; stale, missing or mismatched profiles are unavailable estimates.
If any eligible candidate lacks a usable estimate, cost-aware and lifecycle-aware
routing explicitly fall back to least-load. The decision carries requested strategy,
effective strategy, fallback reason and per-candidate estimate availability.

Numeric zero fields in an unavailable candidate are serialization placeholders, not
zero-latency predictions. Consumers must inspect `estimate_available`. Queue, network
and eviction terms are not comprehensively calibrated by this work.

## Real execution, 2026-09-08

`evidence/calibration/20260908-run01/` contains raw before/after Prometheus snapshots,
response usage, observations, scoped profiles, gRPC routing decisions and a SHA-256
manifest. The run used workstation A5000 and Qwen2.5-0.5B-Instruct, with three measured
serial requests per scope after warm-up. Both scopes generate exactly 16 tokens:

| Input tokens | Prefill tokens / engine service second | Decode tokens / engine service second |
| --- | ---: | ---: |
| 254 | 31505.08 | 420.56 |
| 926 | 54923.61 | 413.76 |

These are pooled calibration rates, not task throughput, speedup, TTFT or an SLO.
Decode counts output tokens minus the first token against the engine decode interval.
The collector requires one completed request in every histogram, idle boundaries,
unchanged preemption count, positive deltas, response/metric token agreement, and
computed KV tokens equal to prompt tokens. Unique cache salts prevent prefix reuse.
The actual image's metric definition excludes cached tokens from computed KV tokens.

The probe validated profiles through real gRPC heartbeats and Gateway routing-client
serialization against an isolated in-memory Scheduler. In-scope requests received a
calibration ID. An out-of-scope request received least-load plus
`candidate_cost_unavailable`. This does not validate the deployed persistent control
plane. Profiles expire after one hour and are not portable across engine restarts.

## Remaining gates

- The image identity is a local image SHA, not a registry digest. The model/tokenizer
  revision labels are not an independently verified complete artifact manifest.
- Two exact input lengths with three samples each do not establish interpolation,
  confidence intervals, concurrent batching or cached-prefix service costs.
- Store/Load size-scoped calibration is absent. Predictive offload remains disabled
  with `transfer_calibration_required`, even when inference calibration matches.
- KV residency and eviction probability remain separate work; free VRAM and cache
  usage do not establish prefix residency or future reuse.
- Four-Harness paired task-level comparisons remain uncompleted. No resume benefit
  percentage may be derived from these service rates.

Continue with Store/Load calibration and residency evidence, then evaluate predictive
offload using paired real Harness trials. The post-tool-wait fork interface remains
deferred under the recorded A-before-B decision. Keep the implementation plan.
