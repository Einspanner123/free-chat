# Offload implementation route after real Harness boundary trials

Date: 2026-09-08

Implementation route: **A — conditional admission-time native pre-offload**.

Performance acceptance: **NOT ACCEPTED**. A-versus-B superiority: **NOT ESTABLISHED**.

## Decision and its scope

Continue implementing A. Do not build B's new exact post-tool-wait engine API as the
current project path. This is an engineering scope decision under the owner's request
to determine which route to execute, not a statement that a head-to-head A/B benchmark
has been completed. The previously frozen project-wide performance acceptance gates
remain in force. Unknown prediction, cost, layout or residency must leave offload
disabled; the always-permitted envelope used below is not the deployable policy.

Evidence establishes that vLLM's existing admission-time native transfer path can
preserve useful prefixes in real Harness tool workflows. No measured result establishes
that introducing a second engine API is necessary to obtain that mechanism. The main
unresolved problem is selecting when copies are worth making, not inability to copy.
Even an exact Tool Wait signal does not establish future eviction or eventual reuse:
both pressure and no-pressure tasks below actually executed a tool and resumed.

B would also need to handle the interval between model completion and the later
Harness callback. Inspection of the running image's native offloading scheduler shows
`request_finished` runs before block free, returns `False, None`, and documents state
cleanup after pending store jobs are built. This is not a pre-existing arbitrary
post-completion retention guarantee. Extending it requires explicit ownership,
generation checks, retention bounds and expiration semantics. Those costs have not
been measured; do not call B slower or faster without implementing and testing it.

## Experimental evidence

Protocol: `20260908-offload-boundary-protocol.md`.
Raw evidence: `evidence/harness-boundary/20260908-paired01/`.
Recomputation: `python -m benchmarks.analyze_offload_boundary <run-directory>`
(the script refuses to overwrite its saved analysis).

80 measured pairs / 160 tasks plus 8 warm-up tasks. Two real Harness runtimes,
one forced repository-heading workflow, one A5000, Qwen2.5-0.5B, a 64 MiB HBM KV
pool. Alternating arm order, per-task cache salts, raw request/response/metrics saved.
The control disables native offload; the treatment permits it on both model calls.
Neither arm goes through FreeChat's Scheduler or Gateway. There is no B arm.

The running container retained start time `2026-09-08T02:15:13.502949333Z` through
the experiment, with local image identity
`sha256:5527760b5854807b7c9d6c0544ea8f619a76c86b220a3f345b47823c6aa19288`.
This is not a registry digest. Runtime integration was checked against the installed
SDK code; the [official Agents documentation](https://developers.openai.com/api/docs/guides/agents)
was consulted for SDK orchestration context, not as evidence of benchmark performance.

| Pressure workload | Resume new-KV tokens, control → pre-offload | Server resume TTFT P95, control → pre-offload | Paired 95% interval for TTFT difference |
| --- | --- | --- | --- |
| Agents SDK | 1129 → 57 | 21.583 → 13.354 ms | [-8.816, -7.628] ms |
| LangGraph | 1110 → 54 | 20.825 → 13.264 ms | [-8.703, -7.011] ms |

In each pressure group, 20 real resume requests loaded external KV. Aggregated loaded
bytes were 263,454,720 (Agents) and 259,522,560 (LangGraph). Each treatment group also
stored 11,796,480 bytes that were not loaded again by terminal task completion.

Without pressure, neither treatment group loaded any CPU KV. They nevertheless
stored 275,251,200 and 271,319,040 bytes respectively. Recomputed resume tokens were
unchanged at 41 and 38. Thus blanket pre-offload is rejected: a real tool wait alone
is insufficient authorization for a useful CPU copy.

## Validation assessment

**Share with caveats for implementation prioritization; needs revision for performance
acceptance or resume claims.**

- All four instrumentation-adjusted task P95 difference intervals cross zero.
  Pressure: Agents [-314.914, 266.747] ms; LangGraph [-230.668, 231.131] ms.
  No pressure: Agents [-8.206, 18.759] ms; LangGraph [-0.528, 67.080] ms.
- Do not hide the adverse raw result: LangGraph/no-pressure raw task P95 increased
  by 171.185 ms, interval [1.658, 271.608] ms. Raw timing includes probe collection;
  adjusted timing subtracts explicitly measured instrumentation. This discrepancy
  blocks a clean task-latency claim and warrants a less intrusive measurement path.
- Exact character-for-character answer correctness was 0/160. All 160 contained
  the correct heading but added explanatory text. Semantic presence is a separate
  diagnostic, not a replacement for the failed strict acceptance criterion.
- Each group contains only 20 pairs. Intervals use 10000 paired bootstrap draws,
  seed 42, interpolated P95, and no correction for multiple comparisons. The exact
  repeated Token reductions reflect deterministic prompts, not generalization.
- The raw-artifact hashes were recomputed by the analysis. All wrong answers remain
  in the population. No charts are used; displayed values are rounded from analysis.json.
- A6000/A4000, larger models, general Tool Agents, OpenCode and OpenHands are not
  covered. A calibrated predictor, live residency catalog, client streaming TTFT,
  throughput and utilization acceptance are absent. Another A4000 workload shared
  the workstation host; its GPU was not used by this experiment.

## Concrete next implementation boundaries

1. Replace the adapter's hard-coded 0.9 reuse assumption with explicitly sourced,
   calibrated forecasts; distinguish future wait prediction from a past tool's duration.
2. Bind transfer cost to native complete-block plans and model/engine generation.
   Do not treat whole-prompt byte estimates as exact scheduled Store/Load bytes.
3. Feed native residency/completion events into a reconciled catalog; dropped or
   missing events must produce uncertainty, not invented residency.
4. Retain no-offload as the fallback, and charge both unused copies and terminal
   copies against the policy. Do not deploy the always-on experiment arm.
5. Run the actual policy through Gateway/Scheduler and all four Harnesses, with coding
   and general-tool tasks and strict task success, before any performance claim.

Reopen B only with evidence that calibrated A misses valuable prefixes specifically
because the decision precedes Tool Wait, and that improving prediction/retention within
the native path is insufficient. Any reversal of the selected route requires a new
traceable owner decision. The implementation plan remains retained.
