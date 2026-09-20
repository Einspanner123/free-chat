# Native KV transfer calibration evidence

The executable probe is `python -m benchmarks.calibrate_transfer`. It uses an
already-running, dedicated single-model worker. The pressure recipe is specific to
the A5000 probe container's 64 MiB KV pool, not general topology discovery.

Each scope runs a warm-up cycle followed by three measured cycles:

1. Submit a new salted target with native prompt offload enabled.
2. Submit eight independently salted pressure requests with offload disabled.
3. Resume the exact target with the same salt and offload disabled.
4. Compare isolated request windows for Store/Load byte, operation and service-time
   deltas. Require external prefix hit tokens in every accepted Load window.

The observer rejects missing counters, counter regression/disappearance, non-idle
boundaries, preemption, multiple completed requests, zero transfer amounts and
non-integral accounting. This is a controlled experiment: boundary checks alone do
not establish exclusive access against arbitrary concurrent clients.

## Evidence

`evidence/transfer-calibration/20260908-run01/` records two successful scopes, each
with three Store and three Load observations after warm-up, on real A5000 inference:

| Scope | Bytes per Store and Load | External hit tokens per resume |
| --- | ---: | ---: |
| 32 repetitions | 2,949,120 | 240 |
| 128 repetitions | 11,206,656 | 912 |

These values match 12,288 KV bytes/token for the configured Dense/GQA model. They
are complete-block transfers, not all prompt tokens. Service-time ratios appear in
the raw observations only as calibration data, not task throughput or speedup.

The successful follow-up `evidence/transfer-calibration/20260908-run02/` also archives
target/resume responses and its source hash, and
requires equal target/resume prompt counts plus Load bytes equal to external hit
tokens times the configured KV layout size. Each run has a separate output directory;
raw earlier evidence is retained, never overwritten.

## What this does not establish

- No real Harness or baseline/candidate performance comparison was run here.
- The two exact byte sizes do not justify interpolation, concurrent-copy cost or
  a universal bandwidth value. Three samples do not establish a stable tail estimate.
- The aggregate external-hit counter is not a per-block residency catalog. The
  current KV controller policy and worker callback buffer do not supply a complete,
  reconciled native CPU/GPU residency inventory. Dropped events must invalidate
  certainty before any catalog is used for routing.
- Image identity and engine instance are probe arguments; production attestation
  and full model/tokenizer artifact identity remain separate gates.
- These measurements are not automatically merged into Scheduler profiles. The
  current request-size estimate uses total prompt tokens, while the native connector
  transfers complete blocks. Calibrate the actual transfer plan and its direction,
  size and generation before authorizing predictive offload from these samples.

Next: implement the explicit block-aligned transfer-plan contract, connect native
completion/residency signals with uncertainty handling, then run paired real Harness
trials. The A-before-B decision and plan-retention requirement remain unchanged.
