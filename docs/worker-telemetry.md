# Worker telemetry and calibration boundary

The collector uses the pinned engine's Prometheus endpoint and host GPU memory
observation, then emits generation-fenced gRPC heartbeats. It supports one model
and engine `0` per worker. It selects exact metric names and labels; unrelated
multi-dimensional series are ignored, while ambiguous selected series fail.

Run `python -m freechat_worker.telemetry --help` on the GPU host. Register the
worker first and supply the same capabilities file and generation. The engine
instance identifier must change on engine restart, along with worker generation.
The collector stops on scrape failure or reset; existing telemetry ages out.
Scheduler rejects samples older than 30 seconds or more than five seconds in
the future. Heartbeats cannot replace newer observations or change engine
instance within an existing generation. Internal transport authentication is
still a separate deployment gate; these checks are consistency checks.

## Available observations

- Running and waiting request counts, plus physical free VRAM from nvidia-smi.
- Engine KV usage ratio. The pinned BlockPool computes this as one minus free
  blocks divided by total blocks excluding its null block. It does not expose
  the identity/value of reusable cached prefixes. Consequently the collector
  leaves KV capacity/free-byte and prefix inventory fields unknown.
- Store/load bytes divided by accumulated operation duration, calculated from
  consecutive scrapes in the same engine instance. This is transfer service
  throughput, not end-to-end request speed or wall-clock PCIe saturation.
- Windows without transfer observations emit no store estimate. A first scrape
  establishes the baseline; missing counters are not silently treated as zero.

## Remaining calibration work

Prefill/decode throughput, KV layout-derived capacity and reusable block lineage
are not yet calibrated by this collector. Existing numeric contract defaults
are not measurements. Scheduler disables predictive offload for this source
with `prefill_calibration_required`; route cost estimates still use contract
defaults and must not be used in performance claims or heterogeneous-placement
acceptance. The next change must replace those defaults with calibrated profiles
and an explicit unavailable-cost fallback.

## Live validation

`benchmarks/telemetry_probe.py` connects real worker HTTP metrics to an isolated
in-memory scheduler over gRPC. It sends a direct real-model request with explicit
native offload enabled to produce a transfer measurement; this is not a policy
performance experiment. Run it with a registered-model capabilities file and
a unique evidence output directory. At least one prior transfer is required to
initialize the native engine's lazily emitted counters.

On 2026-09-08, A5000 validation `evidence/telemetry/20260908-run03/` accepted both
heartbeats, observed a store-window service rate of approximately 11.112 GB/s,
and returned `prefill_calibration_required` through the real gRPC route client.
This one window provides no confidence interval and does not establish expected
bandwidth for other shapes. The raw before/after metrics and hashes are retained.

Run01 retained a failed parser probe (unrelated multi-label metrics); run02
retained the first-counter initialization case. Run03 is the successful probe.
No run is a Harness or end-to-end performance acceptance result.
