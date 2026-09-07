# Predictive KV offload at request admission

Status: RESOLVED

Date: 2026-09-08

## Decision

The scheduler decides before dispatch whether the current request may populate vLLM's native CPU KV offload tier. For workers that advertise native offload support, the trusted Gateway writes the internal `kv_transfer_params.max_offload_tokens` field with either the scheduler's token cap or zero. It omits the vLLM-only field for other backends. Clients cannot supply this field.

The decision uses authenticated lifecycle hints, hint confidence, expected prefix reuse, estimated KV bytes, reported worker prefill throughput, separately reported KV store/load bandwidth, expected wait duration, and KV-pool pressure. It enables offload only when expected avoided recompute exceeds expected transfer time. Missing store bandwidth does not fall back to an assumed symmetric rate, and missing KV-pool telemetry does not fall back to whole-device VRAM pressure. Round-robin, least-load, prefix-affinity, and cost-aware baselines do not enable this policy.

vLLM's `OffloadingConnector` remains responsible for GPU-to-pinned-CPU storage and CPU-to-GPU restoration. FreeChat does not duplicate tensor movement, block storage, or PagedAttention.

## Deferred alternative

A new exact post-tool-wait API in the vLLM fork is deferred, not rejected. It will be reconsidered only after real Harness traces establish the predictive policy's waste and missed-opportunity baseline. The evidence set must include stored bytes never restored, restored bytes, repeated prefill tokens, TTFT P95, task latency P95, and scheduler prediction calibration, split by Harness and workload class.

Thresholds for changing this decision will be frozen after that baseline is collected. Until then, no percentage improvement or end-to-end performance claim is verified.

## Traceability

This record resolves the explicit architecture conflict by owner direction: implement predictive pre-offload first, inspect evidence, and decide separately whether the exact post-tool-wait fork interface is justified.
