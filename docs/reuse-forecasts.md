# Explicit future-reuse forecasts

The four Harness adapters no longer assign 0.9 to every growing-history request.
Lifecycle identity is an observation, not a calibrated probability of future reuse.
Without an applicable forecast the adapter emits probability zero and metadata
`reuse_forecast_status=unavailable`. Zero here is the conservative control value,
not an empirical prediction that reuse is impossible; exclude unavailable values
from calibration scoring and report their coverage separately.

`HarnessCall.reuse_forecast` accepts a `ReuseForecast` containing task, session,
agent, branch and call identity; probability; future reuse horizon in milliseconds;
timezone-aware observation and expiry; and an evidence reference. Validity is capped
at one hour. Expired, future-dated or differently scoped forecasts are not applied.
Terminal/cancelled calls never authorize future copies through the adapter.

The evidence reference is a caller provenance label, not a certificate that a model
has been calibrated. The adapter does not authenticate it against a training artifact
registry. Such validation and actual forecast production remain required work.

The Scheduler rechecks adapter forecast identity and validity after transport, so
a forecast that expires between hint construction and routing cannot authorize
offload. Unavailable or malformed adapter forecasts return explicit refusal reasons.
Existing direct clients without forecast metadata retain the explicit-hints protocol;
this compatibility path is not upgraded to calibrated evidence by this change.
Do not call this interface a trained predictor or a performance improvement.

When a forecast applies, `expected_resume_ms` carries its future horizon and metadata
contains its reference, call and timestamps. When it does not apply, the existing
Tool Wait/Resume duration field remains available for lifecycle compatibility, but
the reuse probability is zero. A past tool duration alone cannot authorize offload.

Predictions must be constructed for the actual next model call identity. Replacing
the call, agent or branch does not transfer the prediction. The SDK-specific runtime
hooks do not yet fetch forecasts from a producer automatically; this is the shared
edge contract and safe default, not completed live predictor integration.

## Metric completion requirements

- Log each forecast before dispatch, with exact scope, observation time, expiry,
  versioned producer identity and validated evidence reference.
- Join it to the later outcome by authenticated task/call identity, not position.
- Record reuse and expiry/terminal closure; pending outcomes are censored, not false.
- Score only valid, resolved predictions with Brier score, calibration bins and
  coverage by Harness/workload. Report missing and invalid forecast counts separately.
- Charge unused Store bytes and repeated Prefill from false negatives separately;
  a well-calibrated reuse probability does not itself establish an eviction probability.
- Use held-out real Harness traces; never fit and validate on the same paired probe.

Current evidence is CPU contract tests across all four adapters. No new GPU or
end-to-end improvement claim is introduced by this change. The A implementation route
and original implementation-plan retention requirements are unchanged.
