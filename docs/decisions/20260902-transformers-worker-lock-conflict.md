# Transformers worker lock conflict

- Status: `RESOLVED`
- Detected: 2026-09-02 Asia/Shanghai
- Resolved: 2026-09-02 Asia/Shanghai
- Owner decision: Option B; accept the candidate's resolved Transformers
  5.16.1, validate it, and freeze the accepted environment and image digest.
- Scope blocked: worker image acceptance, immutable image digest, three-model
  serving baseline, Harness end-to-end metrics, and worker capability claims.

## Conflicting constraints

The FreeChat dependency contract and `versions.lock.yaml` pin Transformers
5.14.1. During the CUDA 13 worker image build, vLLM's runtime dependency input
accepted any Transformers release at or above 5.5.3, so the resolver selected
5.16.1. The same vLLM fork's CUDA test input and compiled test lock explicitly
pin Transformers 5.14.1.

Evidence:

- FreeChat `versions.lock.yaml`: Transformers 5.14.1.
- vLLM `requirements/common.txt`: `transformers >= 5.5.3`.
- vLLM `requirements/test/cuda.in`: `transformers==5.14.1`.
- vLLM `requirements/test/cuda.txt`: `transformers==5.14.1`.
- Candidate image resolver output: Transformers 5.16.1.

The build was stopped before candidate acceptance. No worker image digest or
verified capability claim has been issued from this candidate.

## Options

### A — preserve Transformers 5.14.1 (recommended)

Add an explicit build/runtime constraint for Transformers 5.14.1 in the vLLM
fork, rebuild the candidate, and run the protocol, model, and worker-image
acceptance gates. This follows the existing owner-approved FreeChat lock and
the vLLM CUDA test lock.

### B — advance the worker lock to Transformers 5.16.1

Update the FreeChat lock and image validator, then run the complete protocol,
model, and Harness compatibility matrix before accepting the candidate. This
accepts the newest transitive resolution but expands the compatibility change
and diverges from the vLLM CUDA test lock.

## Conflict-state restrictions

Until the owner selects an option:

- do not resume or accept the candidate worker image build;
- do not populate `worker_image_digest`;
- do not mark the worker image or any dependent capability as verified;
- do not publish performance or reliability claims from this candidate.

## Exit condition

Only an explicit owner selection of option A or B closes this conflict. Record
the decision and exact implementation commit here, change the status to
`RESOLVED`, and complete the affected acceptance gates before issuing claims.

## Resolution

The owner selected option B after confirming that no Transformers runtime
failure had occurred: the only observed failure was the project's exact-version
gate. FreeChat now expects Transformers 5.16.1 for the candidate worker. The
candidate still requires GPU, protocol, model, and immutable-digest acceptance;
this resolution does not itself verify any capability or performance claim.

- FreeChat resolution commit: `dfbd566`.
- Candidate vLLM fork commit: `b95da863d262242e33b8388889baca83a70508a1`.
