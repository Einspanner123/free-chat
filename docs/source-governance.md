# Source ownership and Ross-only development

## Authority

All FreeChat and fork source changes are made on ross. Workstation is a deployment
and explicitly authorized GPU-validation host, not a second development location.
Do not edit project source in workstation or the Mac mirror. A read-only source
recovery from workstation on 2026-09-21 restored Git objects already verified at
the pinned revision; it did not change workstation files or running services.

FreeChat retains its existing Git history. The independent vLLM repository is
`ssh://linkst@ross/media/ross/8TB/linkst/freechat/vllm-fork.git`.
The complete engine is checked out at `third_party/vllm` as a Git submodule.
The gitlink pins the commit; `versions.lock.yaml` must agree with that pointer
and records the expected upstream and tree. Never use submodule update --remote
as part of a pinned build.

Publish a fork commit to the Ross repository before updating FreeChat's gitlink.
Do not force-push, rewrite historical evidence, overwrite unknown working changes,
or update any running service as a side effect of a source synchronization.

## Consolidation checkpoint (2026-09-21)

Original FreeChat HEAD: `709a7b0a000b2002b37d2ab9815167aa23bd4890`.
Preserved Ross source/evidence checkpoint:
`c3054ff080ebe3ecb6e4e3a0309f10d88fd7a26c` on
`chore/ross-source-governance`. The original main worktree is untouched.

An isolated worktree under
`/media/ross/8TB/linkst/freechat/consolidation-20260921-xr6ecd/governance`
holds this branch during reconciliation. This is a temporary safety worktree,
not an additional permanent development authority. Only after reconciliation
will the canonical project working directory be switched in a separate safe step.

The sibling private recovery directory retains the original working-tree archive,
Git diff, status, initial refs, protected configuration where present, and the
read-only import of the local development snapshot. Do not publish that directory
or configuration secrets. Archive SHA-256:
`3f3e8c55750170991a886e6612ebf52e7a253b28057fc4d25ea63feac91dd0eb`.

The fork branch was fast-forwarded from
`5871f38ee947760ed7f5dd5487bb04456db8bc98` to
`8e78a3c613072632aa822c9aed2f698e76046219`, with an explicit ancestor check
and expected-old-value ref update. Its tree is
`45110e50047f52bef7628f2d616945fdb5244468`.

## Confirmed reconciliation decision

Status: **RESOLVED / owner approved on 2026-09-21**.

Owner response: "确认使用本地已验证实现". Adopt RequestLedger and Worker execution
confirmation; preserve Ross calibration/forecast features and regenerate protobuf.

The local snapshot replaces immediate Release/removal with RequestLedger and
incarnation-bound Worker execution confirmation. This is a real capacity-release
behavior change, not whitespace. The owner has explicitly approved adopting that
previously locally tested implementation while retaining Ross calibration and
forecast capabilities and regenerating protobuf.

Automatic text merging reports overlapping changes in scheduler.py, grpc_server.py,
control.proto and its two generated message files. Three same-name untracked tests
also differ: test_forecast_fence.py adds local cases; test_boundary_statistics.py
and test_calibration.py have formatting-only differences. Do not use whole-directory
overwrite or generated-file conflict resolution as a substitute for this decision.

The approved snapshot is now imported into the Ross consolidation branch.
Historical local test counts are not substituted for the new Ross regression.

## Current accepted implementation and checks

The owner-approved import now uses RequestLedger and Worker execution receipts.
The original Ross forecast adapter, calibration/transfer modules and Worker telemetry
were compared and preserved. Protobuf was regenerated with the locked grpcio-tools
1.81.1 on ross. Root README now points to the current root implementation plan;
historical .hermes plans remain intact.

`tools/source_versions.py` is the version-lock reader shared by source validation
and worker-image validation. `python -m tools.check_source_checkout` verifies the
committed/index gitlink, repository URI, actual submodule commit/tree, upstream
ancestry and clean source. `--allow-dirty` is development-only; it does not waive
cleanliness for `--release`. `--check-remote` fetches into a disposable local bare
repository without writing the source repository. A source check is not GPU/runtime
acceptance. These commands are not yet an enforced external CI workflow.

PyYAML 6.0.3 was already locked transitively. The root dev dependency and its two
lockfile metadata entries now declare that direct use. All other package records
are unchanged, and `uv lock --check --offline` passed (155 packages). An online
full-resolution attempt encountered TLS EOF at PyPI; no TLS verification was disabled,
no index was substituted, and no installed environment was changed.

Ross regression: 551 passed, 2 dependency skips, 10 GPU tests deselected.
Ruff passed; strict mypy passed for 118 files. Source-lock reader coverage is 100%;
the checkout validator has 97% combined coverage in the CPU suite.

## Canonical checkout transition is deferred

A read-only pre-transition inventory found a Gateway process (PID 2407820 at the
time of inspection) using the original project directory and two running containers
mounting its parent projects directory. This is a snapshot, not a permanent PID
identity. Do not switch that working directory under active consumers.

All new development is on the Ross governance branch/worktree described above.
The original directory remains a frozen runtime source pending explicit maintenance
authorization; workstation and the Mac snapshot are not development locations.
No service/container/process was stopped, restarted or upgraded.

## Remaining acceptance

- Record final commit identity and fresh-clone source-check evidence.
- Obtain maintenance authorization before changing the original runtime-bound checkout.
- Add the source-check command to the eventual trusted CI/release workflow.
- Continue the real engine execution adapter and all original acceptance gates.

The unresolved image digest still blocks release; no GPU, runtime upgrade,
performance result, external publication or HA acceptance is implied.
