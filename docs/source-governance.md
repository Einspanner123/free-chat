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

## Open reconciliation decision

Status: **CONFLICT / awaiting owner confirmation**.

The local snapshot replaces immediate Release/removal with RequestLedger and
incarnation-bound Worker execution confirmation. This is a real capacity-release
behavior change, not whitespace. The owner has been asked whether to adopt that
previously locally tested implementation while retaining Ross calibration and
forecast capabilities and regenerating protobuf.

Automatic text merging reports overlapping changes in scheduler.py, grpc_server.py,
control.proto and its two generated message files. Three same-name untracked tests
also differ: test_forecast_fence.py adds local cases; test_boundary_statistics.py
and test_calibration.py have formatting-only differences. Do not use whole-directory
overwrite or generated-file conflict resolution as a substitute for this decision.

Until resolved, the Mac snapshot is staging material, not an accepted Ross release.
The historical 531-test result does not verify the current consolidation branch.

## Remaining acceptance

- Record the owner's conflict resolution and reconcile the imported implementation.
- Unify version consumers and implement automated gitlink/lock/tree/dirty-state checks.
- Regenerate protocol outputs and run the Ross CPU regression and quality gates.
- Verify reconstruction from a fresh clone and matching submodule.
- Reconcile canonical plan/README entry points without deleting protected plans.
- Record exact source and test identities before switching the canonical worktree.

The unresolved image digest still blocks release; no GPU, runtime upgrade,
performance result, external publication or HA acceptance is implied.
