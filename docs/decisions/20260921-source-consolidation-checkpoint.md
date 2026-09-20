# Ross source consolidation checkpoint, 2026-09-21

Status: PARTIAL; runtime behavior reconciliation remains CONFLICT.

## Completed changes

- Recovered the exact vLLM fork Git objects from workstation through read-only pack
  export and imported them on ross. No workstation file/ref/service was changed.
- Fast-forwarded the Ross fork branch with ancestor and expected-old-ref checks.
- Preserved the original Ross working tree (including untracked, nonignored files)
  before any source consolidation. A final tar comparison, diff hash and status hash
  all matched the original. Main remains at 709a7b0.
- Created an isolated Ross worktree on chore/ross-source-governance.
- Checkpoint c3054ff080ebe3ecb6e4e3a0309f10d88fd7a26c preserves the previously
  uncommitted Ross source and experiment evidence.
- Checkpoint 4ce8f2753dae1c3c24df69dfc89df4e5d6f2ffcc adds the pinned submodule,
  source URI/tree/path metadata, third-party Ruff exclusion and Ross-only rules.
- No changes were pushed to GitHub; the branch exists in the Ross repository.

## Verification of checkpoint 4ce8f275

- Ross CPU regression: 148 passed, 2 skipped for missing GPU dependencies,
  10 GPU-marked tests deliberately deselected. No GPU workload was run.
- Ruff: passed for tools, tests, libs, services, worker and benchmarks.
- Strict mypy: passed for 95 files in libs, services, worker and benchmarks.
- Fresh clone from the Ross parent repository, using --no-hardlinks: passed.
- Recursive submodule initialization used a command-scoped URL rewrite from the
  canonical SSH URI to the same on-host bare repository (file transport explicitly
  enabled only for that command). SSH-key distribution on every future node was not
  tested. A separate git ls-remote against the canonical SSH URI returned the pin.
- Gitlink, YAML pin, .gitmodules URL, submodule HEAD/tree, upstream ancestry and
  clean working state all matched in the fresh clone.
- Image digest remains UNRESOLVED: this is not release acceptance.

Raw test XML/log and mypy log are retained outside the source tree under
/media/ross/8TB/linkst/freechat/consolidation-20260921-xr6ecd/.
The fresh clone and protected snapshots are in the same private recovery directory.
Do not publish its protected configuration or use it as a second permanent authority.

## Not completed

The local snapshot's RequestLedger/execution-confirmation implementation has not
been imported. Its historical 531-test result does not apply to this checkpoint.
See docs/source-governance.md for the exact conflict and the pending owner question.
Do not infer approval from the recommended answer being preselected.

Version-consumer unification, a committed automated consistency checker, full
post-merge regression, canonical worktree transition and the real engine adapter
remain pending. No remote service, container or deployment was changed.
