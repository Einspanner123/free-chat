# Ross consolidation acceptance: 2026-09-21

Evidence level: LOCAL_SOURCE_AND_CPU_CONTRACT_ONLY.

## Source identity

- Validated FreeChat commit: 53a73da09a976c7df7b1d74fae4826c234216580.
- Branch: chore/ross-source-governance.
- Fork commit: 8e78a3c613072632aa822c9aed2f698e76046219.
- Fork tree: 45110e50047f52bef7628f2d616945fdb5244468.
- Owner resolution: "确认使用本地已验证实现", 2026-09-21.
- All source edits and validation in this task were performed on ross.

## Actual results

Both the consolidation worktree and a fresh clone completed **551 passed,
2 skipped, 10 deselected**. The deselected tests are GPU-marked; the two skipped
tests require the absent GPU dependency group. No GPU workload was executed.

Ruff passed; strict mypy passed for 118 files (libs/services/worker/benchmarks/tools).
Coverage of the shared source-lock reader is 100%; the source-check CLI module has
97% combined coverage in the CPU suite (its direct module-entry line is outside
that pytest coverage run). These are not whole-project coverage percentages.

Protobuf was regenerated with grpcio-tools 1.81.1, matching uv.lock. A second
generation in the fresh clone produced no Git differences.
The Ross calibration/transfer implementations, forecast adapter and Worker
telemetry were preserved unchanged relative to the Ross checkpoint.

Validation used Python 3.12.3 and the existing Ross CPU environment with explicit
PYTHONPATH entries pointing to the worktree or fresh clone being tested. It did not
install dependencies or reuse the Mac environment. Checked versions of pytest,
grpcio-tools, PyYAML, Pydantic and LangGraph match the lock. This is not a successful
from-empty-environment installation test.

The dependency change makes existing locked PyYAML 6.0.3 an explicit root dev
dependency. Only two root dependency/metadata entries were added to uv.lock; every
other package record was compared and is unchanged. uv lock --check --offline
passed, resolving/checking 155 packages. Online full-resolution attempts failed
with TLS EOF from PyPI; TLS verification was not disabled and no mirror substituted.

## Source reconstruction and honest failures

The fresh clone was made on ross with --no-hardlinks and submodule initialization.
Initialization used a command-scoped rewrite of the canonical SSH URL to the same
on-host bare repository with file transport explicitly permitted for that command.

fresh-source-check.json records PASS for committed/index gitlink, URL, actual
commit/tree, upstream ancestry and clean source on the fresh clone.
The new checker has negative tests for stale pins, index/worktree drift, invalid
metadata, uninitialized submodules, dirty sources, and unresolved release digests.

The separate --check-remote probe FAILED. Direct diagnosis returned:
"ssh: Could not resolve hostname ross: Temporary failure in name resolution".
remote-source-check.json preserves the failure. The on-host source repository is
readable and was verified, but this does not certify SSH name resolution/access
from every node. No SSH, DNS or host configuration was changed.

release-source-check.json deliberately rejects release because
worker_image_digest remains UNRESOLVED. Source consistency is not image/GPU
acceptance. The checker is available as a command, not yet a mandatory hosted CI gate.

## Runtime safety and pending work

The original main working directory is used by a running Gateway and is underneath
two container bind mounts. It was NOT switched. A tar comparison confirmed its
tracked/untracked snapshot remains identical to the pre-consolidation backup.

The approved implementation is committed in the isolated Ross development worktree:
 /media/ross/8TB/linkst/freechat/consolidation-20260921-xr6ecd/governance

Workstation, running services, containers, original main source and the Mac source
mirror were not changed. Nothing was pushed to GitHub. Historical plans and evidence
were retained. Canonical checkout transition needs explicit maintenance authorization;
the local hostname/SSH configuration needs a separately scoped resolution.

Real vLLM admission/cancellation/completion integration, per-rank budget reconciliation,
physical GPU/Harness acceptance, immutable worker-image publication and production
claims remain incomplete.

## Reproduction

Use the locked CPU environment and initialize the pinned submodule first.

```sh
python -m pytest -q -m "not gpu and not multinode"
ruff check .
python -m mypy libs services worker benchmarks tools
uv lock --check --offline
python -m tools.check_source_checkout
python -m tools.check_source_checkout --release
python -m tools.check_source_checkout --check-remote
```

The last two commands are currently expected to fail for the explicit reasons above.
Recorded XML/log/coverage/JSON artifacts are actual command outputs. manifest.sha256
pins this evidence set and implementation source at capture time; paths are relative
to the FreeChat root. Later documentation/evidence commits do not change the identity
of the source commit tested above.
