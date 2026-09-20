# Worker admission journal: local evidence

LOCAL_CONTRACT_ONLY. **531 passed, 12 GPU-dependency tests skipped** in the full
Python regression suite; 56 additional cases relative to request-execution evidence.
Ruff passed. Strict mypy passed for 114 source files. Python 3.12.14, local macOS.

The new Worker module has 162/162 covered statements and 53/54 covered branches
(about 99% combined). The remaining branch is constructor connection failure before
the SQLite connection attribute exists. Selected execution contract/Scheduler/Worker
modules collectively have 469 statements and 142 branches; 2 statements and 2 branches
remain uncovered. This is deliberately not a whole-repository coverage claim.

## What actually ran

- Real SQLite WAL/FULL transactions, identity bounds, a single-owner OS lock, symlink
  alias rejection and reopen checks, including two child processes exiting through
  `os._exit(23)` without graceful cleanup.
- A real local Python child waiting on stdin: recording abort intent does not release;
  only observed process exit allows its fixture backend to assert quiescence.
- All five observation statuses crossed with quiescent/submission-fenced combinations;
  negative lookup after dispatch never supplies NOT_ACCEPTED release proof.
- Concurrent duplicate admission, admission/abort ordering, unrelated identity
  progress, dispatch cancellation/ack loss and before/after-commit injected failures.
- Real loopback gRPC joining this durable Worker gate to Scheduler reservation and
  reconciliation. Late arrival is rejected after pre-arrival abort; an admitted call
  holds the sole reservation until the fixture reports confirmed termination.
- Full existing Python regression suite, not just the new module tests.

There was no SSH, remote change, inference/GPU execution, H100 deployment or actual
etcd/NATS fault test. The local fork mirror has no complete engine entrypoint, and no
real vLLM backend is installed or implicitly enabled. Twelve GPU tests were skipped,
not waived. No system performance/resume metric was added. No dependency/protobuf
change was made; uv lock consistency and Buf breaking were not rerun (tools unavailable).

## Reproduction and provenance

The run used `rewrite/.venv` with explicit colon-separated PYTHONPATH entries for
libs/{contracts,control-store,protocol,harness-adapters}/src,
services/{gateway,scheduler,trace-replay,kv-controller,topology-agent}/src and worker/src.

```sh
ruff check .
python -m mypy libs services worker benchmarks
python -m pytest -q --tb=short --cov=freechat_worker.execution --cov=freechat_scheduler.request_execution --cov=freechat_scheduler.request_ledger --cov=freechat_contracts.execution --cov-branch --cov-report=term-missing --cov-report=json:coverage.json --junitxml=tests.xml
```

`pytest.log`, `tests.xml`, `coverage.json`, `ruff.log` and `mypy.log` are actual command
outputs. `manifest.sha256` pins current Python source/tests, protocol definitions,
plans/docs, dependency locks and these evidence outputs (including this README).
Paths in the manifest are relative to `rewrite/`; run `shasum -a 256 -c` there to verify.
Git identity is unavailable because this local workspace has no Git metadata. Earlier
evidence directories were not modified. Hashes record this source snapshot, not future
changes or proof of physical execution.
