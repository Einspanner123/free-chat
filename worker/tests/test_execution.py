from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionStatus
from freechat_worker.execution import DurableExecutionDriver, EngineObservation


def command(**changes: Any) -> ExecutionCommand:
    return ExecutionCommand.model_validate(
        {
            "tenant_id": "tenant",
            "request_id": "request",
            "decision_id": "decision",
            "worker_id": "worker",
            "worker_generation": 1,
            "engine_instance_id": "engine",
            "action": "query",
            **changes,
        }
    )


class Backend:
    def __init__(self) -> None:
        self.submissions: list[str] = []
        self.aborts: list[str] = []
        self.queries: list[str] = []
        self.status = ExecutionStatus.RUNNING
        self.quiet = self.fenced = False
        self.submit_error = self.abort_error = self.query_error = False
        self.overrides: dict[str, Any] = {}
        self.started = asyncio.Event()
        self.submit_wait: asyncio.Event | None = None

    async def submit(self, engine_request_id: str, payload: Any) -> None:
        self.submissions.append(engine_request_id)
        self.started.set()
        if self.submit_wait is not None:
            await self.submit_wait.wait()
        if self.submit_error:
            raise RuntimeError("lost_submit_ack")

    async def abort(self, engine_request_id: str) -> None:
        self.aborts.append(engine_request_id)
        if self.abort_error:
            raise RuntimeError("abort_unavailable")

    async def query(self, engine_request_id: str) -> EngineObservation:
        self.queries.append(engine_request_id)
        if self.query_error:
            raise RuntimeError("query_unavailable")
        return EngineObservation.model_validate(
            {
                "engine_request_id": engine_request_id,
                "observed_at": datetime.now(UTC),
                "status": self.status,
                "quiescent": self.quiet,
                "submission_fenced": self.fenced,
                **self.overrides,
            }
        )


def driver(path: Path, backend: Backend, **kwargs: Any) -> DurableExecutionDriver:
    return DurableExecutionDriver(
        path, backend, worker_id="worker", generation=1, engine_instance_id="engine", **kwargs
    )


@pytest.fixture
def runtime(tmp_path: Path) -> Any:
    backend = Backend()
    gate = driver(tmp_path / "journal.db", backend, create=True)
    yield gate, backend
    gate.close()


async def test_query_unknown_does_not_close_admission(runtime: Any) -> None:
    gate, backend = runtime
    observed = await gate.observe(command())
    assert observed.status == ExecutionStatus.UNKNOWN and not observed.releasable
    assert not observed.admission_closed and not backend.queries
    await gate.admit(command(), {"prompt": "not persisted"})
    assert len(backend.submissions) == 1
    with pytest.raises(ValueError, match="admission_closed"):
        await gate.admit(command(), {})


async def test_abort_before_arrival_survives_reopen_and_lost_receipt(tmp_path: Path) -> None:
    path, backend = tmp_path / "journal.db", Backend()
    gate = driver(path, backend, create=True)
    first = await gate.observe(command(action="abort"))
    assert first.status == ExecutionStatus.NOT_ACCEPTED and first.releasable
    gate.close()
    gate = driver(path, backend)
    try:
        second = await gate.observe(command(action="abort"))
        assert second.observation_sequence > first.observation_sequence
        assert second.releasable
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(), {})
        assert not backend.submissions and not backend.aborts and not backend.queries
    finally:
        gate.close()


@pytest.mark.parametrize("status", list(ExecutionStatus))
@pytest.mark.parametrize(
    "quiet,fenced", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_backend_release_matrix(
    runtime: Any, status: ExecutionStatus, quiet: bool, fenced: bool
) -> None:
    gate, backend = runtime
    await gate.admit(command(), {})
    backend.status, backend.quiet, backend.fenced = status, quiet, fenced
    observed = await gate.observe(command(action="abort"))
    assert observed.releasable == (
        status in {ExecutionStatus.COMPLETED, ExecutionStatus.ABORTED} and quiet and fenced
    )
    assert len(backend.aborts) == 1
    if status == ExecutionStatus.NOT_ACCEPTED:
        assert observed.status == ExecutionStatus.UNKNOWN


async def test_dispatch_ack_loss_is_not_resubmitted_after_restart(tmp_path: Path) -> None:
    path, backend = tmp_path / "journal.db", Backend()
    gate = driver(path, backend, create=True)
    backend.submit_error = True
    with pytest.raises(RuntimeError, match="lost_submit_ack"):
        await gate.admit(command(), {"prompt": "private prompt"})
    first = await gate.observe(command())
    gate.close()
    gate = driver(path, backend)
    try:
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(), {})
        second = await gate.observe(command())
        assert second.observation_sequence > first.observation_sequence
        assert not second.releasable and len(backend.submissions) == 1
        assert b"private prompt" not in path.read_bytes()
    finally:
        gate.close()


async def test_cancellation_during_submit_stays_uncertain(runtime: Any) -> None:
    gate, backend = runtime
    backend.submit_wait = asyncio.Event()
    task = asyncio.create_task(gate.admit(command(), {}))
    await backend.started.wait()
    with pytest.raises(ValueError, match="operations_in_flight"):
        gate.close()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (await gate.observe(command(action="abort"))).releasable
    with pytest.raises(ValueError, match="admission_closed"):
        await gate.admit(command(), {})


async def test_admit_abort_race_serializes_but_other_requests_progress(runtime: Any) -> None:
    gate, backend = runtime
    backend.submit_wait = asyncio.Event()
    submit = asyncio.create_task(gate.admit(command(), {}))
    await backend.started.wait()
    abort = asyncio.create_task(gate.observe(command(action="abort")))
    other = await gate.observe(command(decision_id="other", action="abort"))
    assert other.releasable and not backend.aborts
    backend.submit_wait.set()
    await submit
    assert not (await abort).releasable
    assert len(backend.submissions) == len(backend.aborts) == 1


async def test_simultaneous_duplicate_admissions_only_submit_once(runtime: Any) -> None:
    gate, backend = runtime
    results = await asyncio.gather(
        *(gate.admit(command(), {}) for _ in range(20)), return_exceptions=True
    )
    assert sum(isinstance(value, str) for value in results) == 1
    assert len(backend.submissions) == 1


@pytest.mark.parametrize("failure", ["abort_error", "query_error"])
async def test_backend_failure_keeps_fence_and_retries_abort(runtime: Any, failure: str) -> None:
    gate, backend = runtime
    await gate.admit(command(), {})
    setattr(backend, failure, True)
    with pytest.raises(RuntimeError):
        await gate.observe(command(action="abort"))
    with pytest.raises(ValueError, match="admission_closed"):
        await gate.admit(command(), {})
    setattr(backend, failure, False)
    assert not (await gate.observe(command())).releasable
    assert len(backend.aborts) == 2  # durable cancel intent also retried on QUERY


async def test_terminal_proof_survives_restart_without_backend(tmp_path: Path) -> None:
    path, backend = tmp_path / "journal.db", Backend()
    gate = driver(path, backend, create=True)
    await gate.admit(command(), {})
    backend.status, backend.quiet, backend.fenced = ExecutionStatus.COMPLETED, True, True
    first = await gate.observe(command())
    gate.close()
    gate = driver(path, backend)
    try:
        backend.abort_error = backend.query_error = True
        second = await gate.observe(command(action="abort"))
        assert first.releasable and second.releasable
        assert second.status == ExecutionStatus.COMPLETED
        assert second.observation_sequence == first.observation_sequence + 1
    finally:
        gate.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"worker_id": "other"},
        {"worker_generation": 2},
        {"engine_instance_id": "other"},
        {"request_id": "other"},
        {"tenant_id": "x" * 9000},
    ],
)
async def test_identity_fences(runtime: Any, changes: dict[str, Any]) -> None:
    gate, backend = runtime
    await gate.observe(command())
    with pytest.raises(ValueError):
        await gate.admit(command(**changes), {})
    assert not backend.submissions


async def test_tenants_have_distinct_engine_identities(runtime: Any) -> None:
    gate, _ = runtime
    assert await gate.admit(command(), {}) != await gate.admit(command(tenant_id="other"), {})


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine_request_id": "other"},
        {"observed_at": datetime.now(UTC) - timedelta(minutes=1)},
        {"observed_at": datetime.now(UTC) + timedelta(hours=1)},
        {"observed_at": datetime(2026, 9, 20)},
    ],
)
async def test_bad_observations_fail_closed(runtime: Any, overrides: dict[str, Any]) -> None:
    gate, backend = runtime
    await gate.admit(command(), {})
    backend.overrides = overrides
    with pytest.raises(ValueError):
        await gate.observe(command())


async def test_bounded_journal_does_not_evict_tombstones(tmp_path: Path) -> None:
    gate = driver(tmp_path / "journal.db", Backend(), create=True, max_records=1)
    try:
        assert (await gate.observe(command(action="abort"))).releasable
        with pytest.raises(ValueError, match="journal_full"):
            await gate.observe(command(decision_id="other"))
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(), {})
        assert len(gate._locks) == 1
    finally:
        gate.close()


def test_owner_incarnation_and_missing_storage_fences(tmp_path: Path) -> None:
    path, backend = tmp_path / "journal.db", Backend()
    with pytest.raises(ValueError, match="missing_requires_recovery"):
        driver(path, backend)
    with pytest.raises(ValueError, match="limit_invalid"):
        driver(path, backend, create=True, max_records=0)
    gate = driver(path, backend, create=True)
    with pytest.raises(ValueError, match="already_owned"):
        driver(path, backend)
    gate.close()
    with pytest.raises(ValueError, match="incarnation_mismatch"):
        DurableExecutionDriver(
            path, backend, worker_id="worker", generation=2, engine_instance_id="engine"
        )
    recovered = driver(path, backend)
    recovered.close()
    recovered.close()


@pytest.mark.parametrize("after_commit", [False, True])
async def test_dispatch_storage_failure_never_calls_backend(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, after_commit: bool
) -> None:
    gate, backend = runtime
    await gate.observe(command())
    original = gate._save

    def fail(key: str, record: Any) -> None:
        if after_commit:
            original(key, record)
        raise sqlite3.OperationalError("simulated disk/ack failure")

    monkeypatch.setattr(gate, "_save", fail)
    with pytest.raises(sqlite3.OperationalError):
        await gate.admit(command(), {})
    assert not backend.submissions
    monkeypatch.setattr(gate, "_save", original)
    if after_commit:
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(), {})
    else:
        await gate.admit(command(), {})


@pytest.mark.parametrize("after_commit", [False, True])
async def test_terminal_storage_failure_never_returns_uncommitted_receipt(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, after_commit: bool
) -> None:
    gate, backend = runtime
    await gate.admit(command(), {})
    backend.status, backend.quiet, backend.fenced = ExecutionStatus.COMPLETED, True, True
    original = gate._save

    def fail(key: str, record: Any) -> None:
        if after_commit:
            original(key, record)
        raise sqlite3.OperationalError("simulated disk/ack failure")

    monkeypatch.setattr(gate, "_save", fail)
    with pytest.raises(sqlite3.OperationalError):
        await gate.observe(command())
    monkeypatch.setattr(gate, "_save", original)
    result = await gate.observe(command())
    assert result.releasable
    assert result.observation_sequence == (2 if after_commit else 1)


@pytest.mark.parametrize("abort_first", [False, True])
async def test_abrupt_process_exit_preserves_fence(tmp_path: Path, abort_first: bool) -> None:
    path = tmp_path / "journal.db"
    # No driver.close(), sqlite.close(), or graceful shutdown in the child.
    source = """
import asyncio, os, sys
from pathlib import Path
from freechat_contracts.execution import ExecutionCommand
from freechat_worker.execution import DurableExecutionDriver
class Backend:
    async def submit(self, key, payload):
        os._exit(23)
async def main():
    gate = DurableExecutionDriver(Path(sys.argv[1]), Backend(), worker_id="worker",
        generation=1, engine_instance_id="engine", create=True)
    command = ExecutionCommand(tenant_id="tenant", request_id="request", decision_id="decision",
        worker_id="worker", worker_generation=1, engine_instance_id="engine", action="query")
    if sys.argv[2] == "True":
        await gate.observe(command.model_copy(update={"action": "abort"}))
        os._exit(23)
    await gate.admit(command, {})
asyncio.run(main())
"""
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-c", source, str(path), str(abort_first)
    )
    try:
        assert await asyncio.wait_for(child.wait(), timeout=5) == 23
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    backend = Backend()
    gate = driver(path, backend)
    try:
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(), {})
        result = await gate.observe(command())
        assert result.releasable == abort_first
        assert result.observation_sequence == (2 if abort_first else 1)
        assert not backend.submissions
    finally:
        gate.close()


def test_symlink_cannot_bypass_single_owner(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    gate = driver(path, Backend(), create=True)
    alias = tmp_path / "alias.db"
    alias.symlink_to(path)
    try:
        with pytest.raises(ValueError, match="already_owned"):
            driver(alias, Backend())
    finally:
        gate.close()


@pytest.mark.parametrize("create", [False, True])
@pytest.mark.parametrize("metadata_only", [False, True])
def test_existing_empty_database_is_not_silently_initialized(
    tmp_path: Path, metadata_only: bool, create: bool
) -> None:
    path = tmp_path / "journal.db"
    with sqlite3.connect(path) as db:
        if metadata_only:
            db.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE executions (key TEXT PRIMARY KEY, record TEXT NOT NULL)")
    with pytest.raises(ValueError, match="uninitialized"):
        driver(path, Backend(), create=create)


async def test_closed_driver_and_abort_identity_cannot_admit(runtime: Any) -> None:
    gate, _ = runtime
    with pytest.raises(ValueError, match="requires_query_identity"):
        await gate.admit(command(action=ExecutionAction.ABORT), {})
    gate.close()
    with pytest.raises(ValueError, match="driver_closed"):
        await gate.observe(command())


async def test_real_local_process_exit_required_for_release(tmp_path: Path) -> None:
    """A real CPU child, not inference, CUDA work, or a physical GPU attestation."""

    class ProcessBackend(Backend):
        process: asyncio.subprocess.Process

        async def submit(self, engine_request_id: str, payload: Any) -> None:
            await super().submit(engine_request_id, payload)
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import sys; sys.stdin.read()",
                stdin=asyncio.subprocess.PIPE,
            )

        async def abort(self, engine_request_id: str) -> None:
            await super().abort(engine_request_id)
            # Deliberately only record intent; process is still awaiting input.

        async def query(self, engine_request_id: str) -> EngineObservation:
            stopped = self.process.returncode is not None
            return EngineObservation(
                engine_request_id=engine_request_id,
                observed_at=datetime.now(UTC),
                status=ExecutionStatus.ABORTED if stopped else ExecutionStatus.RUNNING,
                quiescent=stopped,
                submission_fenced=stopped,
            )

    backend = ProcessBackend()
    gate = driver(tmp_path / "journal.db", backend, create=True)
    try:
        await gate.admit(command(), {})
        assert not (await gate.observe(command(action="abort"))).releasable
        assert backend.process.returncode is None
        assert backend.process.stdin is not None
        backend.process.stdin.close()
        await asyncio.wait_for(backend.process.wait(), timeout=5)
        assert (await gate.observe(command(action="abort"))).releasable
        with pytest.raises(ValueError, match="admission_closed"):
            await gate.admit(command(), {})
    finally:
        if hasattr(backend, "process") and backend.process.returncode is None:
            backend.process.kill()
            await backend.process.wait()
        gate.close()


class AsyncEngineFixture:
    """Models the pinned collector/utility boundary, not real GPU completion."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
            ),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            kv_transfer_config=None,
            ec_transfer_config=None,
        )
        self.engine_core = self
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.collector = SimpleNamespace(request_id="internal-identity", get=self.queue.get)
        self.calls: list[Any] = []
        self.quiet = False
        self.error = False

    async def add_request(self, *args: Any) -> Any:
        self.calls.append(("submit", args[0]))
        if self.error:
            raise RuntimeError("lost_add_ack")
        return self.collector

    async def abort(self, request_id: str, internal: bool) -> None:
        self.calls.append(("abort", request_id, internal))

    async def call_utility_async(self, method: str, request_id: str) -> dict[str, Any]:
        self.calls.append(("query", method, request_id))
        return {"request_id": request_id, "quiescent": self.quiet}


async def test_vllm_adapter_requires_engine_barrier_after_final_output(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from freechat_worker.vllm_execution import VllmExecutionBackend

    engine = AsyncEngineFixture()
    backend = VllmExecutionBackend(engine)
    gate = driver(tmp_path / "actual-adapter.db", backend, create=True)  # type: ignore[arg-type]
    try:
        key = await gate.admit(
            command(),
            {
                "prompt": "hello",
                "sampling_params": SimpleNamespace(n=1),
            },
        )
        engine.queue.put_nowait(SimpleNamespace(finished=True))
        assert len([out async for out in backend.stream(key)]) == 1
        assert not (await gate.observe(command())).releasable
        engine.quiet = True
        receipt = await gate.observe(command())
        assert receipt.releasable and receipt.status == ExecutionStatus.COMPLETED
        assert ("query", "freechat_execution_quiescent", "internal-identity") in engine.calls
        with pytest.raises(ValueError, match="execution_admission_closed"):
            await gate.admit(command(), {})
    finally:
        gate.close()


async def test_vllm_adapter_abort_signal_does_not_prove_quiescence() -> None:
    from types import SimpleNamespace

    from freechat_worker.vllm_execution import VllmExecutionBackend

    engine = AsyncEngineFixture()
    backend = VllmExecutionBackend(engine)
    await backend.submit("id", {"prompt": "x", "sampling_params": SimpleNamespace(n=1)})
    await backend.abort("id")
    assert ("abort", "internal-identity", True) in engine.calls
    assert not (await backend.query("id")).quiescent
    engine.quiet = True
    observed = await backend.query("id")
    assert observed.quiescent and observed.status == ExecutionStatus.ABORTED
    assert [out async for out in backend.stream("id")] == []


async def test_vllm_adapter_uncertain_submit_never_retries_or_releases() -> None:
    from types import SimpleNamespace

    from freechat_worker.vllm_execution import VllmExecutionBackend

    engine = AsyncEngineFixture()
    engine.error = True
    backend = VllmExecutionBackend(engine)
    payload = {"prompt": "x", "sampling_params": SimpleNamespace(n=1)}
    with pytest.raises(RuntimeError, match="lost_add_ack"):
        await backend.submit("id", payload)
    with pytest.raises(ValueError, match="duplicate_backend_submit"):
        await backend.submit("id", payload)
    observed = await backend.query("id")
    assert observed.status == ExecutionStatus.UNKNOWN
    assert not observed.quiescent and not observed.submission_fenced
    assert (await backend.query("never-seen")).status == ExecutionStatus.UNKNOWN


@pytest.mark.parametrize("unsupported", ["n", "streaming", "multimodal"])
async def test_vllm_adapter_rejects_untracked_child_or_input_shapes(unsupported: str) -> None:
    from types import SimpleNamespace

    from freechat_worker.vllm_execution import VllmExecutionBackend

    backend = VllmExecutionBackend(AsyncEngineFixture())
    prompt: Any = "x"
    if unsupported == "streaming":
        prompt = object()
    if unsupported == "multimodal":
        prompt = {"prompt_token_ids": [1], "multi_modal_data": {}}
    with pytest.raises(ValueError, match="execution_adapter_requires"):
        await backend.submit(
            "id",
            {
                "prompt": prompt,
                "sampling_params": SimpleNamespace(n=2 if unsupported == "n" else 1),
            },
        )


@pytest.mark.parametrize(
    "feature",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "async_scheduling",
        "kv_transfer_config",
        "ec_transfer_config",
    ],
)
def test_vllm_adapter_rejects_unverified_execution_modes(feature: str) -> None:
    from freechat_worker.vllm_execution import VllmExecutionBackend

    engine = AsyncEngineFixture()
    if feature.endswith("_size"):
        setattr(engine.vllm_config.parallel_config, feature, 2)
    elif feature == "async_scheduling":
        engine.vllm_config.scheduler_config.async_scheduling = True
    else:
        setattr(engine.vllm_config, feature, object())
    with pytest.raises(ValueError, match="requires_sync_single_gpu"):
        VllmExecutionBackend(engine)


async def test_submitted_key_is_read_only_and_requires_prior_dispatch(runtime: Any) -> None:
    gate, backend = runtime
    before = gate._db.total_changes
    with pytest.raises(ValueError, match="execution_admission_unknown"):
        gate.submitted_key(command())
    assert gate._db.total_changes == before
    key = await gate.admit(command(), {})
    before = gate._db.total_changes
    assert gate.submitted_key(command()) == key
    assert gate._db.total_changes == before
    assert len(backend.submissions) == 1
    assert backend.aborts == backend.queries == []


async def test_submitted_key_rejects_observed_but_never_dispatched(runtime: Any) -> None:
    gate, _ = runtime
    await gate.observe(command())
    before = gate._db.total_changes
    with pytest.raises(ValueError, match="execution_not_submitted"):
        gate.submitted_key(command())
    assert gate._db.total_changes == before


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "foreign"},
        {"request_id": "foreign"},
        {"decision_id": "foreign"},
        {"worker_generation": 2},
        {"engine_instance_id": "replacement"},
    ],
)
async def test_submitted_key_rejects_foreign_identity_without_mutation(
    runtime: Any, changes: dict[str, Any]
) -> None:
    gate, backend = runtime
    await gate.admit(command(), {})
    before = gate._db.total_changes
    with pytest.raises(ValueError):
        gate.submitted_key(command(**changes))
    assert gate._db.total_changes == before
    assert len(backend.submissions) == 1 and not backend.aborts
