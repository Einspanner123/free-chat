"""Single-owner, durable Worker admission boundary (not a vLLM engine adapter).

SQLite stores identities and tombstones, never prompts or KV tensors. All inference
ingress must use admit(); an independent engine endpoint invalidates this boundary.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from freechat_contracts.execution import (
    ExecutionAction,
    ExecutionCommand,
    ExecutionReceipt,
    ExecutionStatus,
)
from pydantic import BaseModel, ConfigDict, Field

from freechat_worker.runtime import ContainerBinding, DockerStopProof


class EngineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_request_id: str = Field(min_length=1)
    observed_at: datetime
    status: ExecutionStatus
    quiescent: bool = False
    # Includes in-flight submit delivery/retries, not just absence from a queue.
    submission_fenced: bool = False


class ExecutionBackend(Protocol):
    async def submit(self, engine_request_id: str, payload: Mapping[str, Any]) -> None:
        """Submit this exact identity, including any engine-created child requests."""
        ...

    async def abort(self, engine_request_id: str) -> None:
        """Signal only. Returning is NOT proof of termination or delivery fencing."""
        ...

    async def query(self, engine_request_id: str) -> EngineObservation:
        """Observe all children, execution and pending submissions for the identity."""
        ...


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: ExecutionCommand
    dispatched: bool = False
    closed: bool = False
    cancel_requested: bool = False
    terminal: ExecutionStatus | None = None
    sequence: int = 0


class DurableExecutionDriver:
    """Local Unix, one owner and one event loop per journal/engine incarnation.

    A dispatch intent is committed before calling the backend. Uncertain dispatches
    are queried, never resubmitted. Loss of journal storage requires explicit operator
    recovery: opening a missing journal fails unless create=True is provided.
    """

    def __init__(
        self,
        path: Path,
        backend: ExecutionBackend | None,
        *,
        worker_id: str,
        generation: int,
        engine_instance_id: str,
        create: bool = False,
        container_binding: ContainerBinding | None = None,
        retired: bool = False,
        max_records: int = 10_000,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        identity = ExecutionCommand(
            tenant_id="identity",
            request_id="identity",
            decision_id="identity",
            worker_id=worker_id,
            worker_generation=generation,
            engine_instance_id=engine_instance_id,
            action=ExecutionAction.QUERY,
        )
        if max_records < 1:
            raise ValueError("execution_journal_limit_invalid")
        path = path.resolve()
        if not create and not path.is_file():
            raise ValueError("execution_journal_missing_requires_recovery")
        if backend is None and not retired:
            raise ValueError("execution_backend_required")
        if retired and create:
            raise ValueError("retired_journal_cannot_be_created")
        initialize = create and not path.exists()
        self.identity = (
            identity.worker_id,
            identity.worker_generation,
            identity.engine_instance_id,
        )
        self.backend, self.clock, self.max_records = backend, clock, max_records
        self._locks: dict[str, asyncio.Lock] = {}
        self._active = 0
        self._closed = False
        self._owner = path.with_suffix(path.suffix + ".owner").open("a+b")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise ValueError("execution_journal_already_owned") from None
        try:
            # mode=rw prevents an accidental replacement after the existence check.
            self._db = sqlite3.connect(
                path.resolve().as_uri() + ("?mode=rwc" if create else "?mode=rw"), uri=True
            )
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._initialize(identity, create=initialize)
            self.retirement: DockerStopProof | None = None
            row = self._db.execute("SELECT value FROM meta WHERE id=2").fetchone()
            if row is not None:
                self.retirement = DockerStopProof.model_validate_json(row[0])
            if (self.retirement is not None) != retired:
                raise ValueError(
                    "execution_incarnation_retired"
                    if self.retirement
                    else "execution_retirement_unproven"
                )
            if initialize and container_binding is not None:
                with self._db:
                    self._db.execute(
                        "INSERT INTO meta VALUES (3, ?)", (container_binding.model_dump_json(),)
                    )
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._owner.close()
            raise

    def _initialize(self, identity: ExecutionCommand, *, create: bool) -> None:
        with self._db:
            if not create:
                tables = {
                    row[0]
                    for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if not {"meta", "executions"}.issubset(tables):
                    raise ValueError("execution_journal_uninitialized")
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, value TEXT)")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS executions (key TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            expected = "1:" + identity.model_dump_json()
            row = self._db.execute("SELECT value FROM meta WHERE id=1").fetchone()
            if row is None:
                if not create:
                    raise ValueError("execution_journal_uninitialized")
                self._db.execute("INSERT INTO meta VALUES (1, ?)", (expected,))
            elif row[0] != expected:
                raise ValueError("execution_journal_incarnation_mismatch")

    def _key(self, command: ExecutionCommand) -> str:
        if self._closed:
            raise ValueError("execution_driver_closed")
        if (
            command.worker_id,
            command.worker_generation,
            command.engine_instance_id,
        ) != self.identity:
            raise ValueError("execution_runtime_incarnation_mismatch")
        canonical = command.model_copy(update={"action": ExecutionAction.QUERY}).model_dump_json()
        if len(canonical.encode()) > 8192:
            raise ValueError("execution_identity_too_large")
        # Decision identity is unique within a tenant; changing the request binding
        # must conflict with the existing record, not create a second engine call.
        return hashlib.sha256(
            json.dumps([*self.identity, command.tenant_id, command.decision_id]).encode()
        ).hexdigest()

    def submitted_key(self, command: ExecutionCommand) -> str:
        """Read an existing binding without creating an admission or changing its state."""
        key = self._key(command)
        if self.retirement is not None:
            raise ValueError("execution_incarnation_retired")
        row = self._db.execute("SELECT record FROM executions WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("execution_admission_unknown")
        record = _Record.model_validate_json(row[0])
        if record.command != command.model_copy(update={"action": ExecutionAction.QUERY}):
            raise ValueError("execution_decision_binding_mismatch")
        if not record.dispatched:
            raise ValueError("execution_not_submitted")
        return key

    def _record(self, key: str, command: ExecutionCommand) -> _Record:
        canonical = command.model_copy(update={"action": ExecutionAction.QUERY})
        row = self._db.execute("SELECT record FROM executions WHERE key=?", (key,)).fetchone()
        if row is not None:
            record = _Record.model_validate_json(row[0])
            if record.command != canonical:
                raise ValueError("execution_decision_binding_mismatch")
            return record
        count = self._db.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        if count >= self.max_records:
            raise ValueError("execution_journal_full")
        record = _Record(command=canonical)
        self._save(key, record)
        return record

    def _save(self, key: str, record: _Record) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO executions VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET record=excluded.record",
                (key, record.model_dump_json()),
            )

    def _lock(self, key: str, command: ExecutionCommand) -> asyncio.Lock:
        # Allocate a lock only after the bounded journal accepted the identity.
        self._record(key, command)
        return self._locks.setdefault(key, asyncio.Lock())

    async def admit(self, command: ExecutionCommand, payload: Mapping[str, Any]) -> str:
        """Trusted ingress only; headers alone do not authenticate these identities.

        Duplicate attempts (including retries after lost acknowledgement) are rejected.
        A caller timeout is an uncertain dispatch, never permission to reclaim capacity.
        """
        key = self._key(command)
        if self.retirement is not None:
            raise ValueError("execution_incarnation_retired")
        if command.action != ExecutionAction.QUERY:
            raise ValueError("execution_admission_requires_query_identity")
        lock = self._lock(key, command)
        self._active += 1
        try:
            async with lock:
                record = self._record(key, command)
                if record.closed:
                    raise ValueError("execution_admission_closed")
                record.dispatched = record.closed = True
                self._save(key, record)
                assert self.backend is not None
                await self.backend.submit(key, payload)
                return key
        finally:
            self._active -= 1

    async def observe(self, command: ExecutionCommand) -> ExecutionReceipt:
        key = self._key(command)
        lock = self._lock(key, command)
        self._active += 1
        try:
            async with lock:
                record = self._record(key, command)
                if self.retirement is not None:
                    record.closed = record.cancel_requested = True
                    if record.terminal is None:
                        record.terminal = (
                            ExecutionStatus.ABORTED
                            if record.dispatched
                            else ExecutionStatus.NOT_ACCEPTED
                        )
                    self._save(key, record)
                if command.action == ExecutionAction.ABORT and record.terminal is None:
                    record.closed = record.cancel_requested = True
                    if not record.dispatched:
                        record.terminal = ExecutionStatus.NOT_ACCEPTED
                    # Persist the fence even if abort delivery fails or is cancelled.
                    self._save(key, record)
                if record.terminal is not None:
                    status, quiet, fenced = record.terminal, True, True
                elif not record.dispatched:
                    status, quiet, fenced = ExecutionStatus.UNKNOWN, False, False
                else:
                    assert self.backend is not None
                    if record.cancel_requested:
                        await self.backend.abort(key)
                    observed = await self.backend.query(key)
                    now = self.clock()
                    if observed.engine_request_id != key:
                        raise ValueError("execution_backend_identity_mismatch")
                    if observed.observed_at.tzinfo is None or not (
                        0 <= (now - observed.observed_at).total_seconds() <= 5
                    ):
                        raise ValueError("execution_backend_observation_stale")
                    status, quiet, fenced = (
                        observed.status,
                        observed.quiescent,
                        observed.submission_fenced,
                    )
                    # NOT_ACCEPTED after dispatch is just a negative backend lookup.
                    if status == ExecutionStatus.NOT_ACCEPTED:
                        status, quiet, fenced = ExecutionStatus.UNKNOWN, False, False
                    if (
                        status in {ExecutionStatus.COMPLETED, ExecutionStatus.ABORTED}
                        and quiet
                        and fenced
                    ):
                        record.terminal = status
                record.sequence += 1
                self._save(key, record)
                return ExecutionReceipt(
                    command=command,
                    observation_sequence=record.sequence,
                    observed_at=self.clock(),
                    status=status,
                    quiescent=quiet,
                    admission_closed=record.closed and fenced,
                )
        finally:
            self._active -= 1

    def close(self) -> None:
        if self._active:
            raise ValueError("execution_operations_in_flight")
        if not self._closed:
            self._db.close()
            self._owner.close()
            self._closed = True
