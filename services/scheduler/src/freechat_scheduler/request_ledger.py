"""Atomic request reservations and outbox intents; expiry is not engine quiescence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from freechat_contracts import RequestProfile, RouteDecision
from freechat_contracts.execution import ExecutionReceipt
from freechat_control_store import CompareFailed, InMemoryStore, KeyValueStore
from freechat_trace_replay import EventEnvelope
from freechat_trace_replay.bus import lifecycle_subject
from pydantic import BaseModel, ConfigDict, Field


class RequestState(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETION_PENDING = "completion_pending"
    RELEASED = "released"


class Reservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tenant_id: str
    harness_id: str
    identity_scope: dict[str, str | None]
    identity: str
    fingerprint: str
    decision: RouteDecision
    state: RequestState = RequestState.ACTIVE
    expires_at: datetime
    renewals: tuple[str, ...] = ()
    sequence: int = 0
    execution_receipt: ExecutionReceipt | None = None


class RequestLedgerState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = 1
    reservations: dict[str, Reservation] = Field(default_factory=dict)
    identities: dict[str, str] = Field(default_factory=dict)
    pending: dict[str, EventEnvelope] = Field(default_factory=dict)


Planner = Callable[[dict[str, tuple[int, int]]], RouteDecision]


def _hash(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


class RequestLedger:
    def __init__(
        self,
        store: KeyValueStore | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        key: str = "/freechat/request-ledger",
        max_records: int = 10_000,
        max_pending: int = 20_000,
        max_renewals: int = 4096,
        max_bytes: int = 1_048_576,
    ) -> None:
        if min(max_records, max_pending, max_renewals, max_bytes) < 1:
            raise ValueError("invalid_ledger_limits")
        self.store = store if store is not None else InMemoryStore()
        self.clock, self.key = clock, key
        self.max_bytes = max_bytes
        self.max_records, self.max_pending, self.max_renewals = (
            max_records,
            max_pending,
            max_renewals,
        )

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock_requires_timezone")
        return now

    async def _read(self) -> tuple[int, RequestLedgerState]:
        item = await self.store.get(self.key)
        if item is None:
            if await self.store.list_prefix("/freechat/leases/"):
                raise ValueError("unreconciled_decision_records")
            return 0, RequestLedgerState()
        state = RequestLedgerState.model_validate_json(item.value)
        if state.schema_version != 1:
            raise ValueError("unsupported_request_ledger_schema")
        return item.revision, state

    async def snapshot(self) -> RequestLedgerState:
        return (await self._read())[1].model_copy(deep=True)

    async def _mutate(self, update: Callable[[RequestLedgerState], None]) -> RequestLedgerState:
        for _ in range(32):
            revision, state = await self._read()
            update(state)
            if len(state.pending) > self.max_pending:
                raise ValueError("request_outbox_full")
            payload = state.model_dump_json().encode()
            if len(payload) > self.max_bytes:
                raise ValueError("request_ledger_bytes_exceeded")
            try:
                await self.store.compare_and_put(self.key, revision, payload)
                return state
            except CompareFailed:
                continue
        raise RuntimeError("request_ledger_contention_retry_exhausted")

    def _event(self, state: RequestLedgerState, lease: Reservation, kind: str) -> None:
        event = EventEnvelope(
            event_id=_hash(lease.tenant_id, lease.decision.decision_id, str(lease.sequence), kind),
            event_type=kind,
            tenant_id=lease.tenant_id,
            aggregate_id=lease.decision.decision_id,
            aggregate_generation=lease.decision.worker_generation,
            occurred_at=self._now(),
            payload={
                "harness_id": lease.harness_id,
                "identity_scope": lease.identity_scope,
                "decision_id": lease.decision.decision_id,
                "request_id": lease.decision.request_id,
                "worker_id": lease.decision.worker_id,
                "state": lease.state,
                "sequence": lease.sequence,
                "expires_at": lease.expires_at.isoformat(),
                "reserved_kv_bytes_per_rank": lease.decision.reserved_kv_bytes_per_rank,
                "execution_receipt": None
                if lease.execution_receipt is None
                else lease.execution_receipt.model_dump(mode="json"),
                "decision": lease.decision.model_dump(mode="json")
                if kind == "route.decided"
                else None,
            },
        )
        state.reservations[lease.decision.decision_id] = lease
        state.pending[event.event_id] = event

    @staticmethod
    def _load(state: RequestLedgerState) -> dict[str, tuple[int, int]]:
        load: dict[str, tuple[int, int]] = {}
        for lease in state.reservations.values():
            if lease.state is not RequestState.RELEASED:
                worker = lease.decision.worker_id
                size, count = load.get(worker, (0, 0))
                load[worker] = size + lease.decision.reserved_kv_bytes_per_rank, count + 1
        return load

    async def reserve(
        self, request: RequestProfile, idempotency_key: str, planner: Planner
    ) -> RouteDecision:
        if not request.tenant_id.strip() or not idempotency_key.strip():
            raise ValueError("request_identity_required")
        lifecycle_subject(request.tenant_id, request.hints.harness_id, "route.decided")
        identity = _hash(request.tenant_id, idempotency_key)
        fingerprint = _hash(json.dumps(request.model_dump(mode="json"), sort_keys=True))

        def update(state: RequestLedgerState) -> None:
            existing = state.identities.get(identity)
            if existing is not None:
                lease = state.reservations[existing]
                if lease.fingerprint != fingerprint:
                    raise ValueError("request_idempotency_conflict")
                if lease.state is not RequestState.ACTIVE or lease.expires_at <= self._now():
                    raise ValueError("request_not_active")
                return
            if len(state.reservations) >= self.max_records:
                raise ValueError("request_ledger_full")
            decision = planner(self._load(state))
            if (
                decision.request_id != request.request_id
                or decision.reserved_kv_bytes_per_rank <= 0
            ):
                raise ValueError("invalid_reservation_decision")
            if decision.decision_id in state.reservations:
                raise ValueError("decision_identity_collision")
            lease = Reservation(
                tenant_id=request.tenant_id,
                harness_id=request.hints.harness_id,
                identity_scope={
                    name: getattr(request.hints, name)
                    for name in (
                        "task_id",
                        "session_id",
                        "agent_id",
                        "branch_id",
                        "turn_id",
                        "call_id",
                    )
                },
                identity=identity,
                fingerprint=fingerprint,
                decision=decision,
                expires_at=self._now() + timedelta(milliseconds=decision.lease_ttl_ms),
            )
            state.identities[identity] = decision.decision_id
            self._event(state, lease, "route.decided")

        state = await self._mutate(update)
        return state.reservations[state.identities[identity]].decision

    @staticmethod
    def _require(
        state: RequestLedgerState,
        decision_id: str,
        worker_id: str,
        generation: int,
        tenant_id: str,
    ) -> Reservation:
        lease = state.reservations.get(decision_id)
        if lease is None or lease.tenant_id != tenant_id:
            raise KeyError("decision_not_found")
        if lease.decision.worker_id != worker_id or lease.decision.worker_generation != generation:
            raise ValueError("lease_fencing_mismatch")
        return lease

    async def require(
        self,
        decision_id: str,
        worker_id: str,
        generation: int,
        tenant_id: str,
    ) -> RouteDecision:
        return self._require(
            (await self._read())[1], decision_id, worker_id, generation, tenant_id
        ).decision

    async def renew(
        self,
        decision_id: str,
        worker_id: str,
        generation: int,
        tenant_id: str,
        operation_id: str,
    ) -> None:
        if not operation_id.strip():
            raise ValueError("renewal_identity_required")

        def update(state: RequestLedgerState) -> None:
            lease = self._require(state, decision_id, worker_id, generation, tenant_id)
            if lease.state is not RequestState.ACTIVE or lease.expires_at <= self._now():
                raise ValueError("lease_not_renewable")
            if operation_id in lease.renewals:
                return
            if len(lease.renewals) >= self.max_renewals:
                raise ValueError("renewal_history_full")
            lease = lease.model_copy(
                update={
                    "expires_at": max(
                        lease.expires_at,
                        self._now() + timedelta(milliseconds=lease.decision.lease_ttl_ms),
                    ),
                    "renewals": (*lease.renewals, operation_id),
                    "sequence": lease.sequence + 1,
                }
            )
            self._event(state, lease, "lease.renewed")

        await self._mutate(update)

    async def release(
        self,
        decision_id: str,
        worker_id: str,
        generation: int,
        tenant_id: str,
        *,
        cancelled: bool = False,
    ) -> RouteDecision:
        """Gateway transport intent only. Both completion and cancellation retain capacity."""

        def update(state: RequestLedgerState) -> None:
            lease = self._require(state, decision_id, worker_id, generation, tenant_id)
            target = RequestState.CANCEL_REQUESTED if cancelled else RequestState.COMPLETION_PENDING
            if lease.state in {target, RequestState.RELEASED}:
                return
            if not cancelled and lease.state in {
                RequestState.CANCEL_REQUESTED,
                RequestState.EXPIRED,
            }:
                return  # A late HTTP response must not undo an abort intent.
            lease = lease.model_copy(update={"state": target, "sequence": lease.sequence + 1})
            self._event(
                state, lease, "lease.cancel_requested" if cancelled else "lease.completion_pending"
            )

        return (await self._mutate(update)).reservations[decision_id].decision

    async def observe_execution(self, receipt: ExecutionReceipt) -> None:
        """Only the authenticated runtime client may call this, never an HTTP hint."""

        def update(state: RequestLedgerState) -> None:
            command = receipt.command
            lease = self._require(
                state,
                command.decision_id,
                command.worker_id,
                command.worker_generation,
                command.tenant_id,
            )
            if (
                lease.decision.request_id != command.request_id
                or lease.decision.engine_instance_id != command.engine_instance_id
            ):
                raise ValueError("execution_incarnation_fence")
            previous = lease.execution_receipt
            if previous is not None:
                if receipt.observation_sequence < previous.observation_sequence:
                    raise ValueError("execution_observation_regressed")
                if receipt.observation_sequence == previous.observation_sequence:
                    if receipt != previous:
                        raise ValueError("execution_observation_conflict")
                    return
            if lease.state is RequestState.RELEASED:
                raise ValueError("released_execution_is_terminal")
            if (
                receipt.observed_at.tzinfo is None
                or not 0 <= (self._now() - receipt.observed_at).total_seconds() <= 5
            ):
                raise ValueError("execution_observation_stale")
            # A negative lookup or a completed HTTP stream cannot rule out late admission.
            released = receipt.releasable
            changed = lease.model_copy(
                update={
                    "execution_receipt": receipt,
                    "state": RequestState.RELEASED if released else lease.state,
                }
            )
            if released or previous is None or previous.status != receipt.status:
                changed = changed.model_copy(update={"sequence": lease.sequence + 1})
                self._event(state, changed, "lease.released" if released else "execution.observed")
            else:
                state.reservations[command.decision_id] = changed

        await self._mutate(update)

    async def expire(self) -> None:
        def update(state: RequestLedgerState) -> None:
            for lease in tuple(state.reservations.values()):
                if (
                    lease.state in {RequestState.ACTIVE, RequestState.COMPLETION_PENDING}
                    and lease.expires_at <= self._now()
                ):
                    self._event(
                        state,
                        lease.model_copy(
                            update={
                                "state": RequestState.EXPIRED,
                                "sequence": lease.sequence + 1,
                            }
                        ),
                        "lease.expired",
                    )

        await self._mutate(update)

    async def acknowledge_event(self, event_id: str) -> None:
        def update(state: RequestLedgerState) -> None:
            state.pending.pop(event_id, None)

        await self._mutate(update)
