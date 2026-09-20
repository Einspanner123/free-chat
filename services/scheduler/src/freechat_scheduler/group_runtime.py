"""Fenced group commands and local-only gRPC driver boundary, not a GPU launcher."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts import WorkerCapabilities, WorkerTelemetry
from freechat_control_store import KeyValueStore
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from freechat_scheduler.resource_groups import GroupLease, GroupSpec


class GroupAction(StrEnum):
    START = "start"
    INSPECT = "inspect"
    DRAIN = "drain"
    STOP = "stop"


class RuntimeStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    DRAINED = "drained"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class GroupCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation_id: str
    action: GroupAction
    spec: GroupSpec
    node_id: str
    owner_id: str
    generation: int = Field(ge=1)
    engine_instance_id: str | None = None

    def identity(self) -> str:
        data = self.model_dump(mode="json", exclude={"operation_id"})
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @classmethod
    def for_lease(cls, lease: GroupLease, action: GroupAction) -> GroupCommand:
        command = cls(
            operation_id="",
            action=action,
            spec=lease.spec,
            node_id=lease.node_id,
            owner_id=lease.owner_id,
            generation=lease.generation,
            engine_instance_id=lease.engine_instance_id,
        )
        return command.model_copy(update={"operation_id": command.identity()})


class RankBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gpu_id: str
    available_bytes: int = Field(ge=0)


class GroupReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation_id: str
    node_id: str
    owner_id: str
    group_id: str
    generation: int = Field(ge=1)
    engine_instance_id: str = Field(min_length=1)
    observed_at: datetime
    status: RuntimeStatus
    # STOPPED asserts no process/collective belonging to this incarnation can execute.
    quiescent: bool = False
    capabilities: WorkerCapabilities | None = None
    telemetry: WorkerTelemetry | None = None
    rank_budgets: tuple[RankBudget, ...] = ()

    def validate_envelope(self, command: GroupCommand) -> None:
        if (
            command.operation_id != command.identity()
            or self.operation_id != command.operation_id
            or self.node_id != command.node_id
            or self.owner_id != command.owner_id
            or self.group_id != command.spec.group_id
            or self.generation != command.generation
            or (
                command.engine_instance_id is not None
                and self.engine_instance_id != command.engine_instance_id
            )
        ):
            raise ValueError("runtime_receipt_fence")


class GroupDriver(Protocol):
    async def apply(self, command: GroupCommand) -> GroupReceipt:
        """Idempotent side effects, fresh observation; UNKNOWN must not imply stopped."""
        ...


class LocalRuntimeEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    address: str = Field(pattern=r"^127\.0\.0\.1:[1-9][0-9]{0,4}$")
    token: SecretStr = Field(min_length=32)


class LocalGrpcGroupDriver:
    """Explicit CPU-test transport. Remote hosts require a future mTLS implementation."""

    def __init__(self, endpoints: dict[str, LocalRuntimeEndpoint]) -> None:
        self._endpoints = endpoints

    async def apply(self, command: GroupCommand) -> GroupReceipt:
        endpoint = self._endpoints.get(command.node_id)
        if endpoint is None:
            raise ValueError("runtime_endpoint_missing")
        async with grpc.aio.insecure_channel(endpoint.address) as channel:
            stub = control_pb2_grpc.GroupRuntimeServiceStub(channel)  # type: ignore[no-untyped-call]
            result = await stub.Apply(
                control_pb2.GroupRuntimeCommand(command_json=command.model_dump_json()),
                metadata=(("authorization", f"Bearer {endpoint.token.get_secret_value()}"),),
                timeout=2.0,
            )
        receipt = GroupReceipt.model_validate_json(result.receipt_json)
        receipt.validate_envelope(command)
        return receipt


class LocalGroupRuntimeService(control_pb2_grpc.GroupRuntimeServiceServicer):
    """Only install on a loopback test server with an explicitly provided driver."""

    def __init__(
        self,
        driver: GroupDriver,
        *,
        node_id: str,
        owner_id: str,
        token: SecretStr,
        store: KeyValueStore,
    ) -> None:
        if len(token.get_secret_value()) < 32:
            raise ValueError("runtime_token_too_short")
        self.driver, self.node_id, self.owner_id, self.token = driver, node_id, owner_id, token
        self.store = store
        self._lock = asyncio.Lock()
        self._key = "/freechat/local-runtime/" + hashlib.sha256(node_id.encode()).hexdigest()

    async def _apply_fenced(self, command: GroupCommand) -> GroupReceipt:
        # One node supervisor owns this namespace; drivers must make side effects
        # idempotent by incarnation, including across supervisor restarts.
        async with self._lock:
            record = await self.store.get(self._key)
            state = (
                RuntimeFenceState()
                if record is None
                else RuntimeFenceState.model_validate_json(record.value)
            )
            old = state.groups.get(command.spec.group_id)
            phase = {
                GroupAction.START: 0,
                GroupAction.INSPECT: 1,
                GroupAction.DRAIN: 2,
                GroupAction.STOP: 3,
            }[command.action]
            if old is not None:
                if command.generation < old.generation:
                    raise ValueError("runtime_generation_regressed")
                if command.generation == old.generation:
                    if (
                        command.spec != old.spec
                        or command.owner_id != old.owner_id
                        or phase < old.phase
                        or (old.stopped and command.action is not GroupAction.STOP)
                        or (
                            command.engine_instance_id is not None
                            and old.engine_instance_id is not None
                            and command.engine_instance_id != old.engine_instance_id
                        )
                    ):
                        raise ValueError("runtime_incarnation_or_phase_conflict")
                elif not old.stopped:
                    raise ValueError("runtime_previous_incarnation_not_stopped")
            if old is None or command.generation > old.generation:
                if (
                    command.action is not GroupAction.START
                    or command.engine_instance_id is not None
                ):
                    raise ValueError("runtime_incarnation_unknown")
                if any(
                    not entry.stopped and set(entry.spec.gpu_ids) & set(command.spec.gpu_ids)
                    for entry in state.groups.values()
                ):
                    raise ValueError("runtime_gpus_busy")
                old = RuntimeFence(
                    spec=command.spec, generation=command.generation, owner_id=command.owner_id
                )
            current = old.model_copy(update={"phase": phase})
            state.groups[command.spec.group_id] = current
            item = await self.store.compare_and_put(
                self._key,
                0 if record is None else record.revision,
                state.model_dump_json().encode(),
            )
            receipt = await self.driver.apply(command)
            receipt.validate_envelope(command)
            if (
                receipt.observed_at.tzinfo is None
                or not 0 <= (datetime.now(UTC) - receipt.observed_at).total_seconds() <= 5
            ):
                raise ValueError("runtime_observation_stale")
            if (
                current.engine_instance_id is not None
                and receipt.engine_instance_id != current.engine_instance_id
            ):
                raise ValueError("runtime_observed_instance_changed")
            state.groups[command.spec.group_id] = current.model_copy(
                update={
                    "engine_instance_id": receipt.engine_instance_id,
                    "stopped": current.stopped
                    or (
                        command.action is GroupAction.STOP
                        and receipt.status is RuntimeStatus.STOPPED
                        and receipt.quiescent
                    ),
                }
            )
            await self.store.compare_and_put(
                self._key, item.revision, state.model_dump_json().encode()
            )
            return receipt

    async def Apply(
        self, request: control_pb2.GroupRuntimeCommand, context: Any
    ) -> control_pb2.GroupRuntimeReceipt:
        if re.match(r"^ipv4:127\.0\.0\.1:\d+$", context.peer()) is None:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "local_runtime_only")
        provided = [v for k, v in context.invocation_metadata() if k == "authorization"]
        expected = f"Bearer {self.token.get_secret_value()}"
        if len(provided) != 1 or not hmac.compare_digest(provided[0], expected):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "runtime_identity_required")
        try:
            command = GroupCommand.model_validate_json(request.command_json)
            if (
                command.node_id != self.node_id
                or command.owner_id != self.owner_id
                or command.operation_id != command.identity()
            ):
                raise ValueError("runtime_command_fence")
            receipt = await self._apply_fenced(command)
        except ValueError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            raise AssertionError("abort must terminate RPC") from error
        return control_pb2.GroupRuntimeReceipt(receipt_json=receipt.model_dump_json())


class RuntimeFence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spec: GroupSpec
    generation: int
    owner_id: str
    phase: int = 0
    engine_instance_id: str | None = None
    stopped: bool = False


class RuntimeFenceState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    groups: dict[str, RuntimeFence] = Field(default_factory=dict)
