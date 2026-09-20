"""Pull-based request reconciliation; loopback transport only, no inference engine driver."""

from __future__ import annotations

import asyncio
import hmac
import re
from typing import Any, Literal, Protocol

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionReceipt
from pydantic import BaseModel, ConfigDict, SecretStr

from freechat_scheduler.group_runtime import LocalRuntimeEndpoint
from freechat_scheduler.request_ledger import RequestLedger, RequestState, Reservation


class ExecutionDriver(Protocol):
    async def observe(self, command: ExecutionCommand) -> ExecutionReceipt:
        """ABORT must fence late admissions, not just issue a best-effort stop signal."""
        ...


class RequestExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["local-contract"]
    endpoints: dict[str, LocalRuntimeEndpoint]


class LocalGrpcExecutionDriver:
    def __init__(self, config: RequestExecutionConfig) -> None:
        self.config = config

    async def observe(self, command: ExecutionCommand) -> ExecutionReceipt:
        endpoint = self.config.endpoints.get(command.worker_id)
        if endpoint is None:
            raise ValueError("execution_endpoint_missing")
        async with grpc.aio.insecure_channel(endpoint.address) as channel:
            stub = control_pb2_grpc.RequestExecutionServiceStub(channel)  # type: ignore[no-untyped-call]
            result = await stub.Observe(
                control_pb2.RequestExecutionCommand(command_json=command.model_dump_json()),
                metadata=(("authorization", f"Bearer {endpoint.token.get_secret_value()}"),),
                timeout=2,
            )
        receipt = ExecutionReceipt.model_validate_json(result.receipt_json)
        if receipt.command != command:
            raise ValueError("execution_response_command_mismatch")
        return receipt


class LocalRequestExecutionService(control_pb2_grpc.RequestExecutionServiceServicer):
    """Explicit local test service; no default engine implementation or admission fence."""

    def __init__(
        self,
        driver: ExecutionDriver,
        *,
        worker_id: str,
        generation: int,
        engine_instance_id: str,
        token: SecretStr,
    ) -> None:
        if (
            not worker_id.strip()
            or generation < 1
            or not engine_instance_id.strip()
            or len(token.get_secret_value()) < 32
        ):
            raise ValueError("execution_runtime_identity_required")
        self.driver, self.token = driver, token
        self.identity = (worker_id, generation, engine_instance_id)

    async def Observe(
        self, request: control_pb2.RequestExecutionCommand, context: Any
    ) -> control_pb2.RequestExecutionReceipt:
        if re.match(r"^ipv4:127\.0\.0\.1:\d+$", context.peer()) is None:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "local_execution_only")
        values = [v for k, v in context.invocation_metadata() if k == "authorization"]
        if len(values) != 1 or not hmac.compare_digest(
            values[0], f"Bearer {self.token.get_secret_value()}"
        ):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "execution_identity_required")
        try:
            command = ExecutionCommand.model_validate_json(request.command_json)
            if (
                command.worker_id,
                command.worker_generation,
                command.engine_instance_id,
            ) != self.identity:
                raise ValueError("execution_runtime_incarnation_mismatch")
            receipt = await self.driver.observe(command)
            if receipt.command != command:
                raise ValueError("execution_response_command_mismatch")
        except ValueError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            raise AssertionError("abort must terminate RPC") from error
        return control_pb2.RequestExecutionReceipt(receipt_json=receipt.model_dump_json())


class RequestExecutionReconciler:
    def __init__(self, ledger: RequestLedger, driver: ExecutionDriver) -> None:
        self.ledger, self.driver = ledger, driver
        self._lock = asyncio.Lock()

    async def tick(self) -> dict[str, str]:
        async with self._lock:
            errors: dict[str, str] = {}
            slots = asyncio.Semaphore(8)

            async def observe(lease: Reservation) -> None:
                async with slots:
                    try:
                        decision = lease.decision
                        if not decision.engine_instance_id:
                            raise ValueError("execution_incarnation_unbound")
                        command = ExecutionCommand(
                            tenant_id=lease.tenant_id,
                            request_id=decision.request_id,
                            decision_id=decision.decision_id,
                            worker_id=decision.worker_id,
                            worker_generation=decision.worker_generation,
                            engine_instance_id=decision.engine_instance_id,
                            action=ExecutionAction.ABORT
                            if lease.state in {RequestState.EXPIRED, RequestState.CANCEL_REQUESTED}
                            else ExecutionAction.QUERY,
                        )
                        async with asyncio.timeout(2):
                            receipt = await self.driver.observe(command)
                            if receipt.command != command:
                                raise ValueError("execution_response_command_mismatch")
                            await self.ledger.observe_execution(receipt)
                    except Exception as error:
                        errors[lease.decision.decision_id] = type(error).__name__

            await asyncio.gather(
                *(
                    observe(lease)
                    for lease in (await self.ledger.snapshot()).reservations.values()
                    if lease.state is not RequestState.RELEASED
                )
            )
            return errors
