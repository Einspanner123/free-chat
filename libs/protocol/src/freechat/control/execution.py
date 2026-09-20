"""Shared loopback execution RPC; engine ownership stays in the Worker."""

from __future__ import annotations

import hmac
import re
from typing import Any, Protocol

import grpc
from freechat_contracts.execution import ExecutionCommand, ExecutionReceipt
from pydantic import SecretStr

from freechat.control.v1 import control_pb2, control_pb2_grpc


class ExecutionDriver(Protocol):
    async def observe(self, command: ExecutionCommand) -> ExecutionReceipt:
        """ABORT must fence late admissions, not just issue a best-effort stop signal."""
        ...


class LocalRequestExecutionService(control_pb2_grpc.RequestExecutionServiceServicer):
    """Authenticated same-host transport around a supplied execution driver."""

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
