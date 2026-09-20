"""Pull-based execution reconciliation against generation-bound loopback Workers."""

from __future__ import annotations

import asyncio
from typing import Literal

import grpc
from freechat.control.execution import (
    ExecutionDriver as ExecutionDriver,
)
from freechat.control.execution import (
    LocalRequestExecutionService as LocalRequestExecutionService,
)
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionReceipt
from pydantic import BaseModel, ConfigDict, SecretStr

from freechat_scheduler.group_runtime import LocalRuntimeEndpoint
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_ledger import RequestLedger, RequestState, Reservation

ObservationUnavailableReason = Literal[
    "worker_not_registered",
    "worker_generation_changed",
    "engine_instance_changed",
    "execution_endpoint_missing",
]


class ExecutionObservationUnavailable(ValueError):
    """Safe reason code for a missing execution owner, not proof of termination."""

    def __init__(self, reason: ObservationUnavailableReason) -> None:
        self.reason = reason
        super().__init__(reason)


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
            raise ExecutionObservationUnavailable("execution_endpoint_missing")
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
                    except ExecutionObservationUnavailable as error:
                        errors[lease.decision.decision_id] = error.reason
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


class RegisteredExecutionDriver:
    """Use only a currently registered same-host endpoint; never retarget old work."""

    def __init__(self, registry: InMemoryWorkerRegistry, token: str) -> None:
        if len(token) < 32:
            raise ValueError("worker token requires at least 32 characters")
        self.registry, self.token = registry, SecretStr(token)

    async def observe(self, command: ExecutionCommand) -> ExecutionReceipt:
        worker = next(
            (
                item
                for item in self.registry.snapshot()[1]
                if item.capabilities.worker_id == command.worker_id
            ),
            None,
        )
        if worker is None:
            raise ExecutionObservationUnavailable("worker_not_registered")
        if worker.capabilities.generation != command.worker_generation:
            raise ExecutionObservationUnavailable("worker_generation_changed")
        if worker.telemetry.engine_instance_id != command.engine_instance_id:
            raise ExecutionObservationUnavailable("engine_instance_changed")
        if worker.capabilities.execution_endpoint is None:
            raise ExecutionObservationUnavailable("execution_endpoint_missing")
        endpoint = LocalRuntimeEndpoint(
            address=worker.capabilities.execution_endpoint,
            token=self.token,
        )
        driver = LocalGrpcExecutionDriver(
            RequestExecutionConfig(
                mode="local-contract",
                endpoints={command.worker_id: endpoint},
            )
        )
        return await driver.observe(command)
