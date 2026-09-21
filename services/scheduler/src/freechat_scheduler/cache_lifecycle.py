"""Resolve cache updates from an authenticated tenant's original route."""

from __future__ import annotations

import logging

import grpc
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts.cache_lifecycle import (
    CacheLifecycleCommand,
    CacheLifecycleReceipt,
    CacheLifecycleUpdate,
)
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionStatus
from pydantic import SecretStr

from freechat_scheduler.group_runtime import LocalRuntimeEndpoint
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_ledger import RequestLedger, RequestState

LOGGER = logging.getLogger(__name__)
WORKER_CACHE_RPC_TIMEOUT_SECONDS = 5


class CacheLifecycleController:
    """Apply cache intent to the original owner without modifying request reservations.

    Transport is synchronous and caller-retried; logs are not a durable command outbox.
    A replacement Worker must never inherit cache commands for an old incarnation.
    """

    def __init__(
        self,
        ledger: RequestLedger,
        registry: InMemoryWorkerRegistry,
        token: str,
    ) -> None:
        if len(token) < 32:
            raise ValueError("cache_control_token_required")
        self.ledger = ledger
        self.registry = registry
        self.token = SecretStr(token)

    async def apply(self, tenant_id: str, update: CacheLifecycleUpdate) -> CacheLifecycleReceipt:
        endpoint, command = await self._resolve_owner(tenant_id, update)
        LOGGER.info("cache_lifecycle_requested %s", command.model_dump_json())
        async with grpc.aio.insecure_channel(endpoint.address) as channel:
            stub = control_pb2_grpc.CacheLifecycleServiceStub(channel)  # type: ignore[no-untyped-call]
            reply = await stub.Apply(
                control_pb2.CacheLifecycleCommand(command_json=command.model_dump_json()),
                metadata=(("authorization", f"Bearer {self.token.get_secret_value()}"),),
                timeout=WORKER_CACHE_RPC_TIMEOUT_SECONDS,
            )
        receipt = CacheLifecycleReceipt.model_validate_json(reply.receipt_json)
        if receipt.command != command:
            raise ValueError("cache_response_command_mismatch")
        LOGGER.info("cache_lifecycle_applied %s", receipt.model_dump_json())
        return receipt

    async def _resolve_owner(
        self, tenant_id: str, update: CacheLifecycleUpdate
    ) -> tuple[LocalRuntimeEndpoint, CacheLifecycleCommand]:
        """Read original route and current incarnation; do not re-plan or release it."""
        lease = (await self.ledger.snapshot()).reservations.get(update.decision_id)
        if lease is None or lease.tenant_id != tenant_id:
            raise KeyError("cache_route_not_found")
        decision = lease.decision
        if not decision.engine_instance_id or decision.preparation is None:
            raise ValueError("cache_control_requires_managed_route")
        if update.lifecycle in {"tool_wait", "resume"} and (
            lease.state in {RequestState.CANCEL_REQUESTED, RequestState.EXPIRED}
            or (
                lease.execution_receipt is not None
                and lease.execution_receipt.status
                in {
                    ExecutionStatus.ABORTED,
                    ExecutionStatus.NOT_ACCEPTED,
                }
            )
        ):
            raise ValueError("cancelled_route_cannot_retain")
        worker = next(
            (
                w
                for w in self.registry.snapshot()[1]
                if w.capabilities.worker_id == decision.worker_id
            ),
            None,
        )
        if (
            worker is None
            or worker.capabilities.generation != decision.worker_generation
            or worker.telemetry.engine_instance_id != decision.engine_instance_id
            or worker.capabilities.execution_endpoint is None
        ):
            raise ValueError("cache_execution_owner_unavailable")
        endpoint = LocalRuntimeEndpoint(
            address=worker.capabilities.execution_endpoint,
            token=self.token,
        )
        command = CacheLifecycleCommand(
            owner=ExecutionCommand(
                tenant_id=tenant_id,
                request_id=decision.request_id,
                decision_id=decision.decision_id,
                worker_id=decision.worker_id,
                worker_generation=decision.worker_generation,
                engine_instance_id=decision.engine_instance_id,
                action=ExecutionAction.QUERY,
            ),
            cache_generation=decision.worker_generation,
            update=update,
        )
        return endpoint, command
