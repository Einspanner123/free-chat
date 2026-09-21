"""Authenticated same-host cache control; never transfers KV tensors."""

from __future__ import annotations

import re
from typing import Any, Protocol

import grpc
from freechat_contracts.cache_lifecycle import (
    MAX_CACHE_CONTROL_BYTES,
    CacheLifecycleCommand,
    CacheLifecycleReceipt,
)
from pydantic import SecretStr

from freechat.control.auth import authenticate_control
from freechat.control.v1 import control_pb2, control_pb2_grpc


class CacheLifecycleDriver(Protocol):
    async def apply(self, command: CacheLifecycleCommand) -> CacheLifecycleReceipt: ...


class LocalCacheLifecycleService(control_pb2_grpc.CacheLifecycleServiceServicer):
    def __init__(
        self,
        driver: CacheLifecycleDriver,
        *,
        identity: tuple[str, int, str],
        token: SecretStr,
    ) -> None:
        if (
            not identity[0]
            or identity[1] < 1
            or not identity[2]
            or len(token.get_secret_value()) < 32
        ):
            raise ValueError("cache_control_identity_required")
        self.driver = driver
        self.identity = identity
        self.token = token

    async def Apply(
        self,
        request: control_pb2.CacheLifecycleCommand,
        context: Any,
    ) -> control_pb2.CacheLifecycleReceipt:
        if re.match(r"^ipv4:127\.0\.0\.1:\d+$", context.peer()) is None:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "local_cache_control_only")
        await authenticate_control(context, self.token.get_secret_value())
        try:
            if len(request.command_json.encode()) > MAX_CACHE_CONTROL_BYTES:
                raise ValueError("cache_command_too_large")
            command = CacheLifecycleCommand.model_validate_json(request.command_json)
            owner = command.owner
            if (
                owner.worker_id,
                owner.worker_generation,
                owner.engine_instance_id,
            ) != self.identity:
                raise ValueError("cache_runtime_incarnation_mismatch")
            receipt = await self.driver.apply(command)
            if receipt.command != command:
                raise ValueError("cache_response_command_mismatch")
        except ValueError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            raise AssertionError("abort must terminate RPC") from error
        return control_pb2.CacheLifecycleReceipt(receipt_json=receipt.model_dump_json())
