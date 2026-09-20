"""Read native token budgets only from eligible, registered worker endpoints."""

from __future__ import annotations

import asyncio
import json

import httpx
from freechat_contracts import RequestProfile
from freechat_contracts.preparation import PreparedAdmission

from freechat_scheduler.registry import WorkerSnapshot


class NativePreparer:
    def __init__(
        self,
        token: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if len(token) < 32:
            raise ValueError("worker token requires at least 32 characters")
        self._token = token
        self._transport = transport
        self._slots = asyncio.Semaphore(8)

    async def prepare(
        self,
        request: RequestProfile,
        workers: tuple[WorkerSnapshot, ...],
    ) -> dict[str, PreparedAdmission]:
        if request.native_request_json is None or request.native_protocol is None:
            raise ValueError("native_request_required_for_measured_admission")
        body = json.loads(request.native_request_json)
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        ) as client:

            async def one(worker: WorkerSnapshot) -> PreparedAdmission | None:
                async with self._slots:
                    try:
                        response = await client.post(
                            worker.capabilities.endpoint.rstrip("/") + "/freechat/prepare",
                            headers={
                                "x-freechat-worker-token": self._token,
                                "x-freechat-internal-tenant": request.tenant_id,
                            },
                            json={"protocol": request.native_protocol, "request": body},
                        )
                        response.raise_for_status()
                        prepared = PreparedAdmission.model_validate_json(response.content)
                        if (
                            prepared.worker_id != worker.capabilities.worker_id
                            or prepared.generation != worker.capabilities.generation
                            or prepared.engine_instance_id != worker.telemetry.engine_instance_id
                            or prepared.budget.tenant_id != request.tenant_id
                            or prepared.budget.protocol != request.native_protocol
                            or prepared.budget.body_sha256 != request.native_body_sha256
                        ):
                            return None
                        return prepared
                    except (httpx.HTTPError, ValueError):
                        # Worker errors may echo private prompt content; do not log them.
                        return None

            results = await asyncio.gather(*(one(worker) for worker in workers))
        return {item.worker_id: item for item in results if item is not None}
