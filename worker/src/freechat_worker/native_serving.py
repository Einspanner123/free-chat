"""Admission and execution tracking around vLLM's unchanged HTTP protocol handlers."""

from __future__ import annotations

import hmac
import json
import time
from collections.abc import AsyncGenerator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionStatus

from freechat_worker.capacity import EngineCapacity
from freechat_worker.execution import DurableExecutionDriver, EngineObservation
from freechat_worker.preparation import PreparationService, TokenBudget
from freechat_worker.vllm_execution import VllmExecutionBackend

CURRENT_ROUTE: ContextVar[str | None] = ContextVar("freechat_execution_route", default=None)
INFERENCE_PATHS = frozenset({"/v1/chat/completions", "/v1/responses", "/v1/messages"})


@dataclass
class _Route:
    children: list[str] = field(default_factory=list)
    closed: bool = False
    aborted: bool = False
    budget: TokenBudget | None = None


class NativeExecutionBackend:
    """Aggregate every native generate call under one durable route identity."""

    def __init__(self, engine: Any) -> None:
        self.children = VllmExecutionBackend(engine)
        self.routes: dict[str, _Route] = {}

    async def submit(self, engine_request_id: str, payload: Mapping[str, Any]) -> None:
        if (payload and set(payload) != {"budget"}) or engine_request_id in self.routes:
            raise ValueError("native_route_already_admitted_or_invalid")
        self.routes[engine_request_id] = _Route(
            budget=TokenBudget.model_validate(payload["budget"]) if payload else None
        )

    async def generate(
        self, prompt: Any, sampling_params: Any, request_id: str, **options: Any
    ) -> AsyncGenerator[Any, None]:
        route_id = CURRENT_ROUTE.get()
        if route_id is None or route_id not in self.routes:
            raise ValueError("engine_call_without_admission")
        route = self.routes[route_id]
        if route.closed:
            raise ValueError("native_route_submission_closed")
        if route.budget is not None:
            if route.children:
                raise ValueError("additional_engine_call_requires_new_budget")
            route.budget.check_execution(prompt, sampling_params, time.time())
        # Bound native multi-step expansion; this is not n>1 sample fan-out.
        if len(route.children) >= 128:
            raise ValueError("native_route_child_limit")
        child_id = f"{route_id}:{len(route.children)}"
        route.children.append(child_id)
        await self.children.submit(
            child_id,
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "options": options,
            },
        )
        stream = self.children.stream(child_id)
        try:
            async for output in stream:
                # Protocol handlers retain their original external request identity.
                output.request_id = request_id
                yield output
        finally:
            await stream.aclose()

    def close_submission(self, engine_request_id: str) -> None:
        self.routes[engine_request_id].closed = True

    async def abort(self, engine_request_id: str) -> None:
        route = self.routes.get(engine_request_id)
        if route is None:
            raise ValueError("native_route_unknown")
        route.closed = route.aborted = True
        for child in route.children:
            await self.children.abort(child)

    async def query(self, engine_request_id: str) -> EngineObservation:
        route = self.routes.get(engine_request_id)
        status, quiet, fenced = ExecutionStatus.UNKNOWN, False, False
        if route is not None:
            children = [await self.children.query(child) for child in route.children]
            fenced = route.closed and all(child.submission_fenced for child in children)
            quiet = fenced and all(child.quiescent for child in children)
            if quiet:
                status = (
                    ExecutionStatus.ABORTED
                    if route.aborted
                    or any(child.status == ExecutionStatus.ABORTED for child in children)
                    else ExecutionStatus.COMPLETED
                )
            else:
                status = ExecutionStatus.RUNNING
        return EngineObservation(
            engine_request_id=engine_request_id,
            observed_at=datetime.now(UTC),
            status=status,
            quiescent=quiet,
            submission_fenced=fenced,
        )


class NativeEngineClient:
    """Delegate native rendering/protocol features; fence all generation calls."""

    def __init__(self, engine: Any, backend: NativeExecutionBackend) -> None:
        self.engine, self.backend = engine, backend

    def __getattr__(self, name: str) -> Any:
        if name in {"add_request", "encode"}:
            raise AttributeError("untracked inference entry point disabled")
        return getattr(self.engine, name)

    def generate(
        self, prompt: Any, sampling_params: Any, request_id: str, **options: Any
    ) -> AsyncGenerator[Any, None]:
        return self.backend.generate(prompt, sampling_params, request_id, **options)

    async def abort(self, request_id: Any, **kwargs: Any) -> None:
        del request_id, kwargs
        route_id = CURRENT_ROUTE.get()
        if route_id is None:
            raise ValueError("abort_without_route_identity")
        await self.backend.abort(route_id)


class AdmissionMiddleware:
    """ASGI boundary spans the full native streaming response, not just headers."""

    def __init__(
        self,
        app: Any,
        *,
        driver: DurableExecutionDriver,
        backend: NativeExecutionBackend,
        token: str,
        capacity: EngineCapacity | None = None,
        preparer: PreparationService | None = None,
    ) -> None:
        if len(token) < 32:
            raise ValueError("worker_token_requires_32_characters")
        self.app, self.driver, self.backend, self.token = app, driver, backend, token
        self.capacity = capacity
        self.preparer = preparer

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path == "/health" and scope["method"] == "GET":
            await self.app(scope, receive, send)
            return
        tokens = [v for k, v in scope["headers"] if k.lower() == b"x-freechat-worker-token"]
        if len(tokens) != 1 or not hmac.compare_digest(tokens[0], self.token.encode()):
            await self._error(send, 401, "worker authentication required")
            return
        if scope["method"] == "GET" and path == "/freechat/runtime":
            worker, generation, engine = self.driver.identity
            await self._reply(
                send,
                200,
                {
                    "worker_id": worker,
                    "generation": generation,
                    "engine_instance_id": engine,
                    "capacity": None if self.capacity is None else self.capacity.model_dump(),
                },
            )
            return
        if scope["method"] == "GET" and path in {"/v1/models", "/metrics", "/version"}:
            await self.app(scope, receive, send)
            return
        if scope["method"] != "POST" or (
            path not in INFERENCE_PATHS
            and not (path == "/freechat/prepare" and self.preparer is not None)
        ):
            await self._error(send, 404, "endpoint not enabled by managed worker")
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 4 * 1024 * 1024:
                await self._error(send, 413, "request body too large")
                return
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            if payload.get("background") or payload.get("n", 1) != 1:
                raise ValueError("background and n>1 require further execution tracking")
            headers: dict[str, str] = {}
            for key, value in scope["headers"]:
                name = key.decode("latin1").lower()
                if name.startswith("x-freechat-internal-"):
                    if name in headers:
                        raise ValueError("duplicate execution identity header")
                    headers[name] = value.decode("latin1")
            if path == "/freechat/prepare":
                assert self.preparer is not None
                if not isinstance(payload.get("request"), dict):
                    raise ValueError("preparation requires a native request object")
                preparation_id, prepared_budget = await self.preparer.prepare(
                    headers["x-freechat-internal-tenant"], payload["protocol"], payload["request"]
                )
                await self._reply(send, 200, {
                    "preparation_id": preparation_id,
                    "budget": prepared_budget.model_dump(),
                    "worker_id": self.driver.identity[0],
                    "generation": self.driver.identity[1],
                    "engine_instance_id": self.driver.identity[2],
                })
                return
            budget = None if self.preparer is None else self.preparer.require(
                headers["x-freechat-internal-preparation"],
                headers["x-freechat-internal-tenant"], path, payload,
            )
            if budget is not None and self.capacity is not None:
                total_tokens = budget.input_tokens + budget.output_tokens
                blocks = (
                    total_tokens + self.capacity.block_size_tokens - 1
                ) // self.capacity.block_size_tokens
                required = blocks * self.capacity.block_bytes
                if (
                    total_tokens > self.capacity.max_context_tokens
                    or required > self.capacity.usable_bytes
                    or int(headers["x-freechat-internal-reserved-kv-bytes"]) != required
                ):
                    raise ValueError("reservation_does_not_match_measured_kv_budget")
            command = ExecutionCommand(
                worker_id=self.driver.identity[0],
                action=ExecutionAction.QUERY,
                tenant_id=headers["x-freechat-internal-tenant"],
                request_id=headers["x-freechat-internal-request-id"],
                decision_id=headers["x-freechat-internal-decision-id"],
                worker_generation=int(headers["x-freechat-internal-worker-generation"]),
                engine_instance_id=headers["x-freechat-internal-engine-instance-id"],
            )
            key = await self.driver.admit(
                command, {} if budget is None else {"budget": budget.model_dump()}
            )
        except (KeyError, ValueError) as error:
            await self._error(send, 409, str(error))
            return
        delivered = False

        async def replay_receive() -> Any:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        context_token = CURRENT_ROUTE.set(key)
        try:
            await self.app(scope, replay_receive, send)
        finally:
            self.backend.close_submission(key)
            CURRENT_ROUTE.reset(context_token)

    @staticmethod
    async def _reply(send: Any, status: int, payload: dict[str, Any]) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": json.dumps(payload).encode()})

    @classmethod
    async def _error(cls, send: Any, status: int, message: str) -> None:
        await cls._reply(send, status, {"error": {"message": message}})
