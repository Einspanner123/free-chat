from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import anyio
import grpc
import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from freechat_contracts import AgentHints, RequestProfile, RouteDecision, scoped_cache_salt

from freechat_gateway.auth import APIKeyAuthenticator, AuthContext
from freechat_gateway.console import ConsoleReadModel, EmptyConsoleReadModel
from freechat_gateway.hints import extract_agent_hints
from freechat_gateway.routing import SchedulerClient, StaticSchedulerClient

LOGGER = logging.getLogger(__name__)
_CLEANUP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    api_keys: dict[str, str]
    cache_salt_secret: bytes
    default_worker_endpoint: str = "http://worker:8000"
    request_timeout_seconds: float = 600.0
    origin_node_id: str | None = None
    worker_token: str | None = None

    def __post_init__(self) -> None:
        if len(self.cache_salt_secret) < 32:
            raise ValueError("cache_salt_secret must be at least 32 bytes")
        if self.worker_token is not None and len(self.worker_token) < 32:
            raise ValueError("worker_token must be at least 32 characters")
        if not self.api_keys:
            raise ValueError("at least one API key is required")
        if self.origin_node_id is not None and (
            not self.origin_node_id.strip() or self.origin_node_id != self.origin_node_id.strip()
        ):
            raise ValueError("origin_node_id must be nonblank and trimmed")


def create_app(
    config: GatewayConfig,
    *,
    scheduler: SchedulerClient | None = None,
    console: ConsoleReadModel | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    authenticator = APIKeyAuthenticator(config.api_keys)
    scheduler_client = scheduler or StaticSchedulerClient(
        worker_id="compose-worker",
        endpoint=config.default_worker_endpoint,
    )
    console_read_model = console or EmptyConsoleReadModel()
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.request_timeout_seconds),
        transport=transport,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await http_client.aclose()
        await scheduler_client.aclose()

    app = FastAPI(
        title="FreeChat Agent-Aware Inference Infrastructure",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/control/ui/{view}")
    async def console_snapshot(
        view: str,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Response:
        if view not in {"topology", "traces", "kv-cache", "benchmarks", "playground"}:
            raise HTTPException(status_code=404, detail="unknown console view")
        auth = _authenticate(authenticator, authorization, x_api_key)
        snapshot = await console_read_model.snapshot(view, auth.tenant_id)
        return Response(
            content=snapshot.model_dump_json(by_alias=True),
            media_type="application/json",
        )

    async def proxy(
        path: str,
        request: Request,
        authorization: str | None,
        x_api_key: str | None,
    ) -> Response:
        auth = _authenticate(authenticator, authorization, x_api_key)
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 4 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="request body too large")
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPException(status_code=400, detail="invalid JSON body") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")
        if "kv_transfer_params" in body:
            raise HTTPException(
                status_code=400,
                detail="kv_transfer_params is reserved for the trusted control plane",
            )

        hints = extract_agent_hints(request, body)
        request_id = request.headers.get("x-request-id", str(uuid4()))
        model_id = body.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise HTTPException(status_code=422, detail="model is required")
        cache_salt = scoped_cache_salt(config.cache_salt_secret, auth.tenant_id, hints)
        body["cache_salt"] = cache_salt
        body.pop("agent_lifecycle", None)
        native_body = json.dumps(body, ensure_ascii=False)
        if len(native_body.encode()) > 4 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="prepared request body too large")
        profile = RequestProfile(
            request_id=request_id,
            tenant_id=auth.tenant_id,
            model_id=model_id,
            # Unknown until the selected native engine renders the complete request.
            input_tokens=0,
            output_tokens=1,
            native_protocol=path,
            native_request_json=native_body,
            cache_key=request.headers.get("x-freechat-cache-key"),
            hints=hints,
            local_node_id=config.origin_node_id,
        )
        try:
            decision = await scheduler_client.route(profile)
        except grpc.aio.AioRpcError as error:
            status = {
                grpc.StatusCode.INVALID_ARGUMENT: 422,
                grpc.StatusCode.FAILED_PRECONDITION: 503,
                grpc.StatusCode.UNAVAILABLE: 503,
                grpc.StatusCode.DEADLINE_EXCEEDED: 504,
            }.get(error.code(), 502)
            raise HTTPException(
                status_code=status, detail="scheduler could not admit request"
            ) from error
        upstream_url = f"{decision.endpoint.rstrip('/')}{path}"
        upstream_headers = _upstream_headers(
            request,
            auth,
            request_id=request_id,
            decision_id=decision.decision_id,
            worker_generation=decision.worker_generation,
            engine_instance_id=decision.engine_instance_id,
        )
        if config.worker_token is not None:
            upstream_headers["x-freechat-worker-token"] = config.worker_token
        if decision.preparation is not None:
            upstream_headers["x-freechat-internal-preparation"] = (
                decision.preparation.preparation_id
            )
            upstream_headers["x-freechat-internal-reserved-kv-bytes"] = str(
                decision.reserved_kv_bytes_per_rank
            )
        if decision.kv_transfer.applicable:
            body["kv_transfer_params"] = {
                "max_offload_tokens": decision.kv_transfer.max_offload_tokens
            }
        if path in {"/v1/chat/completions", "/v1/responses", "/v1/messages"}:
            body["agent_lifecycle"] = _agent_lifecycle_body(
                auth,
                hints,
                cache_key=profile.cache_key or cache_salt,
                worker_generation=decision.worker_generation,
            )
        stream = bool(body.get("stream", False))
        response_headers = {
            "x-freechat-request-id": request_id,
            "x-freechat-task-id": hints.task_id,
            "x-freechat-decision-id": decision.decision_id,
            "x-freechat-worker-id": decision.worker_id,
            "x-freechat-reserved-kv-bytes": str(decision.reserved_kv_bytes_per_rank),
            "x-freechat-route-class": "agent-aware" if hints.confidence > 0.25 else "compatible",
            "x-freechat-kv-offload": ("enabled" if decision.kv_transfer.enabled else "disabled"),
        }
        if stream:
            keepalive = asyncio.create_task(_renew_lease_loop(scheduler_client, profile, decision))
            upstream = None
            try:
                upstream = await _send_stream_request(
                    http_client, upstream_url, upstream_headers, body
                )
                media_type = upstream.headers.get("content-type", "text/event-stream").split(
                    ";", 1
                )[0]
                if upstream.is_error:
                    content = await upstream.aread()
                    status_code = upstream.status_code
                    await _finish_route(
                        scheduler_client,
                        profile,
                        decision,
                        keepalive,
                        response=upstream,
                        uncertain=False,
                    )
                    return Response(
                        content=content,
                        status_code=status_code,
                        media_type=media_type,
                        headers=response_headers,
                    )
            except BaseException:
                await _finish_route(
                    scheduler_client,
                    profile,
                    decision,
                    keepalive,
                    response=upstream,
                    uncertain=True,
                )
                raise
            return StreamingResponse(
                _relay_upstream(
                    upstream,
                    scheduler=scheduler_client,
                    profile=profile,
                    decision=decision,
                    keepalive=keepalive,
                ),
                status_code=upstream.status_code,
                media_type=media_type,
                headers=response_headers,
            )
        completed = False
        upstream = None
        keepalive = asyncio.create_task(_renew_lease_loop(scheduler_client, profile, decision))
        try:
            try:
                upstream = await http_client.post(upstream_url, headers=upstream_headers, json=body)
                completed = True
            except httpx.TimeoutException as error:
                raise HTTPException(status_code=504, detail="worker request timed out") from error
            except httpx.RequestError as error:
                raise HTTPException(status_code=502, detail="worker request failed") from error
            media_type = upstream.headers.get("content-type", "application/json").split(";", 1)[0]
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=media_type,
                headers=response_headers,
            )
        finally:
            await _finish_route(
                scheduler_client,
                profile,
                decision,
                keepalive,
                response=upstream,
                uncertain=not completed,
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Response:
        return await proxy("/v1/chat/completions", request, authorization, x_api_key)

    @app.post("/v1/responses")
    async def responses(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Response:
        return await proxy("/v1/responses", request, authorization, x_api_key)

    @app.post("/v1/messages")
    async def messages(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Response:
        return await proxy("/v1/messages", request, authorization, x_api_key)

    return app


def _agent_lifecycle_body(
    auth: AuthContext,
    hints: AgentHints,
    *,
    cache_key: str,
    worker_generation: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tenant_id": auth.tenant_id,
        "cache_key": cache_key,
        "task_id": hints.task_id,
        "agent_id": hints.agent_id,
        "branch_id": hints.branch_id,
        "call_id": hints.call_id,
        "lifecycle": hints.lifecycle,
        "worker_generation": worker_generation,
        "cache_generation": worker_generation,
        "priority": hints.priority,
        "allow_kv_offload": hints.allow_kv_offload,
    }
    if hints.session_id is not None:
        metadata["session_id"] = hints.session_id
    if hints.expected_resume_ms is not None:
        metadata["expected_resume_ms"] = hints.expected_resume_ms
    return metadata


def _authenticate(
    authenticator: APIKeyAuthenticator,
    authorization: str | None,
    x_api_key: str | None,
) -> AuthContext:
    try:
        return authenticator.authenticate(authorization, x_api_key)
    except PermissionError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


def _upstream_headers(
    request: Request,
    auth: AuthContext,
    *,
    request_id: str,
    decision_id: str,
    worker_generation: int,
    engine_instance_id: str | None,
) -> dict[str, str]:
    traceparent = request.headers.get("traceparent", "")
    return {
        "content-type": "application/json",
        "x-freechat-internal-tenant": auth.tenant_id,
        "x-freechat-internal-request-id": request_id,
        "x-freechat-internal-decision-id": decision_id,
        "x-freechat-internal-worker-generation": str(worker_generation),
        "x-freechat-internal-engine-instance-id": engine_instance_id or "",
        "traceparent": traceparent,
    }


async def _send_stream_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> httpx.Response:
    request = client.build_request("POST", url, headers=headers, json=body)
    try:
        return await client.send(request, stream=True)
    except httpx.TimeoutException as error:
        raise HTTPException(status_code=504, detail="worker stream timed out") from error
    except httpx.RequestError as error:
        raise HTTPException(status_code=502, detail="worker stream failed") from error


async def _relay_upstream(
    response: httpx.Response,
    *,
    scheduler: SchedulerClient,
    profile: RequestProfile,
    decision: RouteDecision,
    keepalive: asyncio.Task[None] | None = None,
) -> AsyncGenerator[bytes, None]:
    keepalive = keepalive or asyncio.create_task(_renew_lease_loop(scheduler, profile, decision))
    completed = False
    try:
        async for chunk in response.aiter_raw():
            yield chunk
        completed = True
    finally:
        await _finish_route(
            scheduler,
            profile,
            decision,
            keepalive,
            response=response,
            uncertain=not completed,
        )


async def _finish_route(
    scheduler: SchedulerClient,
    profile: RequestProfile,
    decision: RouteDecision,
    keepalive: asyncio.Task[None],
    *,
    response: httpx.Response | None,
    uncertain: bool,
) -> None:
    # ASGI disconnect cancels the enclosing AnyIO scope at every checkpoint.
    # Cleanup must survive that scope, but must not block shutdown indefinitely.
    keepalive.cancel()
    with anyio.CancelScope(shield=True):
        try:
            async with asyncio.timeout(_CLEANUP_TIMEOUT_SECONDS):
                await asyncio.gather(keepalive, return_exceptions=True)
                if response is not None:
                    await response.aclose()
        except TimeoutError:
            LOGGER.warning("upstream cleanup timed out decision_id=%s", decision.decision_id)
        except Exception:
            LOGGER.exception("upstream cleanup failed decision_id=%s", decision.decision_id)

    # grpc.aio may replace cancellation without AnyIO's scope marker. asyncio's
    # deadline owns task cancellation and still recognizes that timeout.
    # Notification remains an intent, never proof of released GPU capacity.
    with anyio.CancelScope(shield=True):
        try:
            async with asyncio.timeout(_CLEANUP_TIMEOUT_SECONDS):
                await _release_route(scheduler, profile, decision, uncertain=uncertain)
        except TimeoutError:
            LOGGER.warning("scheduler cleanup timed out decision_id=%s", decision.decision_id)


async def _renew_lease_loop(
    scheduler: SchedulerClient,
    profile: RequestProfile,
    decision: RouteDecision,
) -> None:
    interval_seconds = max(0.25, decision.lease_ttl_ms / 3_000)
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await scheduler.renew(profile, decision)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception(
                "scheduler lease renewal failed",
                extra={
                    "decision_id": decision.decision_id,
                    "request_id": profile.request_id,
                },
            )


async def _release_route(
    scheduler: SchedulerClient,
    profile: RequestProfile,
    decision: RouteDecision,
    *,
    uncertain: bool = False,
) -> None:
    try:
        if uncertain:
            await scheduler.cancel(profile, decision)
        else:
            await scheduler.release(profile, decision)
    except Exception:
        LOGGER.exception(
            "scheduler lease release failed",
            extra={
                "decision_id": decision.decision_id,
                "request_id": profile.request_id,
            },
        )
