from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from freechat_contracts import AgentHints, RequestProfile, RouteDecision, derive_cache_salt

from freechat_gateway.auth import APIKeyAuthenticator, AuthContext
from freechat_gateway.console import ConsoleReadModel, EmptyConsoleReadModel
from freechat_gateway.hints import estimate_input_tokens, extract_agent_hints
from freechat_gateway.routing import SchedulerClient, StaticSchedulerClient

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    api_keys: dict[str, str]
    cache_salt_secret: bytes
    default_worker_endpoint: str = "http://worker:8000"
    request_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if len(self.cache_salt_secret) < 32:
            raise ValueError("cache_salt_secret must be at least 32 bytes")
        if not self.api_keys:
            raise ValueError("at least one API key is required")


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
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPException(status_code=400, detail="invalid JSON body") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")

        hints = extract_agent_hints(request, body)
        request_id = request.headers.get("x-request-id", str(uuid4()))
        model_id = str(body.get("model", ""))
        if not model_id:
            raise HTTPException(status_code=422, detail="model is required")
        output_tokens = int(body.get("max_tokens") or body.get("max_output_tokens") or 512)
        cache_salt = derive_cache_salt(
            config.cache_salt_secret,
            auth.tenant_id,
            f"{hints.prefix_scope}:{hints.privacy_domain}",
        )
        profile = RequestProfile(
            request_id=request_id,
            tenant_id=auth.tenant_id,
            model_id=model_id,
            input_tokens=estimate_input_tokens(body),
            output_tokens=output_tokens,
            cache_key=request.headers.get("x-freechat-cache-key"),
            hints=hints,
        )
        decision = await scheduler_client.route(profile)
        upstream_url = f"{decision.endpoint.rstrip('/')}{path}"
        upstream_headers = _upstream_headers(
            request,
            auth,
            request_id=request_id,
            decision_id=decision.decision_id,
            worker_generation=decision.worker_generation,
        )
        body["cache_salt"] = cache_salt
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
            "x-freechat-route-class": "agent-aware" if hints.confidence > 0.25 else "compatible",
        }
        if stream:
            try:
                upstream = await _send_stream_request(
                    http_client, upstream_url, upstream_headers, body
                )
            except HTTPException:
                await _release_route(scheduler_client, profile, decision)
                raise
            media_type = upstream.headers.get(
                "content-type", "text/event-stream"
            ).split(";", 1)[0]
            if upstream.is_error:
                content = await upstream.aread()
                status_code = upstream.status_code
                await upstream.aclose()
                await _release_route(scheduler_client, profile, decision)
                return Response(
                    content=content,
                    status_code=status_code,
                    media_type=media_type,
                    headers=response_headers,
                )
            return StreamingResponse(
                _relay_upstream(
                    upstream,
                    scheduler=scheduler_client,
                    profile=profile,
                    decision=decision,
                ),
                status_code=upstream.status_code,
                media_type=media_type,
                headers=response_headers,
            )
        try:
            try:
                upstream = await http_client.post(
                    upstream_url, headers=upstream_headers, json=body
                )
            except httpx.TimeoutException as error:
                raise HTTPException(
                    status_code=504, detail="worker request timed out"
                ) from error
            except httpx.RequestError as error:
                raise HTTPException(status_code=502, detail="worker request failed") from error
            media_type = upstream.headers.get("content-type", "application/json").split(
                ";", 1
            )[0]
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=media_type,
                headers=response_headers,
            )
        finally:
            await _release_route(scheduler_client, profile, decision)

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
) -> dict[str, str]:
    traceparent = request.headers.get("traceparent", "")
    return {
        "content-type": "application/json",
        "x-freechat-internal-tenant": auth.tenant_id,
        "x-freechat-internal-request-id": request_id,
        "x-freechat-internal-decision-id": decision_id,
        "x-freechat-internal-worker-generation": str(worker_generation),
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
) -> AsyncIterator[bytes]:
    keepalive = asyncio.create_task(_renew_lease_loop(scheduler, profile, decision))
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        keepalive.cancel()
        try:
            with suppress(asyncio.CancelledError):
                await keepalive
        finally:
            try:
                await response.aclose()
            finally:
                await _release_route(scheduler, profile, decision)


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
) -> None:
    try:
        await scheduler.release(profile, decision)
    except Exception:
        LOGGER.exception(
            "scheduler lease release failed",
            extra={
                "decision_id": decision.decision_id,
                "request_id": profile.request_id,
            },
        )
