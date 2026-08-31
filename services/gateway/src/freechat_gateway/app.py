from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from freechat_contracts import RequestProfile, derive_cache_salt

from freechat_gateway.auth import APIKeyAuthenticator, AuthContext
from freechat_gateway.hints import estimate_input_tokens, extract_agent_hints
from freechat_gateway.routing import SchedulerClient, StaticSchedulerClient


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
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    authenticator = APIKeyAuthenticator(config.api_keys)
    scheduler_client = scheduler or StaticSchedulerClient(
        worker_id="compose-worker",
        endpoint=config.default_worker_endpoint,
    )
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.request_timeout_seconds),
        transport=transport,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await http_client.aclose()

    app = FastAPI(
        title="FreeChat Agent-Aware Inference Infrastructure",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
        stream = bool(body.get("stream", False))
        response_headers = {
            "x-freechat-request-id": request_id,
            "x-freechat-task-id": hints.task_id,
            "x-freechat-decision-id": decision.decision_id,
            "x-freechat-route-class": "agent-aware" if hints.confidence > 0.25 else "compatible",
        }
        if stream:
            return StreamingResponse(
                _stream_upstream(http_client, upstream_url, upstream_headers, body),
                media_type="text/event-stream",
                headers=response_headers,
            )
        upstream = await http_client.post(upstream_url, headers=upstream_headers, json=body)
        media_type = upstream.headers.get("content-type", "application/json").split(";", 1)[0]
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=media_type,
            headers=response_headers,
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


async def _stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> AsyncIterator[bytes]:
    async with client.stream("POST", url, headers=headers, json=body) as response:
        if response.is_error:
            yield await response.aread()
            return
        async for chunk in response.aiter_bytes():
            yield chunk
