import json
from collections.abc import AsyncIterator

import httpx
from fastapi.testclient import TestClient
from freechat_contracts import RequestProfile, RouteDecision
from freechat_gateway import GatewayConfig, create_app
from freechat_gateway.routing import StaticSchedulerClient


class StaticAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class RecordingScheduler:
    def __init__(self) -> None:
        self._delegate = StaticSchedulerClient("worker", "http://worker:8000")
        self.releases: list[tuple[RequestProfile, RouteDecision]] = []

    async def route(self, request: RequestProfile) -> RouteDecision:
        return await self._delegate.route(request)

    async def release(self, request: RequestProfile, decision: RouteDecision) -> None:
        self.releases.append((request, decision))

    async def aclose(self) -> None:
        return None


def upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert "freechat" not in body
    assert len(body["cache_salt"]) == 64
    assert body["agent_lifecycle"]["tenant_id"] == "tenant-a"
    assert body["agent_lifecycle"]["worker_generation"] == 1
    assert body["agent_lifecycle"]["cache_generation"] == 1
    assert request.headers["x-freechat-internal-tenant"] == "tenant-a"
    return httpx.Response(200, json={"id": "completion", "model": body["model"]})


def client() -> TestClient:
    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        transport=httpx.MockTransport(upstream),
    )
    return TestClient(app)


def test_chat_compatibility_and_explicit_hints() -> None:
    response = client().post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer secret-key"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "freechat": {
                "agent_hints": {
                    "harness_id": "openai-agents",
                    "task_id": "task-1",
                    "agent_id": "agent-1",
                }
            },
        },
    )
    assert response.status_code == 200
    assert response.headers["x-freechat-task-id"] == "task-1"
    assert response.headers["x-freechat-route-class"] == "agent-aware"


def test_anthropic_messages_is_proxied() -> None:
    response = client().post(
        "/v1/messages",
        headers={"x-api-key": "secret-key"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-freechat-route-class"] == "compatible"


def test_anthropic_messages_receives_authenticated_lifecycle() -> None:
    response = client().post(
        "/v1/messages",
        headers={"x-api-key": "secret-key"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "continue"}],
            "freechat": {
                "agent_hints": {
                    "harness_id": "openhands",
                    "task_id": "task-1",
                    "agent_id": "agent-1",
                    "lifecycle": "resume",
                    "expected_resume_ms": 100,
                }
            },
            "cache_salt": "attacker-controlled",
            "agent_lifecycle": {"tenant_id": "attacker"},
        },
    )
    assert response.status_code == 200
    assert response.headers["x-freechat-route-class"] == "agent-aware"


def test_invalid_resume_hints_return_serializable_422() -> None:
    response = client().post(
        "/v1/messages",
        headers={"x-api-key": "secret-key"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "continue"}],
            "freechat": {
                "agent_hints": {
                    "harness_id": "openhands",
                    "task_id": "task-1",
                    "agent_id": "agent-1",
                    "lifecycle": "resume",
                }
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "value_error"


def test_tenant_cannot_be_supplied_in_hints() -> None:
    response = client().post(
        "/v1/responses",
        headers={"x-api-key": "secret-key"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "input": "hello",
            "freechat": {
                "agent_hints": {
                    "harness_id": "langgraph",
                    "task_id": "task",
                    "agent_id": "agent",
                    "tenant_id": "attacker",
                }
            },
        },
    )
    assert response.status_code == 422


def test_client_cannot_forge_internal_agent_lifecycle() -> None:
    response = client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret-key"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_lifecycle": {"tenant_id": "attacker", "worker_generation": 999},
        },
    )
    assert response.status_code == 200


def test_missing_credentials_is_rejected() -> None:
    response = client().post(
        "/v1/chat/completions",
        json={"model": "model", "messages": []},
    )
    assert response.status_code == 401


def test_console_is_authenticated_and_never_fabricates_evidence() -> None:
    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32)
    )
    with TestClient(app) as client:
        denied = client.get("/control/ui/topology")
        response = client.get("/control/ui/topology", headers={"x-api-key": "secret-key"})
    assert denied.status_code == 401
    assert response.status_code == 200
    assert response.json() == {
        "state": "UNVERIFIED",
        "generated_at": None,
        "summary": "No control-plane evidence has been ingested for topology.",
        "metrics": [],
        "records": [],
    }


def test_streaming_relays_sse_without_buffering_status_loss() -> None:
    stream = StaticAsyncStream([b"data: first\n\n", b"data: [DONE]\n\n"])
    scheduler = RecordingScheduler()

    def stream_upstream(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            201,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=stream,
        )

    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        scheduler=scheduler,
        transport=httpx.MockTransport(stream_upstream),
    )
    with TestClient(app) as test_client, test_client.stream(
        "POST",
        "/v1/responses",
        headers={"x-api-key": "secret-key"},
        json={"model": "local", "input": "hello", "stream": True},
    ) as response:
        payload = b"".join(response.iter_bytes())

    assert response.status_code == 201
    assert response.headers["content-type"].startswith("text/event-stream")
    assert payload == b"data: first\n\ndata: [DONE]\n\n"
    assert stream.closed is True
    assert len(scheduler.releases) == 1


def test_streaming_worker_error_preserves_http_status() -> None:
    scheduler = RecordingScheduler()

    def failed_upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "application/json"},
            json={"error": {"message": "worker overloaded"}},
        )

    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        scheduler=scheduler,
        transport=httpx.MockTransport(failed_upstream),
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "secret-key"},
            json={"model": "local", "messages": [], "stream": True},
        )

    assert response.status_code == 429
    assert response.json() == {"error": {"message": "worker overloaded"}}
    assert len(scheduler.releases) == 1


def test_worker_transport_failure_returns_bad_gateway() -> None:
    scheduler = RecordingScheduler()

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unavailable", request=request)

    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        scheduler=scheduler,
        transport=httpx.MockTransport(unavailable),
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/responses",
            headers={"x-api-key": "secret-key"},
            json={"model": "local", "input": "hello"},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "worker request failed"}
    assert len(scheduler.releases) == 1


def test_streaming_transport_failure_returns_bad_gateway_before_headers() -> None:
    scheduler = RecordingScheduler()

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unavailable", request=request)

    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        scheduler=scheduler,
        transport=httpx.MockTransport(unavailable),
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/responses",
            headers={"x-api-key": "secret-key"},
            json={"model": "local", "input": "hello", "stream": True},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "worker stream failed"}
    assert len(scheduler.releases) == 1


def test_non_streaming_success_releases_route_once() -> None:
    scheduler = RecordingScheduler()
    app = create_app(
        GatewayConfig(api_keys={"tenant-a": "secret-key"}, cache_salt_secret=b"s" * 32),
        scheduler=scheduler,
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "secret-key", "x-request-id": "request-1"},
            json={"model": "local", "messages": []},
        )

    assert response.status_code == 200
    assert len(scheduler.releases) == 1
    profile, decision = scheduler.releases[0]
    assert profile.request_id == "request-1"
    assert decision.request_id == "request-1"
