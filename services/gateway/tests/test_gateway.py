import json

import httpx
from fastapi.testclient import TestClient
from freechat_gateway import GatewayConfig, create_app


def upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert "freechat" not in body
    assert len(body["cache_salt"]) == 64
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
