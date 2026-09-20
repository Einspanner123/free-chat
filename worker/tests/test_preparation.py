from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from freechat_worker.preparation import PreparationService


async def render(path: str, body: dict[str, Any]) -> tuple[list[int], int]:
    assert path == "/v1/chat/completions"
    await asyncio.sleep(0)
    return [1, 2, 3], int(body.get("max_tokens", 4))


async def test_preparation_binds_tenant_protocol_body_and_incarnation() -> None:
    service = PreparationService(render)
    body = {"model": "m", "max_tokens": 4}
    key, budget = await service.prepare("tenant", "/v1/chat/completions", body)
    assert budget.input_tokens == 3 and budget.output_tokens == 4
    assert service.require(key, "tenant", "/v1/chat/completions", body) == budget
    # Lifecycle scheduling metadata is added after placement, not rendered text.
    assert service.require(
        key, "tenant", "/v1/chat/completions", {**body, "agent_lifecycle": {"decision": "d"}}
    ) == budget
    for tenant, path, request in [
        ("other", "/v1/chat/completions", body),
        ("tenant", "/v1/responses", body),
        ("tenant", "/v1/chat/completions", {**body, "max_tokens": 5}),
    ]:
        with pytest.raises(ValueError, match="mismatched"):
            service.require(key, tenant, path, request)
    with pytest.raises(ValueError, match="missing"):
        PreparationService(render).require(key, "tenant", "/v1/chat/completions", body)


async def test_preparation_expiry_and_bound_reclaim_only_expired_entries() -> None:
    now = [100.0]
    service = PreparationService(render, limit=1, ttl=2, clock=lambda: now[0])
    key, _ = await service.prepare("tenant", "/v1/chat/completions", {})
    with pytest.raises(ValueError, match="capacity_exhausted"):
        await service.prepare("tenant", "/v1/chat/completions", {})
    now[0] = 102.0
    with pytest.raises(ValueError, match="expired"):
        service.require(key, "tenant", "/v1/chat/completions", {})
    await service.prepare("tenant", "/v1/chat/completions", {})
    assert len(service.records) == 1 and key not in service.records


async def test_concurrent_rendering_cannot_overfill_preparation_store() -> None:
    service = PreparationService(render, limit=1)
    results = await asyncio.gather(
        service.prepare("tenant", "/v1/chat/completions", {}),
        service.prepare("tenant", "/v1/chat/completions", {}),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert len(service.records) == 1


@pytest.mark.parametrize("change", ["tokens", "output", "fanout", "expired"])
async def test_prepared_budget_is_checked_again_at_engine_boundary(change: str) -> None:
    service = PreparationService(render, clock=lambda: 100.0)
    _, budget = await service.prepare("tenant", "/v1/chat/completions", {})
    prompt: Any = {"prompt_token_ids": [1, 2, 3]}
    params = SimpleNamespace(n=1, max_tokens=4)
    budget.check_execution(prompt, params, 101.0)
    now = 101.0
    if change == "tokens":
        prompt["prompt_token_ids"].append(4)
    elif change == "output":
        params.max_tokens = 5
    elif change == "fanout":
        params.n = 2
    else:
        now = budget.expires_at
    with pytest.raises(ValueError, match="prepared_budget"):
        budget.check_execution(prompt, params, now)


async def test_native_parser_mutation_cannot_change_original_request_fingerprint() -> None:
    async def mutating_render(path: str, body: dict[str, Any]) -> tuple[list[int], int]:
        body["tool_choice"] = "none"
        body["messages"][0]["content"] = "normalized"
        return [1], 4

    service = PreparationService(mutating_render)
    body = {"messages": [{"role": "user", "content": "original"}]}
    key, budget = await service.prepare("tenant", "/v1/chat/completions", body)
    assert body == {"messages": [{"role": "user", "content": "original"}]}
    assert service.require(key, "tenant", "/v1/chat/completions", body) == budget
