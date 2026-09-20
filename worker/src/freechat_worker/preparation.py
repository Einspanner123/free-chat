"""Native rendering budgets, bound to tenant/body and checked before GPU submission."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import secrets
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/messages"}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def body_digest(body: dict[str, Any]) -> str:
    # Scheduling metadata is supplied after preparation and never rendered as text.
    return digest({key: value for key, value in body.items() if key != "agent_lifecycle"})


class TokenBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    protocol: str
    body_sha256: str
    prompt_sha256: str
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    expires_at: float

    def check_execution(self, prompt: Any, sampling_params: Any, now: float) -> None:
        if (
            now >= self.expires_at
            or not isinstance(prompt, dict)
            or digest(prompt.get("prompt_token_ids")) != self.prompt_sha256
            or getattr(sampling_params, "n", None) != 1
            or getattr(sampling_params, "max_tokens", 0) < 1
            or sampling_params.max_tokens > self.output_tokens
        ):
            raise ValueError("execution_does_not_match_prepared_budget")


Renderer = Callable[[str, dict[str, Any]], Awaitable[tuple[list[int], int]]]


class PreparationService:
    def __init__(
        self, renderer: Renderer, *, limit: int = 4096, ttl: float = 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if limit < 1 or ttl <= 0:
            raise ValueError("invalid preparation bounds")
        self.renderer, self.limit, self.ttl, self.clock = renderer, limit, ttl, clock
        self.records: dict[str, TokenBudget] = {}
        self.slots = asyncio.Semaphore(16)

    async def prepare(
        self, tenant_id: str, protocol: str, body: dict[str, Any],
    ) -> tuple[str, TokenBudget]:
        if not tenant_id or protocol not in PATHS:
            raise ValueError("invalid preparation scope")
        self.records = {
            key: budget for key, budget in self.records.items()
            if budget.expires_at > self.clock()
        }
        if len(self.records) >= self.limit:
            raise ValueError("preparation_capacity_exhausted")
        rendering_body = deepcopy(body)
        request_sha256 = body_digest(rendering_body)
        async with self.slots:
            tokens, output = await self.renderer(protocol, rendering_body)
        budget = TokenBudget(
            tenant_id=tenant_id, protocol=protocol, body_sha256=request_sha256,
            prompt_sha256=digest(tokens), input_tokens=len(tokens), output_tokens=output,
            expires_at=self.clock() + self.ttl,
        )
        # Rendering awaits: recheck the bound before installing a result.
        if len(self.records) >= self.limit:
            raise ValueError("preparation_capacity_exhausted")
        key = secrets.token_hex(32)
        self.records[key] = budget
        return key, budget

    def require(
        self, key: str, tenant_id: str, protocol: str, body: dict[str, Any],
    ) -> TokenBudget:
        budget = self.records.get(key)
        if (
            budget is None or budget.expires_at <= self.clock()
            or budget.tenant_id != tenant_id or budget.protocol != protocol
            or budget.body_sha256 != body_digest(body)
        ):
            raise ValueError("preparation_missing_expired_or_mismatched")
        return budget


class NativeRenderer:
    """Thin adapter over the pinned native protocol handlers; no template copy."""

    def __init__(self, state: Any) -> None:
        self.state = state

    async def __call__(self, protocol: str, body: dict[str, Any]) -> tuple[list[int], int]:
        if body.get("background") or body.get("n", 1) != 1 or body.get("use_beam_search"):
            raise ValueError("preparation_requires_foreground_single_sample")
        if protocol == "/v1/responses":
            service = self.state.openai_serving_responses
            request = importlib.import_module(
                "vllm.entrypoints.openai.responses.protocol"
            ).ResponsesRequest.model_validate(body)
            # Built-in tool loops require separately admitted later model calls.
            if any(tool.type != "function" for tool in request.tools):
                raise ValueError("native_builtin_tool_loops_require_per_call_admission")
            error = await service._check_model(request)
            if error is None:
                error = service._validate_create_responses_input(request)
            if error is not None:
                raise ValueError(str(error))
            previous = None
            if request.previous_response_id:
                async with service.response_store_lock:
                    previous = service.response_store.get(request.previous_response_id)
                if previous is None:
                    raise ValueError("previous_response_not_found")
            if service.use_harmony:
                _, inputs = service._make_request_with_harmony(request, previous)
            else:
                _, inputs = await service._make_request(request, previous)
            output_limit = request.max_output_tokens
            truncate = -1 if request.truncation != "disabled" else None
        else:
            if protocol == "/v1/messages":
                service = self.state.anthropic_serving_messages
                request = importlib.import_module(
                    "vllm.entrypoints.anthropic.protocol"
                ).AnthropicMessagesRequest.model_validate(body)
                request = service._convert_anthropic_to_openai_request(
                    request, merge_inline_system=service._merge_inline_system,
                )
            elif protocol == "/v1/chat/completions":
                service = self.state.openai_serving_chat
                request = importlib.import_module(
                    "vllm.entrypoints.openai.chat_completion.protocol"
                ).ChatCompletionRequest.model_validate(body)
            else:
                raise ValueError("unsupported preparation protocol")
            result = await service.render_chat_request(request)
            if not isinstance(result, tuple):
                raise ValueError(str(result))
            _, inputs = result
            output_limit = (
                request.max_completion_tokens
                if request.max_completion_tokens is not None else request.max_tokens
            )
            truncate = request.truncate_prompt_tokens
        if (
            len(inputs) != 1 or inputs[0].get("type", "token") != "token"
            or "prompt_token_ids" not in inputs[0]
        ):
            raise ValueError("preparation_requires_one_text_prompt")
        tokens = list(inputs[0]["prompt_token_ids"])
        max_tokens = importlib.import_module(
            "vllm.entrypoints.serve.utils.api_utils"
        ).get_max_tokens(
            service.model_config.max_model_len, output_limit, len(tokens),
            service.default_sampling_params, service.override_max_tokens,
            truncate_prompt_tokens=truncate,
        )
        return tokens, int(max_tokens)
