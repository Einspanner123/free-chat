from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from freechat_contracts import AgentHints, HintSource
from pydantic import ValidationError


def extract_agent_hints(request: Request, body: dict[str, Any]) -> AgentHints:
    extension = body.pop("freechat", None)
    if isinstance(extension, dict) and isinstance(extension.get("agent_hints"), dict):
        raw = dict(extension["agent_hints"])
        raw.setdefault("source", HintSource.EXPLICIT)
        try:
            return AgentHints.model_validate(raw)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error

    harness_id = request.headers.get("x-freechat-harness-id", "openai-compatible")
    task_id = request.headers.get(
        "x-freechat-task-id",
        request.headers.get("x-request-id", "unknown"),
    )
    agent_id = request.headers.get("x-freechat-agent-id", "unknown")
    branch_id = request.headers.get("x-freechat-branch-id", "main")
    return AgentHints(
        harness_id=harness_id,
        task_id=task_id,
        agent_id=agent_id,
        branch_id=branch_id,
        source=HintSource.INFERRED,
        confidence=0.25,
    )


def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Conservative admission estimate; worker tokenizer remains authoritative."""
    pieces: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                pieces.append(str(message.get("content", "")))
    input_value = body.get("input")
    if input_value is not None:
        pieces.append(str(input_value))
    system_value = body.get("system")
    if system_value is not None:
        pieces.append(str(system_value))
    tools = body.get("tools")
    if tools is not None:
        pieces.append(str(tools))
    return max(1, sum(len(piece) for piece in pieces) // 2)
