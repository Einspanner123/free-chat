import pytest
from freechat_contracts.cache_identity import derive_cache_salt, make_cache_identity
from freechat_contracts.models import (
    AgentHints,
    HintSource,
    Lifecycle,
    validate_lifecycle_transition,
)
from pydantic import ValidationError
from typing_extensions import TypedDict


class CacheIdentityArguments(TypedDict):
    model_id: str
    model_revision: str
    chat_template_hash: str
    adapter_id: str | None
    kv_layout: str
    cache_salt: str
    token_ids: list[int]


def test_inferred_hints_require_non_absolute_confidence() -> None:
    with pytest.raises(ValidationError):
        AgentHints(
            harness_id="black-box",
            task_id="task",
            agent_id="agent",
            source=HintSource.INFERRED,
            confidence=1.0,
        )


def test_tool_wait_requires_resume_prediction() -> None:
    with pytest.raises(ValidationError):
        AgentHints(
            harness_id="agents",
            task_id="task",
            agent_id="agent",
            lifecycle=Lifecycle.TOOL_WAIT,
        )


def test_terminal_cannot_resume() -> None:
    with pytest.raises(ValueError, match="invalid lifecycle"):
        validate_lifecycle_transition(Lifecycle.TERMINAL, Lifecycle.RESUME)


def test_cache_salt_is_tenant_scoped() -> None:
    secret = b"s" * 32
    assert derive_cache_salt(secret, "tenant-a", "task") != derive_cache_salt(
        secret, "tenant-b", "task"
    )


def test_cache_identity_changes_with_tokenizer_revision() -> None:
    common: CacheIdentityArguments = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "model-sha",
        "chat_template_hash": "template-sha",
        "adapter_id": None,
        "kv_layout": "paged-fp16",
        "cache_salt": "salt",
        "token_ids": [1, 2, 3],
    }
    first = make_cache_identity(tokenizer_revision="tokenizer-a", **common)
    second = make_cache_identity(tokenizer_revision="tokenizer-b", **common)
    assert first != second
