import pytest
from freechat_contracts import AgentHints, PrefixScope, scoped_cache_salt


def hints(scope: PrefixScope) -> AgentHints:
    return AgentHints(
        harness_id="harness",
        task_id="task",
        session_id="session",
        agent_id="agent",
        branch_id="branch",
        call_id="call",
        prefix_scope=scope,
    )


@pytest.mark.parametrize("scope", list(PrefixScope))
@pytest.mark.parametrize(
    "field",
    ["harness_id", "task_id", "session_id", "agent_id", "branch_id", "call_id"],
)
def test_scope_identity_matrix(scope: PrefixScope, field: str) -> None:
    fields = {
        PrefixScope.GLOBAL: set(),
        PrefixScope.HARNESS: {"harness_id"},
        PrefixScope.TASK: {"harness_id", "task_id", "session_id"},
        PrefixScope.AGENT: {"harness_id", "task_id", "session_id", "agent_id"},
        PrefixScope.BRANCH: {"harness_id", "task_id", "session_id", "agent_id", "branch_id"},
        PrefixScope.PRIVATE: {
            "harness_id",
            "task_id",
            "session_id",
            "agent_id",
            "branch_id",
            "call_id",
        },
    }
    original = hints(scope)
    changed = original.model_copy(update={field: "other"})
    assert (
        scoped_cache_salt(b"s" * 32, "tenant", original)
        != scoped_cache_salt(b"s" * 32, "tenant", changed)
    ) == (field in fields[scope])


@pytest.mark.parametrize("scope", list(PrefixScope))
def test_tenant_and_privacy_domain_always_bound(scope: PrefixScope) -> None:
    original = hints(scope)
    salt = scoped_cache_salt(b"s" * 32, "tenant", original)
    assert salt == scoped_cache_salt(b"s" * 32, "tenant", original)
    assert salt != scoped_cache_salt(b"s" * 32, "other", original)
    assert salt != scoped_cache_salt(
        b"s" * 32,
        "tenant",
        original.model_copy(update={"privacy_domain": "other"}),
    )


@pytest.mark.parametrize(
    "secret,tenant,scope", [(b"short", "t", "s"), (b"s" * 32, "", "s"), (b"s" * 32, "t", "")]
)
def test_invalid_salt_inputs(secret: bytes, tenant: str, scope: str) -> None:
    from freechat_contracts import derive_cache_salt

    with pytest.raises(ValueError):
        derive_cache_salt(secret, tenant, scope)


def test_canonical_namespace_and_private_identity() -> None:
    first = hints(PrefixScope.AGENT).model_copy(update={"task_id": "a:b", "agent_id": "c"})
    second = first.model_copy(update={"task_id": "a", "agent_id": "b:c"})
    assert scoped_cache_salt(b"s" * 32, "tenant", first) != scoped_cache_salt(
        b"s" * 32, "tenant", second
    )
    with pytest.raises(ValueError, match="call identity"):
        scoped_cache_salt(
            b"s" * 32, "tenant", hints(PrefixScope.PRIVATE).model_copy(update={"call_id": ""})
        )
