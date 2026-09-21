"""Pure fork policy boundaries; not allocator integration or physical GPU evidence."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

HOOK_PATH = (
    Path(__file__).resolve().parents[1] / "third_party/vllm/vllm/v1/core/agent_cache_hooks.py"
)


@pytest.fixture(scope="module")
def hooks() -> Any:
    """Load only this dependency-free file; never import the vLLM package."""
    name = "_freechat_fork_cache_hook_edges"
    spec = importlib.util.spec_from_file_location(name, HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Dataclasses resolve the declaring module.
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def mapping(**changes: Any) -> dict[str, Any]:
    return {
        "tenant_id": "tenant",
        "cache_key": "cache",
        "task_id": "task",
        "agent_id": "agent",
        "branch_id": "branch",
        "call_id": "call",
        "worker_generation": 2,
        "cache_generation": 3,
        **changes,
    }


def metadata(hooks: Any, **changes: Any) -> Any:
    return hooks.AgentLifecycleMetadata.from_mapping(mapping(**changes))


def event(hooks: Any, kind: str, state: Any, blocks: Any = ((1, 2),)) -> Any:
    return hooks.CacheLifecycleEvent(
        hooks.CacheLifecycleEventType(kind), "request", blocks, state, "fixture"
    )


@pytest.mark.parametrize("value", [[], "text", 1])
def test_metadata_requires_an_object(hooks: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="must be an object"):
        hooks.AgentLifecycleMetadata.from_mapping(value)
    assert hooks.AgentLifecycleMetadata.from_mapping(None) is None


def test_unknown_metadata_fields_are_rejected(hooks: Any) -> None:
    with pytest.raises(ValueError, match="unknown agent_lifecycle fields"):
        metadata(hooks, untrusted="ignored?")


@pytest.mark.parametrize(
    "field", ["tenant_id", "cache_key", "task_id", "agent_id", "branch_id", "call_id"]
)
@pytest.mark.parametrize("value", ["", None, 1])
def test_required_identity_cannot_be_empty_or_coerced(hooks: Any, field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
        metadata(hooks, **{field: value})


@pytest.mark.parametrize("value", ["", 0, False])
def test_optional_session_must_be_a_string_if_present(hooks: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="session_id"):
        metadata(hooks, session_id=value)


def test_optional_session_is_preserved(hooks: Any) -> None:
    assert metadata(hooks, session_id="session").session_id == "session"


@pytest.mark.parametrize("field", ["worker_generation", "cache_generation"])
@pytest.mark.parametrize("value", [0, -1, True, "2", 2.5])
def test_generations_are_positive_nonboolean_integers(hooks: Any, field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        metadata(hooks, **{field: value})


@pytest.mark.parametrize("value", [-1, True, "10", 1.5])
def test_resume_horizon_is_nonnegative_integral(hooks: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        metadata(hooks, expected_resume_ms=value)


@pytest.mark.parametrize("lifecycle", ["tool_wait", "resume"])
def test_wait_and_resume_metadata_require_horizon(hooks: Any, lifecycle: str) -> None:
    with pytest.raises(ValueError, match="require expected_resume_ms"):
        metadata(hooks, lifecycle=lifecycle)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"total_blocks": 0}, "total_blocks"),
        ({"max_protected_fraction": -0.1}, "max_protected_fraction"),
        ({"max_protected_fraction": 1.1}, "max_protected_fraction"),
        ({"max_retain_ms": -1}, "max_retain_ms"),
    ],
)
def test_retention_constructor_rejects_invalid_budget(
    hooks: Any, options: Any, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        hooks.LifecycleRetentionPolicy(**{"total_blocks": 8, **options})


def test_environment_requires_explicit_opt_in_and_uses_bounded_values(
    hooks: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_AGENT_CACHE_POLICY", raising=False)
    assert hooks.LifecycleRetentionPolicy.from_environment(8) is None
    monkeypatch.setenv("VLLM_AGENT_CACHE_POLICY", "1")
    monkeypatch.setenv("VLLM_AGENT_CACHE_MAX_PROTECTED_FRACTION", "0.25")
    monkeypatch.setenv("VLLM_AGENT_CACHE_MAX_RETAIN_MS", "40")
    policy = hooks.LifecycleRetentionPolicy.from_environment(8)
    policy.update(
        metadata(hooks, lifecycle="tool_wait", expected_resume_ms=100), [1, 2, 3], now_ms=0
    )
    assert policy.protected_blocks(now_ms=39) == frozenset({1, 2})
    assert policy.protected_blocks(now_ms=40) == frozenset()
    monkeypatch.setenv("VLLM_AGENT_CACHE_MAX_PROTECTED_FRACTION", "garbage")
    with pytest.raises(ValueError):
        hooks.LifecycleRetentionPolicy.from_environment(8)


@pytest.mark.parametrize("ignored", ["no_metadata", "hit", "active", "no_horizon", "zero_capacity"])
def test_retention_ignores_events_without_wait_authority(hooks: Any, ignored: str) -> None:
    policy = hooks.LifecycleRetentionPolicy(
        total_blocks=8, max_protected_fraction=0 if ignored == "zero_capacity" else 1
    )
    state = metadata(hooks, lifecycle="tool_wait", expected_resume_ms=100)
    if ignored == "no_metadata":
        state = None
    elif ignored == "active":
        state = metadata(hooks)
    elif ignored == "no_horizon":
        state = replace(state, expected_resume_ms=None)
    policy.observe(event(hooks, "hit" if ignored == "hit" else "free", state), now_ms=0)
    assert policy.protected_blocks(now_ms=0) == frozenset()


@pytest.mark.parametrize("ignored", ["active", "no_horizon", "zero_capacity"])
def test_direct_update_does_not_protect_without_wait_budget(hooks: Any, ignored: str) -> None:
    policy = hooks.LifecycleRetentionPolicy(
        total_blocks=8, max_protected_fraction=0 if ignored == "zero_capacity" else 1
    )
    state = metadata(hooks, lifecycle="tool_wait", expected_resume_ms=100)
    if ignored == "active":
        state = metadata(hooks)
    elif ignored == "no_horizon":
        state = replace(state, expected_resume_ms=None)
    policy.update(state, [1, 2], now_ms=0)
    assert policy.retained_count == 0


def test_monotonic_default_clock_and_group_duplicates_preserve_eviction_order(
    hooks: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hooks.time, "monotonic", lambda: 1.0)
    policy = hooks.LifecycleRetentionPolicy(total_blocks=8)
    wait = metadata(hooks, lifecycle="tool_wait", expected_resume_ms=100)
    policy.observe(event(hooks, "free", wait, ((1, 2), (1, 2))))
    assert policy.retained_count == 2
    assert policy.eviction_order([3, 1, 2]) == [3, 2, 1]
    policy.update(wait, [3])
    assert policy.protected_blocks() == frozenset({1, 2, 3})
    monkeypatch.setattr(hooks.time, "monotonic", lambda: 1.1)
    assert policy.protected_blocks() == frozenset()


@pytest.mark.parametrize("name", ["max_requests", "max_blocks", "ttl_ms"])
def test_prefix_directory_limits_must_be_positive(hooks: Any, name: str) -> None:
    with pytest.raises(ValueError, match="limits_must_be_positive"):
        hooks.CompletedRequestPrefixes(**{name: 0})


def update(hooks: Any, directory: Any, policy: Any, request: str = "r", **changes: Any) -> Any:
    args = {
        "sequence": 1,
        "lifecycle": "tool_wait",
        "expected_resume_ms": 10,
        "now_ms": 0,
        **changes,
    }
    return directory.update(
        request, "tenant", 2, 3, policy=policy, resident=lambda *args: True, **args
    )


def test_prefix_capture_requires_metadata_and_rejects_oversized_whole_prefix(hooks: Any) -> None:
    directory = hooks.CompletedRequestPrefixes(max_blocks=2)
    policy = hooks.LifecycleRetentionPolicy(total_blocks=8)
    directory.capture("r", None, (((1, b"a"),),), now_ms=0)
    with pytest.raises(ValueError, match="unknown"):
        update(hooks, directory, policy)
    directory.capture("r", metadata(hooks), (((1, b"a"),),), now_ms=0)
    directory.capture("r", metadata(hooks), (((1, b"a"), (2, b"b"), (3, b"c")),), now_ms=0)
    with pytest.raises(ValueError, match="unknown"):
        update(hooks, directory, policy)
    assert policy.retained_count == 0


def test_total_block_budget_evicts_oldest_request_without_truncating_newest(hooks: Any) -> None:
    directory = hooks.CompletedRequestPrefixes(max_requests=10, max_blocks=2)
    policy = hooks.LifecycleRetentionPolicy(total_blocks=8)
    directory.capture("old", metadata(hooks), (((1, b"a"),),), now_ms=0)
    directory.capture("new", metadata(hooks), (((2, b"b"), (3, b"c")),), now_ms=0)
    with pytest.raises(ValueError, match="unknown"):
        update(hooks, directory, policy, "old")
    receipt = update(hooks, directory, policy, "new")
    assert receipt["resident_blocks"] == 2


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"sequence": 0}, "positive"),
        ({"sequence": "1"}, "positive"),
        ({"sequence": 1.1}, "positive"),
        ({"expected_resume_ms": "10"}, "horizon"),
        ({"expected_resume_ms": 1.1}, "horizon"),
        ({"lifecycle": "unknown"}, "valid AgentLifecycle"),
    ],
)
def test_prefix_control_invalid_input_cannot_touch_policy(
    hooks: Any, changes: Any, reason: str
) -> None:
    directory = hooks.CompletedRequestPrefixes()
    policy = hooks.LifecycleRetentionPolicy(total_blocks=8)
    directory.capture("r", metadata(hooks), (((1, b"a"),),), now_ms=0)
    with pytest.raises(ValueError, match=reason):
        update(hooks, directory, policy, **changes)
    assert policy.retained_count == 0


def test_prefix_default_clock_is_used_without_extending_expiry(
    hooks: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hooks.time, "monotonic", lambda: 1.0)
    directory = hooks.CompletedRequestPrefixes(ttl_ms=100)
    policy = hooks.LifecycleRetentionPolicy(total_blocks=8)
    directory.capture("r", metadata(hooks), (((1, b"a"),),))
    receipt = update(hooks, directory, policy, now_ms=None)
    assert receipt["applied_at_ms"] == 1000
    monkeypatch.setattr(hooks.time, "monotonic", lambda: 1.1)
    with pytest.raises(ValueError, match="expired"):
        update(hooks, directory, policy, now_ms=None)


@pytest.mark.parametrize("kind", ["current", "no_metadata", "worker", "cache"])
def test_generation_fence_forwards_only_current_or_unscoped_events(hooks: Any, kind: str) -> None:
    recorded: list[Any] = []
    state = (
        None
        if kind == "no_metadata"
        else metadata(
            hooks,
            worker_generation=4 if kind == "worker" else 2,
            cache_generation=4 if kind == "cache" else 3,
        )
    )
    item = event(hooks, "hit", state)
    fenced = hooks.GenerationFencedHook(
        SimpleNamespace(on_cache_event=recorded.append), worker_generation=2, cache_generation=3
    )
    fenced.on_cache_event(item)
    stale = kind in {"worker", "cache"}
    assert recorded == ([] if stale else [item])
    assert fenced.stale_events == int(stale)


@pytest.mark.parametrize("path", ["module", ":factory", "module:"])
def test_plugin_path_requires_module_and_factory(
    hooks: Any, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setenv("VLLM_AGENT_CACHE_HOOK", path)
    with pytest.raises(ValueError, match="module:attribute"):
        hooks.load_agent_cache_hook()


@pytest.mark.parametrize("callable_handler", [False, True])
def test_plugin_factory_must_return_callable_event_handler(
    hooks: Any, monkeypatch: pytest.MonkeyPatch, callable_handler: bool
) -> None:
    plugin = ModuleType("_freechat_test_cache_plugin")
    recorded: list[Any] = []
    handler = SimpleNamespace(on_cache_event=recorded.append if callable_handler else None)
    plugin.__dict__["factory"] = lambda: handler
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    monkeypatch.setenv("VLLM_AGENT_CACHE_HOOK", plugin.__name__ + ":factory")
    if callable_handler:
        hook = hooks.load_agent_cache_hook()
        hook.on_cache_event("event")
        assert recorded == ["event"]
    else:
        with pytest.raises(TypeError, match="define on_cache_event"):
            hooks.load_agent_cache_hook()


def test_plugin_opt_out_uses_noop_without_imports(
    hooks: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_AGENT_CACHE_HOOK", raising=False)
    assert hooks.load_agent_cache_hook().on_cache_event(event(hooks, "hit", None)) is None
