"""CPU adapter-contract tests; no physical KV residency or GPU benefit is asserted."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from freechat_contracts.cache_lifecycle import CacheLifecycleCommand, CacheLifecycleUpdate
from freechat_contracts.execution import ExecutionCommand
from freechat_worker.vllm_execution import VllmExecutionBackend


class CacheEngine:
    """Expose the pinned AsyncLLM boundary, including its rewritten request ID."""

    def __init__(self) -> None:
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            kv_transfer_config=None,
            ec_transfer_config=None,
        )
        self.engine_core = self
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.aborts: list[str] = []
        self.failure: BaseException | None = None
        self.submit_failure: Exception | None = None
        self.receipt_changes: dict[str, Any] = {}

    async def add_request(self, request_id: str, prompt: Any, params: Any, **options: Any) -> Any:
        if self.submit_failure is not None:
            raise self.submit_failure

        async def get() -> Any:
            return SimpleNamespace(finished=True)

        return SimpleNamespace(request_id=f"{request_id}-internal", get=get)

    async def abort(self, request_id: str, *, internal: bool) -> None:
        assert internal is True
        self.aborts.append(request_id)

    async def call_utility_async(self, method: str, *args: Any) -> dict[str, Any]:
        self.calls.append((method, args))
        if self.failure is not None:
            raise self.failure
        return {
            "request_id": args[0],
            "sequence": args[4],
            "lifecycle": args[5],
            "resident_blocks": 3,
            "protected_blocks": 3 if args[5] == "tool_wait" else 0,
            "status": "applied",
            "applied_at_ms": 123,
            "replayed": False,
            **self.receipt_changes,
        }


def intent(lifecycle: str = "tool_wait") -> CacheLifecycleCommand:
    return CacheLifecycleCommand(
        owner=ExecutionCommand(
            tenant_id="tenant",
            request_id="external",
            decision_id="decision",
            worker_id="worker",
            worker_generation=7,
            engine_instance_id="engine",
            action="query",
        ),
        update=CacheLifecycleUpdate.model_validate(
            {
                "decision_id": "decision",
                "sequence": 2,
                "lifecycle": lifecycle,
                "expected_resume_ms": 500 if lifecycle == "tool_wait" else None,
            }
        ),
        cache_generation=7,
    )


@pytest.fixture
async def completed() -> AsyncIterator[tuple[CacheEngine, VllmExecutionBackend]]:
    engine = CacheEngine()
    backend = VllmExecutionBackend(engine)
    await backend.submit(
        "admitted-child", {"prompt": "text", "sampling_params": SimpleNamespace(n=1)}
    )
    async for output in backend.stream("admitted-child"):
        assert output.finished
    yield engine, backend


@pytest.mark.parametrize("lifecycle", ["tool_wait", "resume", "terminal", "cancelled"])
async def test_cache_utility_uses_internal_identity_and_bound_command(
    completed: tuple[CacheEngine, VllmExecutionBackend], lifecycle: str
) -> None:
    engine, backend = completed
    command = intent(lifecycle)
    receipt = await backend.cache_lifecycle("admitted-child", command)
    assert receipt.request_id == "admitted-child-internal"
    assert receipt.sequence == command.update.sequence
    assert receipt.lifecycle == lifecycle
    assert receipt.protected_blocks == (3 if lifecycle == "tool_wait" else 0)
    assert engine.calls == [
        (
            "freechat_update_cache_lifecycle",
            (
                "admitted-child-internal",
                "tenant",
                7,
                7,
                2,
                lifecycle,
                command.update.expected_resume_ms,
            ),
        )
    ]
    assert engine.aborts == []


@pytest.mark.parametrize(
    "reason",
    [
        "agent_cache_policy_disabled",
        "prefix_caching_disabled",
        "completed_prefix_unknown_or_expired",
        "completed_prefix_identity_mismatch",
        "lifecycle_sequence_conflict",
        "lifecycle_sequence_out_of_order",
        "completed_prefix_lifecycle_closed",
        "cache_lifecycle_requires_completed_request",
    ],
)
async def test_serialized_engine_rejections_become_stable_contract_errors(
    completed: tuple[CacheEngine, VllmExecutionBackend], reason: str
) -> None:
    engine, backend = completed
    failure = RuntimeError(f"Utility method failed: ValueError('{reason}')")
    engine.failure = failure
    with pytest.raises(ValueError, match=f"^{reason}$") as error:
        await backend.cache_lifecycle("admitted-child", intent())
    assert error.value.__cause__ is failure
    assert len(engine.calls) == 1
    assert engine.aborts == []


@pytest.mark.parametrize(
    "failure", [RuntimeError("engine transport lost"), asyncio.CancelledError()]
)
async def test_unrecognized_failure_or_cancellation_propagates_without_retry(
    completed: tuple[CacheEngine, VllmExecutionBackend], failure: BaseException
) -> None:
    engine, backend = completed
    engine.failure = failure
    with pytest.raises(type(failure)) as error:
        await backend.cache_lifecycle("admitted-child", intent())
    assert error.value is failure
    assert len(engine.calls) == 1
    assert engine.aborts == []


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "foreign-internal"},
        {"sequence": 3},
        {"lifecycle": "resume", "protected_blocks": 0},
    ],
)
async def test_engine_receipt_must_match_actual_request_sequence_and_lifecycle(
    completed: tuple[CacheEngine, VllmExecutionBackend], changes: dict[str, Any]
) -> None:
    engine, backend = completed
    engine.receipt_changes = changes
    with pytest.raises(ValueError, match=r"^cache_engine_receipt_identity_mismatch$"):
        await backend.cache_lifecycle("admitted-child", intent())
    assert len(engine.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [{"protected_blocks": 4}, {"resident_blocks": True}, {"status": "not_resident"}],
)
async def test_malformed_engine_residency_is_not_accepted(
    completed: tuple[CacheEngine, VllmExecutionBackend], changes: dict[str, Any]
) -> None:
    engine, backend = completed
    engine.receipt_changes = changes
    with pytest.raises(ValueError):
        await backend.cache_lifecycle("admitted-child", intent())
    assert len(engine.calls) == 1


@pytest.mark.parametrize("state", ["unknown", "unconfirmed", "active"])
async def test_incomplete_submission_cannot_issue_cache_utility(state: str) -> None:
    engine = CacheEngine()
    backend = VllmExecutionBackend(engine)
    if state == "unconfirmed":
        engine.submit_failure = RuntimeError("uncertain submit")
        with pytest.raises(RuntimeError, match="uncertain submit"):
            await backend.submit(
                "admitted-child", {"prompt": "text", "sampling_params": SimpleNamespace(n=1)}
            )
    elif state == "active":
        await backend.submit(
            "admitted-child", {"prompt": "text", "sampling_params": SimpleNamespace(n=1)}
        )
    reason = (
        "cache_lifecycle_requires_completed_request"
        if state == "active"
        else "execution_submission_unconfirmed"
    )
    with pytest.raises(ValueError, match=f"^{reason}$"):
        await backend.cache_lifecycle("admitted-child", intent())
    assert engine.calls == []
    assert engine.aborts == []


@pytest.mark.parametrize("lifecycle", ["tool_wait", "resume", "terminal", "cancelled"])
async def test_aborted_submission_only_allows_closing_cache_intent(lifecycle: str) -> None:
    engine = CacheEngine()
    backend = VllmExecutionBackend(engine)
    await backend.submit(
        "admitted-child", {"prompt": "text", "sampling_params": SimpleNamespace(n=1)}
    )
    await backend.abort("admitted-child")
    if lifecycle in {"tool_wait", "resume"}:
        with pytest.raises(ValueError, match=r"^cancelled_route_cannot_retain$"):
            await backend.cache_lifecycle("admitted-child", intent(lifecycle))
        assert engine.calls == []
    else:
        receipt = await backend.cache_lifecycle("admitted-child", intent(lifecycle))
        assert receipt.lifecycle == lifecycle
        assert receipt.protected_blocks == 0
        assert len(engine.calls) == 1
    assert engine.aborts == ["admitted-child-internal"]
