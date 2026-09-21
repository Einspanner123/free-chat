"""Pinned AsyncLLM execution adapter; first slice is synchronous single-GPU text."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from freechat_contracts.cache_lifecycle import CacheLifecycleCommand, PrefixLifecycleReceipt
from freechat_contracts.execution import ExecutionStatus

from freechat_worker.execution import EngineObservation


@dataclass
class _Submission:
    collector: Any = None
    submitted: bool = False
    finished: bool = False
    abort_requested: bool = False
    stream_claimed: bool = False


class VllmExecutionBackend:
    """One adapter owns one AsyncLLM incarnation; never reconnect to another engine.

    Unknown/uncertain submits cannot be retried. GPU quiescence is obtained through
    the pinned EngineCore utility, not from HTTP or the output iterator ending.
    """

    def __init__(self, engine: Any) -> None:
        config = engine.vllm_config
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or config.scheduler_config.async_scheduling
            or config.kv_transfer_config is not None
            or config.ec_transfer_config is not None
        ):
            raise ValueError("execution_adapter_requires_sync_single_gpu")
        self.engine = engine
        self._requests: dict[str, _Submission] = {}

    async def submit(self, engine_request_id: str, payload: Mapping[str, Any]) -> None:
        if engine_request_id in self._requests:
            raise ValueError("execution_duplicate_backend_submit")
        if set(payload) not in (
            {"prompt", "sampling_params"},
            {"prompt", "sampling_params", "options"},
        ):
            raise ValueError("execution_payload_requires_prompt_and_sampling_params")
        params = payload["sampling_params"]
        prompt = payload["prompt"]
        if getattr(params, "n", None) != 1 or not isinstance(prompt, (str, dict)):
            raise ValueError("execution_adapter_requires_single_text_request")
        if isinstance(prompt, dict):
            allowed = {
                "type",
                "prompt_token_ids",
                "prompt",
                "cache_salt",
                "arrival_time",
                "prompt_token_offsets",
                "assistant_tokens_mask",
            }
            if (
                "prompt_token_ids" not in prompt
                or set(prompt) - allowed
                or prompt.get("type", "token") != "token"
            ):
                raise ValueError("execution_adapter_requires_text_tokens")
        options = payload.get("options", {})
        if not isinstance(options, dict) or set(options) - {
            "prompt_text",
            "lora_request",
            "tokenization_kwargs",
            "trace_headers",
            "priority",
            "data_parallel_rank",
            "session_id",
            "reasoning_ended",
            "reasoning_parser_kwargs",
        }:
            raise ValueError("execution_generate_options_invalid")
        # Install identity BEFORE awaiting preprocessing/engine submission.
        state = self._requests[engine_request_id] = _Submission()
        state.collector = await self.engine.add_request(
            engine_request_id, prompt, params, **options
        )
        state.submitted = True

    def _state(self, engine_request_id: str) -> _Submission:
        state = self._requests.get(engine_request_id)
        if state is None or not state.submitted:
            raise ValueError("execution_submission_unconfirmed")
        return state

    async def stream(self, engine_request_id: str) -> AsyncGenerator[Any, None]:
        state = self._state(engine_request_id)
        if state.stream_claimed:
            raise ValueError("execution_stream_already_claimed")
        state.stream_claimed = True
        try:
            while not state.abort_requested:
                try:
                    output = await asyncio.wait_for(state.collector.get(), timeout=0.1)
                except TimeoutError:
                    continue
                state.finished = bool(output.finished)
                yield output
                if state.finished:
                    return
        finally:
            if not state.finished:
                await self.abort(engine_request_id)

    async def abort(self, engine_request_id: str) -> None:
        state = self._state(engine_request_id)
        state.abort_requested = True
        # AsyncLLM has replaced the external identity with an internal request ID.
        await self.engine.abort(state.collector.request_id, internal=True)

    async def cache_lifecycle(
        self,
        engine_request_id: str,
        command: CacheLifecycleCommand,
    ) -> PrefixLifecycleReceipt:
        state = self._state(engine_request_id)
        update = command.update
        if not state.finished and not state.abort_requested:
            raise ValueError("cache_lifecycle_requires_completed_request")
        if state.abort_requested and update.lifecycle in {"tool_wait", "resume"}:
            raise ValueError("cancelled_route_cannot_retain")
        request_id = state.collector.request_id
        try:
            value = await self.engine.engine_core.call_utility_async(
                "freechat_update_cache_lifecycle",
                request_id,
                command.owner.tenant_id,
                command.owner.worker_generation,
                command.cache_generation,
                update.sequence,
                update.lifecycle,
                update.expected_resume_ms,
            )
        except Exception as error:
            # Upstream utility RPC serializes failure text rather than exception types.
            reasons = (
                "agent_cache_policy_disabled",
                "prefix_caching_disabled",
                "completed_prefix_unknown_or_expired",
                "completed_prefix_identity_mismatch",
                "lifecycle_sequence_conflict",
                "lifecycle_sequence_out_of_order",
                "completed_prefix_lifecycle_closed",
                "cache_lifecycle_requires_completed_request",
            )
            for reason in reasons:
                if reason in str(error):
                    raise ValueError(reason) from error
            raise
        receipt = PrefixLifecycleReceipt.model_validate(value)
        if (
            receipt.request_id != request_id
            or receipt.sequence != update.sequence
            or receipt.lifecycle != update.lifecycle
        ):
            raise ValueError("cache_engine_receipt_identity_mismatch")
        return receipt

    async def query(self, engine_request_id: str) -> EngineObservation:
        state = self._requests.get(engine_request_id)
        if state is None or not state.submitted:
            status, quiet = ExecutionStatus.UNKNOWN, False
        elif not state.finished and not state.abort_requested:
            status, quiet = ExecutionStatus.RUNNING, False
        else:
            request_id = state.collector.request_id
            snapshot = await self.engine.engine_core.call_utility_async(
                "freechat_execution_quiescent", request_id
            )
            if not isinstance(snapshot, dict) or snapshot.get("request_id") != request_id:
                raise ValueError("execution_engine_snapshot_identity_mismatch")
            quiet = snapshot.get("quiescent") is True
            status = (
                (ExecutionStatus.COMPLETED if state.finished else ExecutionStatus.ABORTED)
                if quiet
                else ExecutionStatus.RUNNING
            )
        return EngineObservation(
            engine_request_id=engine_request_id,
            observed_at=datetime.now(UTC),
            status=status,
            quiescent=quiet,
            submission_fenced=bool(state and state.submitted),
        )
