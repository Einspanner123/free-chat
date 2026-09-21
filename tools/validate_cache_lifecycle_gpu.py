"""Exercise completed-prefix lifecycle control on real CUDA; stdout only."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from typing import Any


async def validate(model: str, memory_fraction: float) -> dict[str, Any]:
    os.environ["VLLM_AGENT_CACHE_POLICY"] = "1"
    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    AsyncEngineArgs = importlib.import_module("vllm.engine.arg_utils").AsyncEngineArgs
    AsyncLLM = importlib.import_module("vllm.v1.engine.async_llm").AsyncLLM
    if not torch.cuda.is_available():
        raise RuntimeError("real_cuda_required")
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model,
            max_model_len=1024,
            num_gpu_blocks_override=128,
            gpu_memory_utilization=memory_fraction,
            enforce_eager=True,
            async_scheduling=False,
            max_num_seqs=4,
            enable_prefix_caching=True,
        )
    )
    base = {
        "tenant_id": "gpu-validation",
        "cache_key": "prefix-control",
        "task_id": "prefix-control",
        "session_id": "prefix-control",
        "agent_id": "agent",
        "branch_id": "main",
        "call_id": "initial",
        "lifecycle": "active",
        "worker_generation": 1,
        "cache_generation": 1,
    }
    results: dict[str, Any] = {}

    async def generate(name: str, count: int, salt: str) -> tuple[str, Any]:
        params = vllm.SamplingParams(
            max_tokens=8,
            temperature=0,
            extra_args={"agent_lifecycle": {**base, "call_id": name}},
        )
        collector = await engine.add_request(
            name,
            {"prompt_token_ids": [100 + i % 20 for i in range(count)], "cache_salt": salt},
            params,
        )
        while True:
            output = await collector.get()
            if output.finished:
                assert output.outputs[0].token_ids
                return str(collector.request_id), output

    async def update(
        request: str, seq: int, state: str, horizon: int | None, tenant: str = "gpu-validation"
    ) -> dict[str, Any]:
        value = await engine.engine_core.call_utility_async(
            "freechat_update_cache_lifecycle",
            request,
            tenant,
            1,
            1,
            seq,
            state,
            horizon,
        )
        if not isinstance(value, dict) or value.get("request_id") != request:
            raise AssertionError("cache_lifecycle_receipt_identity_mismatch")
        return value

    async def reject(
        request: str,
        seq: int,
        state: str,
        horizon: int | None,
        reason: str,
        tenant: str = "gpu-validation",
    ) -> None:
        try:
            await update(request, seq, state, horizon, tenant)
        except Exception as error:
            if reason not in str(error):
                raise
        else:
            raise AssertionError("invalid_cache_update_was_accepted")

    try:
        async with asyncio.timeout(180):
            request, first = await generate("initial", 256, "target")
            await reject(request, 1, "tool_wait", 30000, "identity_mismatch", "other")
            wait = await update(request, 1, "tool_wait", 30000)
            assert wait["status"] == "applied" and wait["protected_blocks"] > 0
            for index in range(6):
                await generate(f"protected-pressure-{index}", 896, f"protected-{index}")
            duplicate = await update(request, 1, "tool_wait", 30000)
            assert duplicate == {**wait, "replayed": True}
            await reject(request, 1, "resume", None, "sequence_conflict")
            resume = await update(request, 2, "resume", None)
            assert resume["protected_blocks"] == 0
            await reject(request, 3, "tool_wait", 30000, "lifecycle_closed")
            second_request, second = await generate("resume", 256, "target")
            assert second.num_cached_tokens > 0
            assert second.outputs[0].token_ids == first.outputs[0].token_ids
            for index in range(6):
                await generate(f"pressure-{index}", 896, f"pressure-{index}")
            recycled = await update(second_request, 1, "tool_wait", 30000)
            assert recycled["status"] == "not_resident"
            assert recycled["resident_blocks"] == recycled["protected_blocks"] == 0
            reset_request, _ = await generate("before-reset", 256, "reset")
            assert await engine.reset_prefix_cache()
            await reject(reset_request, 1, "tool_wait", 30000, "unknown_or_expired")
            active = await engine.add_request(
                "active",
                {"prompt_token_ids": [100] * 256, "cache_salt": "active"},
                vllm.SamplingParams(
                    max_tokens=768,
                    temperature=0,
                    ignore_eos=True,
                    extra_args={"agent_lifecycle": {**base, "call_id": "active"}},
                ),
            )
            first_token = await active.get()
            assert not first_token.finished and first_token.outputs[0].token_ids
            active_id = str(active.request_id)
            await reject(active_id, 1, "tool_wait", 30000, "requires_completed_request")
            await engine.abort(active_id, internal=True)
            async with asyncio.timeout(10):
                while True:
                    quiet = await engine.engine_core.call_utility_async(
                        "freechat_execution_quiescent",
                        active_id,
                    )
                    if quiet["quiescent"]:
                        break
                    await asyncio.sleep(0.01)
            cancelled = await update(active_id, 1, "cancelled", None)
            assert cancelled["protected_blocks"] == 0
            results = {
                "inflight_update_rejected": True,
                "aborted_prefix_unprotected": True,
                "protected_pressure_requests": 6,
                "wait": wait,
                "resume": resume,
                "recycled": recycled,
                "resume_cached_tokens": second.num_cached_tokens,
                "generated_tokens_equal": True,
                "duplicate_idempotent": True,
                "tenant_rejected": True,
                "sequence_conflict_rejected": True,
                "late_wait_rejected": True,
                "reset_discards_lineage": True,
            }
        torch.cuda.synchronize()
        return {
            "scope": "REAL_GPU_COMPLETED_PREFIX_CONTROL",
            "accepted": True,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": model,
            "results": results,
            "gateway_harness_transport_verified": False,
            "performance_gain_claimed": False,
        }
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    print(json.dumps(asyncio.run(validate(args.model, args.gpu_memory_utilization)), indent=2))


if __name__ == "__main__":
    main()
