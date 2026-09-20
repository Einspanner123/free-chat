"""Real-weight, real-CUDA admission/completion/cancellation probe (not API acceptance)."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any

from freechat_contracts.execution import ExecutionAction, ExecutionCommand
from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.vllm_execution import VllmExecutionBackend


async def validate(model: str, repeats: int) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    SamplingParams = importlib.import_module("vllm").SamplingParams
    AsyncEngineArgs = importlib.import_module("vllm.engine.arg_utils").AsyncEngineArgs
    AsyncLLM = importlib.import_module("vllm.v1.engine.async_llm").AsyncLLM

    if not torch.cuda.is_available():
        raise RuntimeError("real_cuda_required")
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model,
            max_model_len=2048,
            gpu_memory_utilization=0.2,
            enforce_eager=True,
            async_scheduling=False,
            max_num_seqs=8,
        )
    )
    results: list[dict[str, Any]] = []
    try:
        backend = VllmExecutionBackend(engine)
        with tempfile.TemporaryDirectory(prefix="freechat-execution-gpu-") as directory:
            gate = DurableExecutionDriver(
                Path(directory) / "journal.db",
                backend,
                worker_id="gpu-probe",
                generation=1,
                engine_instance_id=directory,
                create=True,
            )
            try:
                for index in range(repeats + 2):
                    cancel = index >= repeats
                    midstream = index == repeats + 1
                    command = ExecutionCommand(
                        tenant_id="gpu-validation",
                        request_id=str(index),
                        decision_id=str(index),
                        worker_id="gpu-probe",
                        worker_generation=1,
                        engine_instance_id=directory,
                        action=ExecutionAction.QUERY,
                    )
                    key = await gate.admit(
                        command,
                        {
                            "prompt": "Tell me about GPU memory.",
                            "sampling_params": SamplingParams(
                                max_tokens=512 if cancel else 8,
                                temperature=0,
                                ignore_eos=cancel,
                            ),
                        },
                    )
                    tokens = 0
                    if cancel:
                        stream = None
                        if midstream:
                            stream = backend.stream(key)
                            first = await anext(stream)
                            tokens = len(first.outputs[0].token_ids)
                            if tokens < 1 or first.finished:
                                raise AssertionError("must_cancel_during_real_generation")
                        receipt = await gate.observe(
                            command.model_copy(update={"action": ExecutionAction.ABORT})
                        )
                        if stream is not None:
                            await stream.aclose()
                    else:
                        async for output in backend.stream(key):
                            tokens = len(output.outputs[0].token_ids)
                        receipt = await gate.observe(command)
                    if not receipt.releasable:
                        raise AssertionError("engine_terminal_receipt_not_releasable")
                    try:
                        await gate.admit(command, {})
                    except ValueError as error:
                        if str(error) != "execution_admission_closed":
                            raise
                    else:
                        raise AssertionError("duplicate_admission_was_accepted")
                    results.append(
                        {
                            "request": index,
                            "cancel": cancel,
                            "midstream": midstream,
                            "tokens": tokens,
                            "status": receipt.status.value,
                            "releasable": receipt.releasable,
                        }
                    )
            finally:
                gate.close()
        torch.cuda.synchronize()
        return {
            "scope": "REAL_GPU_EXECUTION_ADAPTER",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": model,
            "results": results,
            "full_gateway_scheduler_api_acceptance": False,
        }
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps(asyncio.run(validate(args.model, args.repeats)), indent=2))


if __name__ == "__main__":
    main()
