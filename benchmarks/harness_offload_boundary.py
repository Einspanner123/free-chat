"""Real Harness boundary experiment: native LRU versus pre-offload envelope.

This deliberately bypasses FreeChat routing; it cannot validate its predictor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import uuid4

import httpx
from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from freechat_worker.telemetry import samples
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from benchmarks.records import emit_record

NAMES = {
    "vllm:request_prompt_tokens_count",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:kv_offload_store_size_sum",
    "vllm:kv_offload_load_size_sum",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
}


class GraphState(TypedDict, total=False):
    messages: list[dict[str, Any]]


class ProbeTransport(httpx.AsyncBaseTransport):
    def __init__(self, endpoint: str, model: str, record_prefix: str, enabled: bool) -> None:
        self.transport = httpx.AsyncHTTPTransport(retries=0)
        self.monitor = httpx.AsyncClient(base_url=endpoint, trust_env=False, timeout=120)
        self.model = model
        self.record_prefix = record_prefix
        self.enabled = enabled
        self.salt = str(uuid4())
        self.calls: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, str]] = []
        self.instrumentation_seconds = 0.0

    async def snapshot(self) -> tuple[str, dict[str, float]]:
        response = await self.monitor.get("/metrics")
        response.raise_for_status()
        return response.text, samples(response.text, self.model, "0", names=NAMES)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        begin = time.perf_counter()
        before_raw, before = await self.snapshot()
        body = json.loads(request.content)
        body.pop("freechat", None)
        body["cache_salt"] = self.salt
        body["kv_transfer_params"] = {"max_offload_tokens": 4096 if self.enabled else 0}
        # The benchmark defines exactly one tool step, then a final answer.
        if self.calls:
            body["tool_choice"] = "none"
        forwarded = httpx.Request(
            request.method,
            request.url,
            json=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "content-length"},
        )
        dispatch = time.perf_counter()
        response = await self.transport.handle_async_request(forwarded)
        await response.aread()
        finished = time.perf_counter()
        payload = json.loads(response.content)
        if response.status_code != 200:
            raise RuntimeError(f"worker status {response.status_code}: {payload}")
        for _ in range(100):
            after_raw, after = await self.snapshot()
            if after.get("vllm:request_prompt_tokens_count", 0) > before.get(
                "vllm:request_prompt_tokens_count", 0
            ):
                break
            await asyncio.sleep(0.1)
        delta = {name: after.get(name, 0) - before.get(name, 0) for name in NAMES}
        if delta["vllm:request_prompt_tokens_count"] != 1 or any(v < 0 for v in delta.values()):
            raise RuntimeError("non-isolated or reset measurement window")
        if delta["vllm:num_preemptions_total"] != 0:
            raise RuntimeError("preemption contaminated trial")
        index = len(self.calls)
        for suffix, raw in (
            ("before.prom", before_raw),
            ("after.prom", after_raw),
            ("request.json", json.dumps(body)),
            ("response.json", json.dumps(payload)),
        ):
            self.artifacts.append(emit_record(f"{self.record_prefix}/call{index}-{suffix}", raw))
        self.calls.append(
            {
                "request_ms": (finished - dispatch) * 1000,
                "deltas": delta,
                "usage": payload.get("usage"),
                "response": payload,
            }
        )
        self.instrumentation_seconds += (dispatch - begin) + (time.perf_counter() - finished)
        return response

    async def pressure(self) -> None:
        for _ in range(8):
            response = await self.monitor.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": "Independent cache pressure data. " * 420}
                    ],
                    "cache_salt": str(uuid4()),
                    "max_tokens": 1,
                    "temperature": 0,
                    "kv_transfer_params": {"max_offload_tokens": 0},
                },
            )
            response.raise_for_status()
        await asyncio.sleep(0.2)

    async def aclose(self) -> None:
        await self.transport.aclose()
        await self.monitor.aclose()


async def trial(
    args: argparse.Namespace, harness: str, scenario: str, enabled: bool, record_prefix: str
) -> dict[str, Any]:
    probe = ProbeTransport(args.endpoint, args.model, record_prefix, enabled)
    client = AsyncOpenAI(
        api_key="local-probe",
        base_url=args.endpoint + "/v1",
        max_retries=0,
        http_client=cast(Any, httpx.AsyncClient(transport=probe, timeout=120)),
    )
    tool_calls: list[str] = []
    instructions = (
        "You must call read_heading exactly once, then return exactly the heading it returns. "
        "Do not call any tool after it returns. "
        + "Repository context: a Python inference infrastructure project. "
        * 100
    )
    tool_result = "# FreeChat"

    async def read_heading_impl() -> str:
        tool_calls.append("read_heading")
        if scenario == "pressure":
            await probe.pressure()
        else:
            await asyncio.sleep(0.2)
        heading = next(
            line for line in args.readme.read_text().splitlines() if line.startswith("# ")
        )
        return str(heading)

    tool_result = next(
        line for line in args.readme.read_text().splitlines() if line.startswith("# ")
    )
    started = time.perf_counter()
    try:
        if harness == "agents":

            @function_tool
            async def read_heading() -> str:
                """Read the repository's first Markdown heading."""
                return await read_heading_impl()

            agent = Agent(
                name="reader",
                instructions=instructions,
                tools=[read_heading],
                model=OpenAIChatCompletionsModel(model=args.model, openai_client=client),
                model_settings=ModelSettings(
                    temperature=0, max_tokens=64, tool_choice="read_heading"
                ),
            )
            result = await Runner.run(
                agent,
                "Read the repository heading.",
                max_turns=3,
                run_config=RunConfig(tracing_disabled=True),
            )
            answer = str(result.final_output)
        else:

            async def model_node(state: GraphState) -> GraphState:
                messages = state.get(
                    "messages",
                    [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": "Read the repository heading."},
                    ],
                )
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    temperature=0,
                    max_tokens=64,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "read_heading",
                                "description": "Read the repository first Markdown heading.",
                                "parameters": {"type": "object", "properties": {}, "required": []},
                            },
                        }
                    ],
                    tool_choice={"type": "function", "function": {"name": "read_heading"}},
                )
                message = response.choices[0].message
                return {"messages": [*messages, message.model_dump(exclude_none=True)]}

            async def tool_node(state: GraphState) -> GraphState:
                messages = state["messages"]
                calls = messages[-1].get("tool_calls", [])
                if len(calls) != 1 or calls[0]["function"]["name"] != "read_heading":
                    raise RuntimeError("unexpected graph tool calls")
                value = await read_heading_impl()
                return {
                    "messages": [
                        *messages,
                        {"role": "tool", "tool_call_id": calls[0]["id"], "content": value},
                    ]
                }

            graph = StateGraph(GraphState)
            graph.add_node("first", model_node)
            graph.add_node("tool", tool_node)
            graph.add_node("resume", model_node)
            graph.add_edge(START, "first")
            graph.add_edge("first", "tool")
            graph.add_edge("tool", "resume")
            graph.add_edge("resume", END)
            result_state = await graph.compile().ainvoke({})
            answer = str(result_state["messages"][-1].get("content", ""))
        elapsed = time.perf_counter() - started
        record = {
            "harness": harness,
            "scenario": scenario,
            "preoffload": enabled,
            "calls": probe.calls,
            "tool_calls": tool_calls,
            "answer": answer,
            "expected": tool_result,
            "correct": answer.strip() == tool_result,
            "semantic_heading_present": tool_result.removeprefix("# ") in answer,
            "raw_task_ms": elapsed * 1000,
            "adjusted_task_ms": (elapsed - probe.instrumentation_seconds) * 1000,
            "instrumentation_ms": probe.instrumentation_seconds * 1000,
            "artifacts": probe.artifacts,
            "scope": "Forced one-tool workflow; routing bypassed; not calibrated A or B.",
        }
        if len(probe.calls) != 2 or tool_calls != ["read_heading"]:
            raise RuntimeError("incomplete two-call workflow")
        emit_record(f"{record_prefix}/result.json", json.dumps(record, indent=2))
        return record
    finally:
        await client.close()


async def run(args: argparse.Namespace) -> None:
    records = []
    for harness in args.harnesses:
        for scenario in args.scenarios:
            for repeat in range(-1, args.pairs):
                for enabled in [False, True] if repeat % 2 == 0 else [True, False]:
                    name = f"{harness}-{scenario}-{repeat}-{int(enabled)}"
                    record = await trial(args, harness, scenario, enabled, name)
                    records.append(
                        {"name": name, "warmup": repeat < 0, "correct": record["correct"]}
                    )
                    print(
                        json.dumps(
                            {
                                "name": name,
                                "correct": record["correct"],
                                "adjusted_ms": record["adjusted_task_ms"],
                            }
                        ),
                        flush=True,
                    )
    emit_record("index.json", json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://100.99.131.26:18001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument(
        "--harnesses", nargs="+", choices=["agents", "langgraph"], default=["agents", "langgraph"]
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["pressure", "no-pressure"],
        default=["pressure", "no-pressure"],
    )
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
