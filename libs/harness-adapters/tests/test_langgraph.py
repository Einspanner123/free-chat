from __future__ import annotations

from typing import Any, TypedDict

import pytest
from freechat_contracts import Lifecycle
from freechat_harness_adapters.langgraph import (
    LangGraphRequestContext,
    bind_langgraph_task,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    result: str


def test_real_graph_interrupt_and_resume_keep_checkpoint_identity() -> None:
    requests: list[dict[str, Any]] = []

    def review(state: ReviewState, config: RunnableConfig) -> ReviewState:
        del state
        answer = interrupt("approve")
        request_context = LangGraphRequestContext.from_config(config)
        requests.append(
            request_context.apply(
                {"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": []},
                lifecycle=Lifecycle.RESUME,
                expected_resume_ms=500,
            )
        )
        return {"result": str(answer)}

    builder = StateGraph(ReviewState)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    initial_config = {
        "configurable": {"thread_id": "thread-1", "freechat_task_id": "task-1"}
    }

    interrupted = graph.invoke({"result": ""}, initial_config)
    assert "__interrupt__" in interrupted
    snapshot = graph.get_state(initial_config)
    resume_config = bind_langgraph_task(snapshot.config, task_id="task-1")
    wait_context = LangGraphRequestContext.from_config(resume_config, node="review")
    wait_body = wait_context.apply(
        {"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": []},
        lifecycle=Lifecycle.TOOL_WAIT,
        expected_resume_ms=500,
    )
    completed = graph.invoke(Command(resume="approved"), resume_config)

    assert completed["result"] == "approved"
    wait_hints = wait_body["freechat"]["agent_hints"]
    resume_hints = requests[0]["freechat"]["agent_hints"]
    assert wait_hints["session_id"] == resume_hints["session_id"] == "thread-1"
    assert wait_hints["task_id"] == resume_hints["task_id"] == "task-1"
    assert wait_hints["turn_id"] == resume_hints["turn_id"]
    assert wait_hints["lifecycle"] == "tool_wait"
    assert resume_hints["lifecycle"] == "resume"


def test_thread_id_is_required() -> None:
    with pytest.raises(ValueError, match=r"configurable\.thread_id"):
        LangGraphRequestContext.from_config({})
