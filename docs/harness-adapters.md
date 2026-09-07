# Harness adapter boundary

FreeChat does not own an Agent loop. Harness adapters translate stable runtime
identity and lifecycle events into the common `freechat.agent_hints` request
extension; the Gateway authenticates the tenant and validates the extension,
and the Scheduler retains final resource authority.

## Supported contract depth

| Harness | Implemented boundary | Verified scope | Remaining acceptance |
|---|---|---|---|
| OpenAI Agents SDK | Run hooks plus model decorator | Real `Runner.run()` tool call traversed the Gateway and Qwen2.5-0.5B Worker; the second model request carried Resume identity | Cancellation/failure replay, task-quality gate and benchmark matrix |
| LangGraph | Runnable/checkpoint context bridge | Real interrupt and `Command(resume=...)` preserve task, thread and checkpoint identity | Gateway, real model, branching/failure replay and benchmark matrix |
| OpenCode | Native session-event state machine | ToolPart pending/running/completed/error, parallel calls, replay regression and cross-session rejection | Live plugin/SSE run, model request interception, cancellation/failure replay and benchmark matrix |
| OpenHands | SDK event state machine | Action/Observation/error pairing, parallel calls and late-event replay protection | Live SDK callback, model request interception, cancellation/failure replay and benchmark matrix |

The two event bridges are dependency-light on purpose. They accept native event
dictionaries at the transport boundary and do not import private framework
internals. This makes recorded trace replay deterministic and keeps framework
upgrades from entering the Scheduler contract. It does not substitute for live
Harness acceptance.

## OpenCode bridge

Create one bridge for one session, feed it `message.part.updated` and session
events, then apply the current state immediately before the model request:

```python
from freechat_harness_adapters import OpenCodeLifecycle

bridge = OpenCodeLifecycle(session_id="ses_123", task_id="task_123")
bridge.on_event(event)
request_body = bridge.apply(request_body)
```

The bridge keys tool state by `callID` and records `messageID` as the turn.
Pending and running calls produce Tool Wait. Completed and failed calls produce
Resume only after every parallel call is terminal. A late pending/running update
cannot reopen a completed call.

## OpenHands bridge

Create one bridge for one conversation and feed events from the conversation
callback before applying hints to the next model request:

```python
from freechat_harness_adapters import OpenHandsLifecycle

bridge = OpenHandsLifecycle(conversation_id="conversation-123", task_id="task-123")
bridge.on_event(event)
request_body = bridge.apply(request_body)
```

Action events enter Tool Wait. Observation, user-rejection and agent-error
events close their matching `tool_call_id`; Resume begins only when no parallel
call remains. Conversation errors and interrupts enter Cancelled. Terminal and
Cancelled states are sticky so delayed replay cannot resurrect a task.

## Evidence rule

Unit or contract tests may verify identity preservation and transition logic.
Only a real Harness process, real Gateway/worker request, archived trace and
protocol-matched failure matrix may satisfy `harness-integrations` in the
Claims Ledger. No event-contract result is a latency, cache-hit or throughput
claim.

The 2026-09-07 OpenAI Agents run executed `read_file("README.md")` and recorded
paired Active and Resume model requests followed by Terminal. The 0.5B model's
final answer did not match the expected README heading, so this run validates
the lifecycle path but explicitly fails the task-quality gate. The archived
record is `evidence/harnesses/openai-agents-qwen05b-20260907.json`; it must not
be used as performance or model-routing acceptance.
