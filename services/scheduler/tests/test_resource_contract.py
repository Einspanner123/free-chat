import asyncio
import json
import time

import grpc
import httpx
import pytest
from freechat.control.v1 import control_pb2_grpc
from freechat_contracts import AgentHints
from freechat_contracts.preparation import PreparedAdmission, TokenBudget, body_digest
from freechat_gateway import GatewayConfig, create_app
from freechat_gateway.routing import GrpcSchedulerClient
from freechat_scheduler.grpc_server import LeaseBook, SchedulerGrpcService
from freechat_scheduler.preparation import NativePreparer
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.resources import required_kv_bytes_per_rank
from freechat_scheduler.scheduler import NoEligibleWorker, Scheduler
from test_request_ledger import confirm_execution
from test_scheduler import MODEL, add_worker, profile


def contract_preparer() -> NativePreparer:
    """Deterministic renderer fixture, not tokenizer or GPU evidence."""

    def handler(message: httpx.Request) -> httpx.Response:
        payload = json.loads(message.content)
        body = payload["request"]
        result = PreparedAdmission(
            preparation_id="a" * 64,
            worker_id="worker",
            generation=1,
            engine_instance_id="fixture-engine",
            budget=TokenBudget(
                tenant_id=message.headers["x-freechat-internal-tenant"],
                protocol=payload["protocol"],
                body_sha256=body_digest(body),
                prompt_sha256="b" * 64,
                input_tokens=1,
                output_tokens=body["max_tokens"],
                expires_at=time.time() + 60,
            ),
        )
        return httpx.Response(200, json=result.model_dump())

    return NativePreparer("t" * 32, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("tokens,expected", [(1, 16), (16, 16), (17, 32), (31, 32)])
def test_geometry_aligns_blocks_includes_output_and_never_divides_by_tp(
    tp: int, tokens: int, expected: int
) -> None:
    model = MODEL.model_copy(update={"tensor_parallel_size": tp})
    request = profile(input_tokens=0, output_tokens=tokens, estimated_kv_bytes=1)
    assert required_kv_bytes_per_rank(request, model) == expected * 16_384


@pytest.mark.parametrize(
    "missing", ["kv_admission_bytes_per_token_per_rank", "kv_block_size_tokens"]
)
def test_unknown_geometry_is_not_zero(missing: str) -> None:
    assert required_kv_bytes_per_rank(profile(), MODEL.model_copy(update={missing: None})) is None


@pytest.mark.parametrize(
    "case,reason",
    [
        ("unknown_geometry", "kv_geometry_unknown"),
        ("unknown_budget", "kv_admission_budget_unknown"),
        ("insufficient_budget", "vram_capacity"),
        ("unknown_locality", "request_locality_unknown"),
        ("remote_denied", "remote_worker_forbidden"),
    ],
)
def test_admission_rejects_unknown_or_insufficient_resources(case: str, reason: str) -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    snapshot = registry.snapshot()[1][0]
    caps, telemetry = snapshot.capabilities, snapshot.telemetry
    request = profile(estimated_kv_bytes=0)
    if case == "unknown_geometry":
        caps = caps.model_copy(
            update={"models": (MODEL.model_copy(update={"kv_block_size_tokens": None}),)}
        )
    elif case in {"unknown_budget", "insufficient_budget"}:
        telemetry = telemetry.model_copy(
            update={
                "kv_admission_available_bytes_per_rank": None if case == "unknown_budget" else 1
            }
        )
    else:
        request = request.model_copy(
            update={
                "local_node_id": None if case == "unknown_locality" else "elsewhere",
                "hints": request.hints.model_copy(update={"allow_remote_worker": False}),
            }
        )
    registry.upsert(caps, telemetry)
    with pytest.raises(NoEligibleWorker) as error:
        Scheduler(registry).route(request)
    assert reason in error.value.rejected["worker"]


def test_unknown_locality_for_permissive_request_has_explicit_cost_fallback() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    result = Scheduler(registry).route(profile(local_node_id=None, estimated_kv_bytes=0))
    assert result.strategy == "least-load"
    assert not result.selected.estimate_available
    assert result.selected.unavailable_reason == "request_locality_unknown"


def test_known_zero_budget_and_worker_locality_restriction() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    snapshot = registry.snapshot()[1][0]
    registry.upsert(
        snapshot.capabilities.model_copy(update={"allow_remote_requests": False}),
        snapshot.telemetry.model_copy(update={"kv_admission_available_bytes_per_rank": 0}),
    )
    with pytest.raises(NoEligibleWorker) as error:
        Scheduler(registry).route(profile(local_node_id=None))
    reasons = error.value.rejected["worker"]
    assert "vram_capacity" in reasons
    assert "request_locality_unknown" in reasons
    assert "kv_admission_budget_unknown" not in reasons


@pytest.mark.parametrize("origin", ["", " ", " ross"])
def test_gateway_origin_configuration_rejects_ambiguous_identity(origin: str) -> None:
    with pytest.raises(ValueError, match="origin_node_id"):
        GatewayConfig(
            api_keys={"tenant": "key"}, cache_salt_secret=b"s" * 32, origin_node_id=origin
        )


async def test_http_gateway_grpc_scheduler_preserve_trusted_locality_and_enforce_memory() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross")
    service = SchedulerGrpcService(Scheduler(registry), LeaseBook(), preparer=contract_preparer())
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(service, server)  # type: ignore[no-untyped-call]
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    forwarded: list[dict[str, object]] = []

    def worker(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-freechat-internal-preparation"] == "a" * 64
        assert int(request.headers["x-freechat-internal-reserved-kv-bytes"]) == 48 * 16384
        forwarded.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "synthetic"})

    scheduler_client = GrpcSchedulerClient(f"127.0.0.1:{port}")
    app = create_app(
        GatewayConfig(
            api_keys={"tenant": "key"}, cache_salt_secret=b"s" * 32, origin_node_id="ross"
        ),
        scheduler=scheduler_client,
        transport=httpx.MockTransport(worker),
    )
    hints = AgentHints(
        harness_id="test",
        task_id="task",
        session_id="session",
        agent_id="agent",
        branch_id="branch-a",
        call_id="call",
        prefix_scope="branch",
        allow_remote_worker=False,
    )
    body = {
        "model": MODEL.model_id,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "freechat": {"agent_hints": hints.model_dump(mode="json")},
    }
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway"
            ) as client,
        ):
            headers = {"authorization": "Bearer key", "x-freechat-local-node-id": "spoofed"}
            response = await client.post("/v1/chat/completions", headers=headers, json=body)
            assert response.status_code == 200
            hints.branch_id = "branch-b"
            body["freechat"] = {"agent_hints": hints.model_dump(mode="json")}
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=body)
            ).status_code == 200
            assert forwarded[0]["cache_salt"] != forwarded[1]["cache_salt"]
            # Missing client KV estimate cannot turn a positive request into zero demand.
            snapshot = registry.snapshot()[1][0]
            registry.upsert(
                snapshot.capabilities,
                snapshot.telemetry.model_copy(
                    update={"kv_admission_available_bytes_per_rank": 1},
                ),
            )
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=body)
            ).status_code == 503
            assert len(forwarded) == 2
            # Locality restriction also crosses the actual RPC boundary.
            registry.upsert(
                snapshot.capabilities.model_copy(update={"node_id": "remote"}),
                snapshot.telemetry,
            )
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=body)
            ).status_code == 503
            assert len(forwarded) == 2
    finally:
        await server.stop(None)


async def test_two_http_calls_compete_for_one_reservation_then_capacity_returns() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker", node="ross", free_vram_bytes=32 * 16_384)
    ledger = LeaseBook()
    server = grpc.aio.server()
    control_pb2_grpc.add_SchedulerServiceServicer_to_server(  # type: ignore[no-untyped-call]
        SchedulerGrpcService(Scheduler(registry), ledger, preparer=contract_preparer()),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    entered, complete = asyncio.Event(), asyncio.Event()

    async def worker(_: httpx.Request) -> httpx.Response:
        entered.set()
        await complete.wait()
        return httpx.Response(200, json={"id": "synthetic"})

    app = create_app(
        GatewayConfig(
            api_keys={"tenant": "key"}, cache_salt_secret=b"s" * 32, origin_node_id="ross"
        ),
        scheduler=GrpcSchedulerClient(f"127.0.0.1:{port}"),
        transport=httpx.MockTransport(worker),
    )
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway",
            ) as client,
        ):
            body = {
                "model": MODEL.model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            }
            headers = {"x-api-key": "key"}
            first = asyncio.create_task(
                client.post("/v1/chat/completions", headers=headers, json=body)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                assert (
                    await client.post("/v1/chat/completions", headers=headers, json=body)
                ).status_code == 503
            finally:
                complete.set()
                first_response = await first
            assert first_response.status_code == 200
            # A completed HTTP response is not a Worker completion proof.
            held = next(iter((await ledger.snapshot()).reservations.values()))
            assert held.state == "completion_pending"
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=body)
            ).status_code == 503
            await confirm_execution(ledger, held.decision, held.tenant_id)
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=body)
            ).status_code == 200
            snapshot = await ledger.snapshot()
            assert len(snapshot.reservations) == 2
            assert len(snapshot.pending) == 5
            assert sorted(item.state for item in snapshot.reservations.values()) == [
                "completion_pending",
                "released",
            ]
    finally:
        await server.stop(None)
