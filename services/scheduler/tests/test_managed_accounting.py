from __future__ import annotations

import logging

import pytest
from freechat_contracts.execution import ExecutionAction, ExecutionCommand, ExecutionStatus
from freechat_scheduler import Scheduler
from freechat_scheduler.grpc_server import SchedulerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_execution import (
    ExecutionObservationUnavailable,
    RegisteredExecutionDriver,
    RequestExecutionReconciler,
    RetiredExecutionConfig,
    RetiredExecutionRoute,
)
from freechat_scheduler.request_ledger import RequestLedger
from test_request_execution import FixtureExecutionDriver, start_runtime
from test_request_ledger import request as ledger_request
from test_request_ledger import reserve, setup
from test_scheduler import add_worker, profile


def test_scheduler_owned_load_is_not_counted_twice() -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross", active=2)
    add_worker(registry, "b", node="ross", active=3)
    for worker in registry.snapshot()[1]:
        registry.upsert(
            worker.capabilities,
            worker.telemetry.model_copy(
                update={
                    "admission_accounting": "scheduler_exclusive_gross",
                }
            ),
        )
    decision = Scheduler(registry).route(profile(), reserved={"a": (100, 2)})
    assert decision.worker_id == "a"  # max(2,2), not 2+2.


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("missing", "worker_not_registered"),
        ("generation", "worker_generation_changed"),
        ("engine", "engine_instance_changed"),
        ("endpoint", "execution_endpoint_missing"),
        ("remote", None),
    ],
)
async def test_registered_execution_never_retargets_old_or_remote_work(
    failure: str, reason: str | None
) -> None:
    registry = InMemoryWorkerRegistry()
    if failure != "missing":
        add_worker(registry, "a", node="ross")
        item = registry.snapshot()[1][0]
        registry.upsert(
            item.capabilities.model_copy(
                update={
                    "execution_endpoint": "remote:1234"
                    if failure == "remote"
                    else (None if failure == "endpoint" else "127.0.0.1:1234"),
                }
            ),
            item.telemetry,
        )
    command = ExecutionCommand(
        action=ExecutionAction.QUERY,
        tenant_id="tenant",
        request_id="r",
        decision_id="d",
        worker_id="a",
        worker_generation=2 if failure == "generation" else 1,
        engine_instance_id="other" if failure == "engine" else "fixture-engine",
    )
    with pytest.raises(ValueError) as raised:
        await RegisteredExecutionDriver(registry, "t" * 32).observe(command)
    if reason is not None:
        assert isinstance(raised.value, ExecutionObservationUnavailable)
        assert raised.value.reason == reason
    else:
        assert not isinstance(raised.value, ExecutionObservationUnavailable)


async def test_development_outbox_uses_service_logging_not_silent_ack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "a", node="ross")
    scheduler = Scheduler(registry)
    ledger = RequestLedger()
    request = profile()
    await ledger.reserve(request, "key", lambda reserved: scheduler.route(request))
    service = SchedulerGrpcService(scheduler, ledger, log_events=True)
    with caplog.at_level(logging.WARNING, logger="freechat_scheduler.grpc_server"):
        await service.flush_events()
    assert (await ledger.snapshot()).pending
    with caplog.at_level(logging.INFO, logger="freechat_scheduler.grpc_server"):
        await service.flush_events()
    assert not (await ledger.snapshot()).pending
    assert "lifecycle_event" in caplog.text and "route.decided" in caplog.text


@pytest.mark.parametrize("failure", ["none", "nonterminal", "mismatch", "endpoint_collision"])
async def test_historical_receipt_routes_do_not_replace_current_registration(failure: str) -> None:
    ledger, scheduler = RequestLedger(), setup()
    old = await reserve(ledger, scheduler, ledger_request())
    await ledger.release(
        old.decision_id, old.worker_id, old.worker_generation, "tenant-a", cancelled=True
    )
    # The authoritative current Worker has already changed generation and engine.
    registry = InMemoryWorkerRegistry()
    add_worker(registry, "worker-0", node="ross")
    previous = registry.snapshot()[1][0]
    backend = FixtureExecutionDriver()
    backend.status = ExecutionStatus.ABORTED
    backend.quiet = backend.closed = failure != "nonterminal"
    backend.mismatch = failure == "mismatch"
    server, endpoint = await start_runtime(backend)
    registry.upsert(
        previous.capabilities.model_copy(
            update={
                "generation": 2,
                "execution_endpoint": endpoint.address
                if failure == "endpoint_collision"
                else "127.0.0.1:1",
            }
        ),
        previous.telemetry.model_copy(update={"generation": 2, "engine_instance_id": "new-engine"}),
    )
    registry_before = registry.snapshot()
    ledger_before = await ledger.snapshot()
    driver = RegisteredExecutionDriver(
        registry,
        endpoint.token.get_secret_value(),
        RetiredExecutionConfig(
            routes=(
                RetiredExecutionRoute(
                    worker_id=old.worker_id,
                    generation=old.worker_generation,
                    engine_instance_id="fixture-engine",
                    address=endpoint.address,
                ),
            )
        ),
    )
    try:
        errors = await RequestExecutionReconciler(ledger, driver).tick()
        assert registry.snapshot() == registry_before
        if failure == "none":
            assert not errors
            assert (await ledger.snapshot()).reservations[old.decision_id].state.value == "released"
            assert len(backend.commands) == 1 and backend.commands[0].worker_generation == 1
            # Repeated reconciliation must not query or account the completed decision again.
            assert not await RequestExecutionReconciler(ledger, driver).tick()
            assert len(backend.commands) == 1
        else:
            assert errors and await ledger.snapshot() == ledger_before
            if failure == "endpoint_collision":
                assert not backend.commands
        assert backend.commands == [] or backend.commands[0].engine_instance_id == "fixture-engine"
    finally:
        await server.stop(0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_id", "other"),
        ("generation", 3),
        ("engine_instance_id", "other"),
    ],
)
async def test_retired_route_requires_all_incarnation_fields(field: str, value: object) -> None:
    registry = InMemoryWorkerRegistry()
    backend = FixtureExecutionDriver()
    server, endpoint = await start_runtime(backend)
    route = RetiredExecutionRoute(
        worker_id="worker-0",
        generation=1,
        engine_instance_id="fixture-engine",
        address=endpoint.address,
    ).model_copy(update={field: value})
    driver = RegisteredExecutionDriver(
        registry, endpoint.token.get_secret_value(), RetiredExecutionConfig(routes=(route,))
    )
    command = ExecutionCommand(
        action=ExecutionAction.ABORT,
        tenant_id="tenant",
        request_id="r",
        decision_id="d",
        worker_id="worker-0",
        worker_generation=1,
        engine_instance_id="fixture-engine",
    )
    try:
        with pytest.raises(ExecutionObservationUnavailable, match="worker_not_registered"):
            await driver.observe(command)
        assert not backend.commands
    finally:
        await server.stop(0)


@pytest.mark.parametrize("address", ["remote:50054", "0.0.0.0:50054", "127x0x0x1:50054"])
def test_retired_routes_reject_remote_or_malformed_endpoints(address: str) -> None:
    with pytest.raises(ValueError):
        RetiredExecutionRoute(worker_id="w", generation=1, engine_instance_id="e", address=address)


@pytest.mark.parametrize("change", [{}, {"address": "127.0.0.1:50055"}, {"generation": 2}])
def test_retired_routes_reject_duplicate_identity_or_endpoint(change: dict[str, object]) -> None:
    route = RetiredExecutionRoute(
        worker_id="w", generation=1, engine_instance_id="e", address="127.0.0.1:50054"
    )
    with pytest.raises(ValueError, match="duplicate_retired_execution"):
        RetiredExecutionConfig(routes=(route, route.model_copy(update=change)))


async def test_historical_routes_cannot_be_silently_ignored_in_contract_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from freechat_scheduler.grpc_server import serve

    config = RetiredExecutionConfig(
        routes=(
            RetiredExecutionRoute(
                worker_id="w", generation=1, engine_instance_id="e", address="127.0.0.1:50054"
            ),
        )
    )
    monkeypatch.setenv("FREECHAT_RETIRED_EXECUTION_ROUTES", config.model_dump_json())
    with pytest.raises(ValueError, match="require_managed"):
        await serve("127.0.0.1:0", contract_only=True)
