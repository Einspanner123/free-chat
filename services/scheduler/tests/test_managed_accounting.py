from __future__ import annotations

import logging

import pytest
from freechat_contracts.execution import ExecutionAction, ExecutionCommand
from freechat_scheduler import Scheduler
from freechat_scheduler.grpc_server import SchedulerGrpcService
from freechat_scheduler.registry import InMemoryWorkerRegistry
from freechat_scheduler.request_execution import RegisteredExecutionDriver
from freechat_scheduler.request_ledger import RequestLedger
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


@pytest.mark.parametrize("failure", ["missing", "generation", "engine", "remote"])
async def test_registered_execution_never_retargets_old_or_remote_work(failure: str) -> None:
    registry = InMemoryWorkerRegistry()
    if failure != "missing":
        add_worker(registry, "a", node="ross")
        item = registry.snapshot()[1][0]
        registry.upsert(
            item.capabilities.model_copy(
                update={
                    "execution_endpoint": "remote:1234"
                    if failure == "remote"
                    else "127.0.0.1:1234",
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
    with pytest.raises(ValueError):
        await RegisteredExecutionDriver(registry, "t" * 32).observe(command)


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
