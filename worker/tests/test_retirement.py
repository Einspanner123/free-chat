from __future__ import annotations

import asyncio
import fcntl
import json
import socket
import sqlite3
from contextlib import suppress
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import grpc
import httpx
import pytest
from freechat.control.v1 import control_pb2, control_pb2_grpc
from freechat_contracts.execution import ExecutionStatus
from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.retirement import retire_journal, serve_retired
from freechat_worker.runtime import ContainerBinding, DockerInspector, DockerStopProof
from test_execution import Backend, command

CONTAINER = "a" * 64
RUNTIME = "c" * 32
OBSERVER_RUNTIME = "d" * 32


@pytest.fixture(autouse=True)
def observer_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREECHAT_RUNTIME_ID", OBSERVER_RUNTIME)


class Inspector(DockerInspector):
    def __init__(self, root: Path) -> None:
        now = datetime.now(UTC)
        self.target: dict[str, Any] = {
            "Id": CONTAINER,
            "Created": (now - timedelta(seconds=20)).isoformat(),
            "Config": {
                "Hostname": "network-owner",
                "Labels": {"io.freechat.runtime-id": RUNTIME},
                "Env": [f"FREECHAT_RUNTIME_ID={RUNTIME}"],
                "Entrypoint": ["python", "-m", "freechat_worker.serve"],
                "Cmd": ["--worker-id", "worker", "--state-dir", str(root)],
            },
            "HostConfig": {
                "RestartPolicy": {"Name": "no"},
                "Privileged": False,
                "PidMode": "",
                "CapAdd": None,
            },
            "State": {
                "Status": "exited",
                "Running": False,
                "Restarting": False,
                "Paused": False,
                "Dead": False,
                "Pid": 0,
                "StartedAt": (now - timedelta(seconds=10)).isoformat(),
                "FinishedAt": (now - timedelta(seconds=1)).isoformat(),
            },
            "Mounts": [
                {"Type": "volume", "Source": "/volume", "Destination": str(root), "RW": True}
            ],
        }
        self.observer = deepcopy(self.target)
        self.observer["Id"] = "b" * 64
        self.calls: list[str] = []

    def container_for_runtime(self, runtime_id: str) -> str:
        assert runtime_id == OBSERVER_RUNTIME
        return str(self.observer["Id"])

    def verify_unique_runtime(self, runtime_id: str, container_id: str) -> None:
        assert runtime_id == RUNTIME and container_id == CONTAINER

    def inspect(self, identity: str) -> dict[str, Any]:
        self.calls.append(identity)
        return self.target if identity == CONTAINER else self.observer


def open_gate(
    root: Path, *, create: bool = False, retired: bool = False, binding: bool = True
) -> DurableExecutionDriver:
    directory = root / "worker"
    directory.mkdir(exist_ok=True)
    return DurableExecutionDriver(
        directory / "1.sqlite",
        None if retired else Backend(),
        worker_id="worker",
        generation=1,
        engine_instance_id="engine",
        create=create,
        retired=retired,
        container_binding=ContainerBinding(
            runtime_id=RUNTIME, opened_at=datetime.now(UTC) - timedelta(seconds=5)
        )
        if binding
        else None,
    )


def retire(root: Path, inspector: Inspector) -> DockerStopProof:
    return retire_journal(
        root,
        worker_id="worker",
        generation=1,
        engine_instance_id="engine",
        container_id=CONTAINER,
        inspector=inspector,
    )


async def test_retirement_fences_all_late_admission_and_preserves_completed_results(
    tmp_path: Path,
) -> None:
    gate = open_gate(tmp_path, create=True)
    await gate.admit(command(), {})
    running = await gate.observe(command())
    completed = command(decision_id="done", request_id="done")
    await gate.admit(completed, {})
    assert isinstance(gate.backend, Backend)
    gate.backend.status = ExecutionStatus.COMPLETED
    gate.backend.quiet = gate.backend.fenced = True
    await gate.observe(completed)
    gate.close()
    inspector = Inspector(tmp_path)
    first = retire(tmp_path, inspector)
    assert retire(tmp_path, inspector) == first  # Durable idempotence, not a fresh timestamp.
    with pytest.raises(ValueError, match="incarnation_retired"):
        open_gate(tmp_path)
    retired = open_gate(tmp_path, retired=True)
    try:
        result = await retired.observe(command())
        assert result.status == ExecutionStatus.ABORTED and result.releasable
        assert result.observation_sequence > running.observation_sequence
        assert (await retired.observe(completed)).status == ExecutionStatus.COMPLETED
        unknown = command(decision_id="never-arrived", request_id="never-arrived")
        proof = await retired.observe(unknown)
        assert proof.status == ExecutionStatus.NOT_ACCEPTED and proof.releasable
        with pytest.raises(ValueError, match="incarnation_retired"):
            await retired.admit(unknown, {})
        with pytest.raises(ValueError, match="incarnation_mismatch"):
            await retired.observe(command(engine_instance_id="replacement"))
        with pytest.raises(ValueError, match="binding_mismatch"):
            await retired.observe(command(request_id="wrong-request"))
    finally:
        retired.close()
    # Sequences and fences survive restarting the observer itself.
    retired = open_gate(tmp_path, retired=True)
    try:
        assert (await retired.observe(command())).observation_sequence > result.observation_sequence
    finally:
        retired.close()


@pytest.mark.parametrize(
    "case",
    [
        "running",
        "paused",
        "restarting",
        "dead",
        "pid",
        "auto_restart",
        "privileged",
        "host_pid",
        "capability",
        "label",
        "environment",
        "recreated",
        "container",
        "worker",
        "launcher",
        "state_mount",
        "state_path",
        "future_journal",
        "missing_binding",
    ],
)
def test_unproven_retirement_preserves_live_journal(tmp_path: Path, case: str) -> None:
    gate = open_gate(tmp_path, create=True, binding=case != "missing_binding")
    gate.close()
    inspector = Inspector(tmp_path)
    target = inspector.target
    if case in {"running", "paused", "restarting", "dead"}:
        target["State"][case.capitalize()] = True
    elif case == "pid":
        target["State"]["Pid"] = 1234
    elif case == "auto_restart":
        target["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif case == "privileged":
        target["HostConfig"]["Privileged"] = True
    elif case == "host_pid":
        target["HostConfig"]["PidMode"] = "host"
    elif case == "capability":
        target["HostConfig"]["CapAdd"] = ["SYS_ADMIN"]
    elif case == "label":
        target["Config"]["Labels"] = {}
    elif case == "environment":
        target["Config"]["Env"] = []
    elif case == "recreated":
        target["Created"] = datetime.now(UTC).isoformat()
    elif case == "container":
        target["Id"] = "c" * 64
    elif case == "worker":
        target["Config"]["Cmd"][1] = "wrong"
    elif case == "launcher":
        target["Config"]["Entrypoint"] = ["sleep"]
    elif case == "state_mount":
        inspector.observer["Mounts"][0]["Source"] = "/another-volume"
    elif case == "state_path":
        target["Config"]["Cmd"][-1] = "/elsewhere"
    elif case == "future_journal":
        target["State"]["FinishedAt"] = (datetime.now(UTC) - timedelta(seconds=6)).isoformat()
    with pytest.raises(ValueError):
        retire(tmp_path, inspector)
    gate = open_gate(tmp_path)
    assert gate.retirement is None
    gate.close()


@pytest.mark.parametrize("lock", ["runtime", "journal"])
def test_live_owner_prevents_even_docker_inspection(tmp_path: Path, lock: str) -> None:
    gate = open_gate(tmp_path, create=True)
    inspector = Inspector(tmp_path)
    if lock == "journal":
        try:
            with pytest.raises(ValueError, match="already_owned"):
                retire(tmp_path, inspector)
        finally:
            gate.close()
    else:
        gate.close()
        with (tmp_path / "worker" / "runtime.owner").open("a+b") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(ValueError, match="still_owned"):
                retire(tmp_path, inspector)
    assert not inspector.calls


def test_retired_mode_cannot_fabricate_termination(tmp_path: Path) -> None:
    gate = open_gate(tmp_path, create=True)
    gate.close()
    with pytest.raises(ValueError, match="retirement_unproven"):
        open_gate(tmp_path, retired=True)


def test_inspector_rejects_unscoped_names_before_accessing_socket() -> None:
    inspector = DockerInspector("/nonexistent.sock")
    try:
        for value in ("../containers", "worker", "a" * 13):
            with pytest.raises(ValueError, match="identity_invalid"):
                inspector.inspect(value)
    finally:
        inspector.close()


def test_identity_mismatch_and_absent_journal_are_not_reconstructed(tmp_path: Path) -> None:
    gate = open_gate(tmp_path, create=True)
    gate.close()
    inspector = Inspector(tmp_path)
    with pytest.raises(ValueError, match="incarnation_mismatch"):
        retire_journal(
            tmp_path,
            worker_id="worker",
            generation=1,
            engine_instance_id="wrong",
            container_id=CONTAINER,
            inspector=inspector,
        )
    assert not inspector.calls
    with pytest.raises(ValueError, match="existing_journal"):
        retire_journal(
            tmp_path,
            worker_id="worker",
            generation=2,
            engine_instance_id="engine",
            container_id=CONTAINER,
            inspector=inspector,
        )


def test_container_binding_is_not_backfilled_when_reopening_legacy_state(tmp_path: Path) -> None:
    gate = open_gate(tmp_path, create=True, binding=False)
    gate.close()
    gate = open_gate(tmp_path, binding=True)
    gate.close()
    with sqlite3.connect(tmp_path / "worker" / "1.sqlite") as db:
        assert db.execute("SELECT value FROM meta WHERE id=3").fetchone() is None


def test_binding_requires_explicit_runtime_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREECHAT_RUNTIME_ID", raising=False)
    assert ContainerBinding.current() is None
    monkeypatch.setenv("FREECHAT_RUNTIME_ID", "not-a-runtime-id")
    with pytest.raises(ValueError):
        ContainerBinding.current()
    monkeypatch.setenv("FREECHAT_RUNTIME_ID", RUNTIME)
    binding = ContainerBinding.current()
    assert binding is not None and binding.runtime_id == RUNTIME


@pytest.mark.parametrize("identities", [[], [CONTAINER], [CONTAINER, "b" * 64]])
def test_runtime_label_must_identify_exactly_one_container(identities: list[str]) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET" and request.url.path == "/containers/json"
        assert request.url.params["all"] == "1"
        assert json.loads(request.url.params["filters"]) == {
            "label": [f"io.freechat.runtime-id={RUNTIME}"]
        }
        return httpx.Response(200, json=[{"Id": value} for value in identities])

    inspector = DockerInspector("/unused.sock")
    inspector.close()
    inspector.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://docker", trust_env=False
    )
    try:
        if identities == [CONTAINER]:
            inspector.verify_unique_runtime(RUNTIME, CONTAINER)
        else:
            with pytest.raises(ValueError, match="label_not_unique"):
                inspector.verify_unique_runtime(RUNTIME, CONTAINER)
        assert len(calls) == 1
    finally:
        inspector.close()


@pytest.mark.parametrize(
    "listen,token,reason",
    [
        ("0.0.0.0:50052", "t" * 32, "loopback_required"),
        ("127x0x0x1:50052", "t" * 32, "loopback_required"),
        ("127.0.0.1:50052", "short", "at least 32"),
    ],
)
async def test_retired_rpc_validates_boundary_before_opening_state(
    tmp_path: Path, listen: str, token: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        await serve_retired(
            tmp_path,
            worker_id="worker",
            generation=1,
            engine_instance_id="engine",
            listen=listen,
            token=token,
        )
    assert not list(tmp_path.iterdir())


async def test_historical_rpc_coexists_with_replacement_owner_and_journal(tmp_path: Path) -> None:
    gate = open_gate(tmp_path, create=True)
    await gate.admit(command(), {})
    gate.close()
    retire(tmp_path, Inspector(tmp_path))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    with (tmp_path / "worker" / "runtime.owner").open("a+b") as replacement_owner:
        fcntl.flock(replacement_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        replacement = DurableExecutionDriver(
            tmp_path / "worker" / "2.sqlite",
            Backend(),
            worker_id="worker",
            generation=2,
            engine_instance_id="new-engine",
            create=True,
        )
        task = asyncio.create_task(
            serve_retired(
                tmp_path,
                worker_id="worker",
                generation=1,
                engine_instance_id="engine",
                listen=f"127.0.0.1:{port}",
                token="t" * 32,
                exclusive_runtime=False,
            )
        )
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                async with asyncio.timeout(5):
                    await channel.channel_ready()
                stub = control_pb2_grpc.RequestExecutionServiceStub(channel)  # type: ignore[no-untyped-call]
                result = await stub.Observe(
                    control_pb2.RequestExecutionCommand(command_json=command().model_dump_json()),
                    metadata=(("authorization", "Bearer " + "t" * 32),),
                    timeout=2,
                )
                assert '"status":"aborted"' in result.receipt_json
                new_command = command(worker_generation=2, engine_instance_id="new-engine")
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await stub.Observe(
                        control_pb2.RequestExecutionCommand(
                            command_json=new_command.model_dump_json()
                        ),
                        metadata=(("authorization", "Bearer " + "t" * 32),),
                        timeout=2,
                    )
                assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                await replacement.admit(new_command, {})
                assert (await replacement.observe(new_command)).status == ExecutionStatus.RUNNING
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            replacement.close()
