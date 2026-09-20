"""Container-bound termination evidence for managed, single-container Workers."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContainerBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    opened_at: datetime

    @classmethod
    def current(cls) -> ContainerBinding | None:
        runtime_id = os.environ.get("FREECHAT_RUNTIME_ID")
        if runtime_id is None:
            return None
        return cls(runtime_id=runtime_id, opened_at=datetime.now(UTC))


class DockerStopProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Literal["docker-stopped"] = "docker-stopped"
    container_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    finished_at: datetime
    observed_at: datetime

    @model_validator(mode="after")
    def chronological(self) -> DockerStopProof:
        dates = (self.started_at, self.finished_at, self.observed_at)
        if any(value.tzinfo is None for value in dates):
            raise ValueError("runtime_evidence_requires_timezone")
        if not self.started_at <= self.finished_at <= self.observed_at:
            raise ValueError("runtime_evidence_time_order")
        return self


class DockerInspector:
    """Only GET a named container over a local Unix socket; never mutate Docker."""

    def __init__(self, socket_path: str = "/var/run/docker.sock") -> None:
        self.client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path),
            base_url="http://docker",
            timeout=5,
            trust_env=False,
        )

    def inspect(self, container_id: str) -> dict[str, Any]:
        if re.fullmatch(r"[0-9a-f]{12}|[0-9a-f]{64}", container_id) is None:
            raise ValueError("docker_container_identity_invalid")
        response = self.client.get(f"/containers/{container_id}/json")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def container_for_runtime(self, runtime_id: str) -> str:
        if re.fullmatch(r"[0-9a-f]{32}", runtime_id) is None:
            raise ValueError("docker_runtime_identity_invalid")
        response = self.client.get(
            "/containers/json",
            params={
                "all": "1",
                "filters": json.dumps({"label": [f"io.freechat.runtime-id={runtime_id}"]}),
            },
        )
        response.raise_for_status()
        identities = [str(item["Id"]) for item in response.json()]
        if len(identities) != 1 or re.fullmatch(r"[0-9a-f]{64}", identities[0]) is None:
            raise ValueError("runtime_container_label_not_unique")
        return identities[0]

    def verify_unique_runtime(self, runtime_id: str, container_id: str) -> None:
        if self.container_for_runtime(runtime_id) != container_id:
            raise ValueError("runtime_container_label_not_unique")

    def close(self) -> None:
        self.client.close()


def _state_mount(container: dict[str, Any], destination: str) -> tuple[str, str]:
    mounts = [item for item in container["Mounts"] if item["Destination"] == destination]
    if len(mounts) != 1 or not mounts[0]["RW"]:
        raise ValueError("runtime_state_mount_required")
    item = mounts[0]
    if item["Type"] not in {"volume", "bind"} or not item["Source"]:
        raise ValueError("runtime_state_mount_invalid")
    return str(item["Type"]), str(item["Source"])


def stopped_container_proof(
    container: dict[str, Any],
    observer: dict[str, Any],
    binding: ContainerBinding,
    *,
    container_id: str,
    worker_id: str,
    state_root: str,
    now: datetime,
) -> DockerStopProof:
    """Called while holding the same runtime.owner lock as the managed launcher."""
    config, host, state = container["Config"], container["HostConfig"], container["State"]
    if (
        container["Id"] != container_id
        or config.get("Labels", {}).get("io.freechat.runtime-id") != binding.runtime_id
        or [item for item in config["Env"] if item.startswith("FREECHAT_RUNTIME_ID=")]
        != [f"FREECHAT_RUNTIME_ID={binding.runtime_id}"]
        or observer["Id"] == container_id
    ):
        raise ValueError("runtime_container_binding_mismatch")
    if (
        state["Status"] != "exited"
        or state["Running"]
        or state["Restarting"]
        or state["Paused"]
        or state["Dead"]
        or state["Pid"] != 0
        or host["RestartPolicy"]["Name"] not in {"no", ""}
    ):
        raise ValueError("runtime_container_not_stopped")
    if host["Privileged"] or host["PidMode"] or host.get("CapAdd"):
        raise ValueError("runtime_container_isolation_unsupported")
    entrypoint = config["Entrypoint"]
    if not isinstance(entrypoint, list) or entrypoint[-2:] != ["-m", "freechat_worker.serve"]:
        raise ValueError("runtime_managed_launcher_required")
    args = config["Cmd"]
    if (
        not isinstance(args, list)
        or args.count("--worker-id") != 1
        or any(arg.startswith("--worker-id=") for arg in args)
    ):
        raise ValueError("runtime_worker_binding_required")
    position = args.index("--worker-id")
    if position + 1 == len(args) or args[position + 1] != worker_id:
        raise ValueError("runtime_worker_binding_mismatch")
    # The managed launcher must acquire the very same lock before starting its engine.
    root = "/var/lib/freechat"
    if "--state-dir" in args:
        if args.count("--state-dir") != 1:
            raise ValueError("runtime_state_argument_invalid")
        position = args.index("--state-dir")
        if position + 1 == len(args):
            raise ValueError("runtime_state_argument_invalid")
        root = args[position + 1]
    if root != state_root or any(arg.startswith("--state-dir=") for arg in args):
        raise ValueError("runtime_state_argument_mismatch")
    if _state_mount(container, root) != _state_mount(observer, state_root):
        raise ValueError("runtime_state_mount_mismatch")
    proof = DockerStopProof(
        container_id=container_id,
        started_at=state["StartedAt"],
        finished_at=state["FinishedAt"],
        observed_at=now,
    )
    created_at = datetime.fromisoformat(container["Created"])
    if (
        created_at.tzinfo is None
        or binding.opened_at.tzinfo is None
        or not created_at <= binding.opened_at <= proof.finished_at
    ):
        raise ValueError("runtime_journal_outside_container_lifetime")
    return proof
