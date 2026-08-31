from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime

import psutil
from pydantic import BaseModel, ConfigDict, Field


class GPUDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    uuid: str
    name: str
    driver_version: str
    memory_total_bytes: int
    compute_capability: str


class TopologySnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    hostname: str
    operating_system: str
    kernel: str
    cpu_count: int
    memory_total_bytes: int
    gpus: tuple[GPUDevice, ...]
    nvidia_smi_topology: str | None
    nvidia_p2p_read: str | None
    nvidia_p2p_write: str | None
    numa_hardware: str | None
    pcie_devices: str | None
    network_links_json: str | None
    nvcc_version: str | None
    nsight_compute_version: str | None
    nsight_systems_version: str | None
    collection_scope: str


def collect_snapshot() -> TopologySnapshot:
    return TopologySnapshot(
        hostname=platform.node(),
        operating_system=platform.platform(),
        kernel=platform.release(),
        cpu_count=psutil.cpu_count(logical=True) or 1,
        memory_total_bytes=psutil.virtual_memory().total,
        gpus=tuple(_collect_gpus()),
        nvidia_smi_topology=_run_optional(["nvidia-smi", "topo", "-m"]),
        nvidia_p2p_read=_run_optional(["nvidia-smi", "topo", "-p2p", "r"]),
        nvidia_p2p_write=_run_optional(["nvidia-smi", "topo", "-p2p", "w"]),
        numa_hardware=_run_optional(["numactl", "--hardware"]),
        pcie_devices=_run_optional(["lspci", "-Dnn"]),
        network_links_json=_run_optional(["ip", "-j", "link", "show"]),
        nvcc_version=_run_optional(["nvcc", "--version"]),
        nsight_compute_version=_run_optional(["ncu", "--version"]),
        nsight_systems_version=_run_optional(["nsys", "--version"]),
        collection_scope=os.environ.get("FREECHAT_TOPOLOGY_SCOPE", "host-observation"),
    )


def _collect_gpus() -> list[GPUDevice]:
    output = _run_optional(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    devices: list[GPUDevice] = []
    for line in output.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 6:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        index, uuid, name, driver, memory_mib, capability = columns
        devices.append(
            GPUDevice(
                index=int(index),
                uuid=uuid,
                name=name,
                driver_version=driver,
                memory_total_bytes=int(memory_mib) * 1024**2,
                compute_capability=capability,
            )
        )
    return devices


def _run_optional(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
