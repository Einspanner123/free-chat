from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ParallelMode(StrEnum):
    INDEPENDENT = "independent"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    EXPERT = "expert"


@dataclass(frozen=True, slots=True)
class DeviceLink:
    left_gpu: str
    right_gpu: str
    same_node: bool
    p2p_verified: bool
    nccl_verified: bool
    bandwidth_gib_per_second: float | None = None
    latency_microseconds: float | None = None


@dataclass(frozen=True, slots=True)
class ParallelCandidate:
    mode: ParallelMode
    gpu_ids: tuple[str, ...]
    homogeneous_compute_capability: bool
    measured_communication_fraction: float | None = None
    measured_end_to_end_speedup: float | None = None


@dataclass(frozen=True, slots=True)
class ParallelDecision:
    accepted: bool
    mode: ParallelMode
    gpu_ids: tuple[str, ...]
    reasons: tuple[str, ...]


class ParallelPlanner:
    """Rejects unsafe collectives until topology and end-to-end evidence exist."""

    def __init__(self, *, maximum_communication_fraction: float = 0.25) -> None:
        self._maximum_communication_fraction = maximum_communication_fraction

    def evaluate(
        self,
        candidate: ParallelCandidate,
        links: tuple[DeviceLink, ...],
    ) -> ParallelDecision:
        if candidate.mode is ParallelMode.INDEPENDENT:
            return ParallelDecision(True, candidate.mode, candidate.gpu_ids, ())
        reasons: list[str] = []
        if len(candidate.gpu_ids) < 2:
            reasons.append("collective_requires_multiple_gpus")
        relevant_links = tuple(
            link
            for link in links
            if link.left_gpu in candidate.gpu_ids and link.right_gpu in candidate.gpu_ids
        )
        if len(relevant_links) < len(candidate.gpu_ids) - 1:
            reasons.append("topology_link_unmeasured")
        if any(not link.same_node for link in relevant_links):
            reasons.append("cross_node_collective_forbidden")
        if any(not link.p2p_verified for link in relevant_links):
            reasons.append("p2p_unverified")
        if any(not link.nccl_verified for link in relevant_links):
            reasons.append("nccl_unverified")
        if not candidate.homogeneous_compute_capability:
            reasons.append("heterogeneous_compute_capability")
        if candidate.measured_communication_fraction is None:
            reasons.append("communication_fraction_unmeasured")
        elif candidate.measured_communication_fraction > self._maximum_communication_fraction:
            reasons.append("communication_fraction_too_high")
        if candidate.measured_end_to_end_speedup is None:
            reasons.append("end_to_end_gain_unmeasured")
        elif candidate.measured_end_to_end_speedup <= 1.0:
            reasons.append("no_end_to_end_gain")
        return ParallelDecision(
            accepted=not reasons,
            mode=candidate.mode,
            gpu_ids=candidate.gpu_ids,
            reasons=tuple(reasons),
        )
