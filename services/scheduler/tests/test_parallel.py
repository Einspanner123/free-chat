import pytest
from freechat_scheduler import (
    DeviceLink,
    ParallelCandidate,
    ParallelMode,
    ParallelPlanner,
)

PLANNER = ParallelPlanner(maximum_communication_fraction=0.2)


def test_cross_node_tensor_parallel_is_explicitly_rejected() -> None:
    candidate = ParallelCandidate(
        mode=ParallelMode.TENSOR,
        gpu_ids=("ross/a6000", "workstation/a5000"),
        homogeneous_compute_capability=True,
        measured_communication_fraction=0.1,
        measured_end_to_end_speedup=1.2,
    )
    links = (
        DeviceLink(
            "ross/a6000",
            "workstation/a5000",
            same_node=False,
            p2p_verified=False,
            nccl_verified=False,
        ),
    )
    decision = PLANNER.evaluate(candidate, links)
    assert not decision.accepted
    assert "cross_node_collective_forbidden" in decision.reasons
    assert "p2p_unverified" in decision.reasons


def test_workstation_pair_requires_measured_gain() -> None:
    candidate = ParallelCandidate(
        mode=ParallelMode.PIPELINE,
        gpu_ids=("workstation/a5000", "workstation/a4000"),
        homogeneous_compute_capability=True,
        measured_communication_fraction=0.12,
        measured_end_to_end_speedup=None,
    )
    links = (
        DeviceLink(
            "workstation/a5000",
            "workstation/a4000",
            same_node=True,
            p2p_verified=True,
            nccl_verified=True,
        ),
    )
    decision = PLANNER.evaluate(candidate, links)
    assert not decision.accepted
    assert decision.reasons == ("end_to_end_gain_unmeasured",)


def test_collective_is_enabled_only_after_all_gates() -> None:
    candidate = ParallelCandidate(
        mode=ParallelMode.TENSOR,
        gpu_ids=("node/gpu0", "node/gpu1"),
        homogeneous_compute_capability=True,
        measured_communication_fraction=0.16,
        measured_end_to_end_speedup=1.18,
    )
    links = (
        DeviceLink(
            "node/gpu0",
            "node/gpu1",
            same_node=True,
            p2p_verified=True,
            nccl_verified=True,
            bandwidth_gib_per_second=22.0,
            latency_microseconds=8.0,
        ),
    )
    assert PLANNER.evaluate(candidate, links).accepted


@pytest.mark.parametrize("gpus", [(), ("",), ("node/a", "node/a")])
def test_invalid_gpu_sets(gpus: tuple[str, ...]) -> None:
    assert not PLANNER.evaluate(
        ParallelCandidate(ParallelMode.INDEPENDENT, gpus, True), ()
    ).accepted


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_measurements_cannot_pass_gates(value: float) -> None:
    candidate = ParallelCandidate(ParallelMode.TENSOR, ("a", "b"), True, value, 2)
    assert not PLANNER.evaluate(candidate, (DeviceLink("a", "b", True, True, True),)).accepted


def test_duplicate_edges_do_not_hide_disconnected_gpu() -> None:
    candidate = ParallelCandidate(ParallelMode.TENSOR, ("a", "b", "c", "d"), True, 0.1, 2)
    edge = DeviceLink("a", "b", True, True, True)
    decision = PLANNER.evaluate(candidate, (edge, edge, DeviceLink("b", "c", True, True, True)))
    assert "topology_link_unmeasured" in decision.reasons
    assert "duplicate_or_self_link" in decision.reasons


@pytest.mark.parametrize(
    "communication,speedup,reason",
    [
        (None, 2, "communication_fraction_unmeasured"),
        (0.5, 2, "communication_fraction_too_high"),
        (0.1, 1, "no_end_to_end_gain"),
    ],
)
def test_parallel_measurement_boundaries(
    communication: float | None, speedup: float, reason: str
) -> None:
    candidate = ParallelCandidate(ParallelMode.TENSOR, ("a", "b"), True, communication, speedup)
    assert reason in PLANNER.evaluate(candidate, (DeviceLink("a", "b", True, True, True),)).reasons


def test_independent_singleton_and_collective_singleton() -> None:
    assert PLANNER.evaluate(ParallelCandidate(ParallelMode.INDEPENDENT, ("a",), True), ()).accepted
    assert (
        "collective_requires_multiple_gpus"
        in PLANNER.evaluate(
            ParallelCandidate(ParallelMode.TENSOR, ("a",), True, 0.1, 2), ()
        ).reasons
    )


@pytest.mark.parametrize("threshold", [-1, 2, float("nan"), float("inf")])
def test_invalid_parallel_policy(threshold: float) -> None:
    with pytest.raises(ValueError):
        ParallelPlanner(maximum_communication_fraction=threshold)


@pytest.mark.parametrize(
    "bandwidth,latency", [(0, 1), (-1, 1), (float("nan"), 1), (1, -1), (1, float("inf"))]
)
def test_invalid_link_samples(bandwidth: float, latency: float) -> None:
    decision = PLANNER.evaluate(
        ParallelCandidate(ParallelMode.TENSOR, ("a", "b"), True, 0.1, 2),
        (DeviceLink("a", "b", True, True, True, bandwidth, latency),),
    )
    assert "invalid_link_measurement" in decision.reasons


def test_connected_chain_can_be_listed_in_any_order() -> None:
    decision = PLANNER.evaluate(
        ParallelCandidate(ParallelMode.TENSOR, ("a", "b", "c", "d"), True, 0.1, 2),
        (
            DeviceLink("c", "d", True, True, True),
            DeviceLink("b", "c", True, True, True),
            DeviceLink("a", "b", True, True, True),
        ),
    )
    assert decision.accepted
