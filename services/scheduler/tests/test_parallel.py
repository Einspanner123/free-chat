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
