from __future__ import annotations

from dataclasses import dataclass

from freechat_contracts import (
    AgentHints,
    CacheAction,
    CacheActionKind,
    HintSource,
    Lifecycle,
    ReuseClass,
)


@dataclass(frozen=True, slots=True)
class TierCosts:
    hbm_pressure_ms_per_gib: float
    cpu_load_bandwidth_bytes_per_second: float
    nvme_load_bandwidth_bytes_per_second: float
    prefill_tokens_per_second: float


@dataclass(frozen=True, slots=True)
class BlockRecord:
    block_id: str
    tenant_id: str
    worker_id: str
    worker_generation: int
    cache_generation: int
    token_count: int
    bytes: int
    reference_count: int
    current_tier: str
    hints: AgentHints


class LifecycleAwarePolicy:
    def __init__(self, costs: TierCosts) -> None:
        self._costs = costs

    def choose(self, block: BlockRecord) -> CacheAction:
        recompute_ms = block.token_count / self._costs.prefill_tokens_per_second * 1_000
        cpu_load_ms = block.bytes / self._costs.cpu_load_bandwidth_bytes_per_second * 1_000
        nvme_load_ms = block.bytes / self._costs.nvme_load_bandwidth_bytes_per_second * 1_000
        hbm_pressure_ms = block.bytes / (1024**3) * self._costs.hbm_pressure_ms_per_gib
        probability = block.hints.expected_reuse_probability
        if block.hints.source is HintSource.INFERRED:
            probability *= block.hints.confidence
        expected_recompute_ms = probability * recompute_ms

        if block.hints.reuse_class is ReuseClass.IMMUTABLE_SHARED and block.reference_count > 1:
            return self._action(
                block,
                CacheActionKind.RETAIN,
                "shared immutable prefix has live references",
                hbm_pressure_ms,
            )

        if (
            block.hints.lifecycle in {Lifecycle.TERMINAL, Lifecycle.CANCELLED}
            and block.reference_count == 0
        ):
            return self._action(
                block,
                CacheActionKind.EVICT,
                "private lifecycle ended and no references remain",
                recompute_ms,
            )

        if block.hints.reuse_class is ReuseClass.EPHEMERAL_REASONING:
            return self._action(
                block,
                CacheActionKind.EVICT,
                "reasoning suffix has near-zero expected reuse",
                recompute_ms,
            )

        if block.hints.lifecycle is Lifecycle.TOOL_WAIT:
            if not block.hints.allow_kv_offload:
                return self._action(
                    block,
                    CacheActionKind.RETAIN,
                    "tool wait forbids offload",
                    hbm_pressure_ms,
                )
            if cpu_load_ms < expected_recompute_ms and cpu_load_ms <= hbm_pressure_ms:
                return self._action(
                    block,
                    CacheActionKind.OFFLOAD_CPU,
                    "CPU restore is cheaper than expected recompute and HBM pressure",
                    cpu_load_ms,
                )
            if nvme_load_ms < expected_recompute_ms and nvme_load_ms <= hbm_pressure_ms:
                return self._action(
                    block,
                    CacheActionKind.OFFLOAD_NVME,
                    "NVMe restore is cheaper than expected recompute and HBM pressure",
                    nvme_load_ms,
                )

        if expected_recompute_ms > hbm_pressure_ms:
            return self._action(
                block,
                CacheActionKind.RETAIN,
                "expected recompute exceeds HBM opportunity cost",
                hbm_pressure_ms,
            )
        return self._action(
            block,
            CacheActionKind.EVICT,
            "expected reuse does not justify residency",
            recompute_ms,
        )

    @staticmethod
    def _action(
        block: BlockRecord,
        kind: CacheActionKind,
        reason: str,
        estimated_cost_ms: float,
    ) -> CacheAction:
        return CacheAction(
            kind=kind,
            block_id=block.block_id,
            worker_id=block.worker_id,
            worker_generation=block.worker_generation,
            cache_generation=block.cache_generation,
            reason=reason,
            estimated_cost_ms=max(0.0, estimated_cost_ms),
        )
