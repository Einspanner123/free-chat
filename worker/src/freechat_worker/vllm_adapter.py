from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freechat_contracts import CacheEvent, CacheEventKind

from freechat_worker.cache_events import CacheEventBuffer
from freechat_worker.generation_guard import GenerationGuard, StaleGeneration

_KIND_BY_NAME = {
    "allocate": CacheEventKind.ALLOCATE,
    "free": CacheEventKind.FREE,
    "hit": CacheEventKind.HIT,
    "evict": CacheEventKind.EVICT,
}


@dataclass(frozen=True, slots=True)
class VllmHookStats:
    accepted: int
    stale: int
    uncorrelated: int
    malformed: int


class VllmCacheHookBridge:
    """Map synchronous fork events into the bounded worker event path."""

    def __init__(
        self,
        buffer: CacheEventBuffer,
        guard: GenerationGuard,
        *,
        worker_id: str,
        block_tokens: tuple[int, ...],
        block_bytes: tuple[int, ...],
    ) -> None:
        if len(block_tokens) != len(block_bytes) or not block_tokens:
            raise ValueError("block token and byte layouts must be non-empty and aligned")
        if any(item < 1 for item in (*block_tokens, *block_bytes)):
            raise ValueError("block token and byte sizes must be positive")
        self._buffer = buffer
        self._guard = guard
        self._worker_id = worker_id
        self._block_tokens = block_tokens
        self._block_bytes = block_bytes
        self._accepted = 0
        self._stale = 0
        self._uncorrelated = 0
        self._malformed = 0

    def on_cache_event(self, event: Any) -> None:
        metadata = getattr(event, "metadata", None)
        if metadata is None:
            self._uncorrelated += 1
            return
        try:
            self._guard.validate(metadata.worker_generation, metadata.cache_generation)
        except StaleGeneration:
            self._stale += 1
            return
        kind = _KIND_BY_NAME.get(str(getattr(event, "event_type", "")))
        groups = getattr(event, "block_ids", None)
        if kind is None or not isinstance(groups, tuple) or len(groups) > len(
            self._block_tokens
        ):
            self._malformed += 1
            return
        for group_index, block_ids in enumerate(groups):
            if not isinstance(block_ids, tuple):
                self._malformed += 1
                return
            for block_id in block_ids:
                accepted = self._buffer.try_emit(
                    CacheEvent(
                        kind=kind,
                        tenant_id=metadata.tenant_id,
                        worker_id=self._worker_id,
                        worker_generation=metadata.worker_generation,
                        cache_generation=metadata.cache_generation,
                        block_id=f"{group_index}:{block_id}",
                        cache_key=metadata.cache_key,
                        token_count=self._block_tokens[group_index],
                        bytes=self._block_bytes[group_index],
                        metadata={
                            "request_id": str(getattr(event, "request_id", "")),
                            "reason": str(getattr(event, "reason", "")),
                            "task_id": metadata.task_id,
                            "agent_id": metadata.agent_id,
                            "branch_id": metadata.branch_id,
                            "call_id": metadata.call_id,
                            "lifecycle": str(metadata.lifecycle),
                        },
                    )
                )
                if accepted:
                    self._accepted += 1

    @property
    def stats(self) -> VllmHookStats:
        return VllmHookStats(
            accepted=self._accepted,
            stale=self._stale,
            uncorrelated=self._uncorrelated,
            malformed=self._malformed,
        )


_configured_bridge: VllmCacheHookBridge | None = None


def configure_vllm_cache_hook(bridge: VllmCacheHookBridge) -> None:
    global _configured_bridge
    _configured_bridge = bridge


def create_vllm_cache_hook() -> VllmCacheHookBridge:
    if _configured_bridge is None:
        raise RuntimeError("vLLM cache hook bridge is not configured")
    return _configured_bridge
