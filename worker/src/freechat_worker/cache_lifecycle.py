"""Resolve trusted admission identities before entering native cache control."""

from __future__ import annotations

from datetime import UTC, datetime

from freechat_contracts.cache_lifecycle import CacheLifecycleCommand, CacheLifecycleReceipt

from freechat_worker.execution import DurableExecutionDriver
from freechat_worker.native_serving import NativeExecutionBackend


class WorkerCacheLifecycleDriver:
    """Bridge journal ownership to residency control, not execution cancellation."""

    def __init__(self, gate: DurableExecutionDriver, backend: NativeExecutionBackend) -> None:
        self.gate = gate
        self.backend = backend

    async def apply(self, command: CacheLifecycleCommand) -> CacheLifecycleReceipt:
        key = self.gate.submitted_key(command.owner)
        # This execution slice uses the Worker generation as the cache epoch.
        if command.cache_generation != command.owner.worker_generation:
            raise ValueError("cache_generation_not_owned")
        prefixes = await self.backend.cache_lifecycle(key, command)
        return CacheLifecycleReceipt(
            command=command,
            prefixes=prefixes,
            observed_at=datetime.now(UTC),
        )
