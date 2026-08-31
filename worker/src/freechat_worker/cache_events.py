from __future__ import annotations

import asyncio
import queue
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from freechat_contracts import CacheEvent

EventSink = Callable[[tuple[CacheEvent, ...]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class BufferStats:
    accepted: int
    dropped: int
    delivered: int


class CacheEventBuffer:
    """Bounded, non-blocking bridge from vLLM cache hooks to control events.

    Cache callbacks call `try_emit`; network and serialization work happens in
    `run`. Overflow is observable and never stalls the engine scheduler thread.
    """

    def __init__(self, *, capacity: int = 65_536, batch_size: int = 256) -> None:
        if capacity < 1 or batch_size < 1 or batch_size > capacity:
            raise ValueError("invalid event buffer capacity or batch size")
        self._queue: queue.Queue[CacheEvent] = queue.Queue(maxsize=capacity)
        self._batch_size = batch_size
        self._accepted = 0
        self._dropped = 0
        self._delivered = 0

    def try_emit(self, event: CacheEvent) -> bool:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            return False
        self._accepted += 1
        return True

    @property
    def stats(self) -> BufferStats:
        return BufferStats(self._accepted, self._dropped, self._delivered)

    async def drain_once(self, sink: EventSink) -> int:
        try:
            first = self._queue.get_nowait()
        except queue.Empty:
            return 0
        batch = [first]
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        immutable_batch = tuple(batch)
        try:
            await sink(immutable_batch)
        except Exception:
            for event in immutable_batch:
                self._queue.task_done()
                try:
                    self._queue.put_nowait(event)
                except queue.Full:
                    self._dropped += 1
            raise
        for _ in immutable_batch:
            self._queue.task_done()
        self._delivered += len(immutable_batch)
        return len(immutable_batch)

    async def run(self, sink: EventSink, stop: asyncio.Event) -> None:
        while not stop.is_set() or not self._queue.empty():
            delivered = await self.drain_once(sink)
            if delivered == 0:
                await asyncio.sleep(0.01)
