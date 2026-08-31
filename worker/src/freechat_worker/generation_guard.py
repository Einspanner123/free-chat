from __future__ import annotations

from dataclasses import dataclass


class StaleGeneration(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenerationGuard:
    worker_generation: int
    cache_generation: int

    def validate(self, worker_generation: int, cache_generation: int) -> None:
        if worker_generation != self.worker_generation:
            raise StaleGeneration(
                f"worker generation {worker_generation} != {self.worker_generation}"
            )
        if cache_generation != self.cache_generation:
            raise StaleGeneration(
                f"cache generation {cache_generation} != {self.cache_generation}"
            )
