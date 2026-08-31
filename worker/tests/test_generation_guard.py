import pytest
from freechat_worker import GenerationGuard, StaleGeneration


def test_stale_cache_action_is_rejected() -> None:
    guard = GenerationGuard(worker_generation=3, cache_generation=9)
    with pytest.raises(StaleGeneration, match="cache generation"):
        guard.validate(worker_generation=3, cache_generation=8)
