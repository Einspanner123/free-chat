from benchmarks.probe_lifecycle_retention import _hit_tokens


def test_hit_tokens_counts_all_cache_groups() -> None:
    assert _hit_tokens(
        [{"event_type": "hit", "block_ids": [[1, 2], [5]]}], 16
    ) == 48


def test_hit_tokens_is_zero_without_hit() -> None:
    assert _hit_tokens([{"event_type": "allocate", "block_ids": [[1]]}], 16) == 0
