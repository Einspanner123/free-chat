"""Unit tests for PrefixCache (pure logic, no torch / no model)."""

import pytest

from optimization.prefix_cache import PrefixCache, longest_prefix_len


def test_longest_prefix_len():
    assert longest_prefix_len([1, 2, 3, 4], (1, 2, 3)) == 3
    assert longest_prefix_len([1, 2], (1, 2, 3, 4)) == 2
    assert longest_prefix_len([9, 9], (1, 2, 3)) == 0
    assert longest_prefix_len([], (1, 2)) == 0
    assert longest_prefix_len([1, 2, 3], ()) == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        PrefixCache(capacity=0)


def test_lookup_no_match_returns_zero():
    pc = PrefixCache(capacity=4)
    pc.store([1, 2, 3], "cacheA")
    matched, val = pc.lookup([9, 9, 9])
    assert matched == 0
    assert val is None


def test_lookup_exact_prefix_hit():
    pc = PrefixCache(capacity=4)
    pc.store([1, 2, 3], "cacheA")
    matched, val = pc.lookup([1, 2, 3, 4, 5])
    assert matched == 3
    assert val == "cacheA"


def test_lookup_returns_longest_prefix():
    pc = PrefixCache(capacity=4)
    pc.store([1, 2], "short")
    pc.store([1, 2, 3, 4], "long")
    matched, val = pc.lookup([1, 2, 3, 4, 5, 6])
    assert matched == 4
    assert val == "long"


def test_store_evicts_lru():
    pc = PrefixCache(capacity=2)
    pc.store([1], "a")
    pc.store([2], "b")
    pc.store([3], "c")  # evicts (1,) (oldest)
    assert len(pc) == 2
    assert (1,) not in pc.keys()
    assert (2,) in pc.keys() and (3,) in pc.keys()


def test_lru_touch_on_lookup_keeps_recent():
    pc = PrefixCache(capacity=2)
    pc.store([1], "a")
    pc.store([2], "b")
    pc.lookup([1, 9, 9])  # touch [1] -> becomes most recent
    pc.store([3], "c")  # should evict (2,), not (1,)
    assert (1,) in pc.keys()
    assert (2,) not in pc.keys()
    assert (3,) in pc.keys()


def test_lookup_does_not_accept_partial_prefix_as_full():
    # cached key [1,2,3]; new prompt [1,2] is SHORTER than the key -> not a prefix of it
    pc = PrefixCache(capacity=4)
    pc.store([1, 2, 3], "long")
    matched, val = pc.lookup([1, 2])
    assert matched == 0
    assert val is None
