import os, sys, pytest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

class TestKVCache:
    def test_cache_init(self):
        from optimization.kv_cache import KVCache
        cache = KVCache(max_size=4096)
        assert cache.max_size == 4096
        assert cache.size == 0

    def test_cache_set_get(self):
        from optimization.kv_cache import KVCache
        cache = KVCache()
        cache.set("prefix1", {"key": "value1"})
        assert cache.get("prefix1") == {"key": "value1"}
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        from optimization.kv_cache import KVCache
        cache = KVCache(max_size=3)
        for i in range(5):
            cache.set(f"prefix{i}", {"data": i})
        assert cache.size <= 3

    def test_cache_clear(self):
        from optimization.kv_cache import KVCache
        cache = KVCache()
        cache.set("p1", {})
        cache.set("p2", {})
        cache.clear()
        assert cache.size == 0

    def test_cache_lru_behavior(self):
        from optimization.kv_cache import KVCache
        cache = KVCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")  # Access a → makes it recently used
        cache.set("d", 4)  # Should evict b (least recently used)
        assert cache.get("a") == 1  # a is still there
        assert cache.get("b") is None  # b was evicted

    def test_cache_contains(self):
        from optimization.kv_cache import KVCache
        cache = KVCache()
        cache.set("key", "val")
        assert cache.contains("key") is True
        assert cache.contains("missing") is False

    def test_cache_keys(self):
        from optimization.kv_cache import KVCache
        cache = KVCache()
        cache.set("a", 1)
        cache.set("b", 2)
        keys = cache.keys()
        assert "a" in keys
        assert "b" in keys

    def test_cache_delete(self):
        from optimization.kv_cache import KVCache
        cache = KVCache()
        cache.set("key", "val")
        cache.delete("key")
        assert cache.contains("key") is False

    def test_cache_stats(self):
        from optimization.kv_cache import KVCache
        cache = KVCache(max_size=100)
        cache.set("a", 1)
        cache.get("a")
        cache.get("b")  # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["hit_rate"] > 0

class TestPrefixCache:
    def test_prefix_cache(self):
        from optimization.kv_cache import PrefixCache
        pc = PrefixCache()
        pc.store("system prompt: be helpful", {"kv": "data"})
        assert pc.lookup("system prompt: be helpful") is not None

    def test_prefix_match(self):
        from optimization.kv_cache import PrefixCache
        pc = PrefixCache()
        pc.store("The quick brown fox", {"kv": "cache1"})
        result, score = pc.match_prefix("The quick brown fox jumps")
        assert result is not None
        assert score > 0.5

    def test_prefix_no_match(self):
        from optimization.kv_cache import PrefixCache
        pc = PrefixCache()
        pc.store("Python programming", {"kv": "data"})
        result, score = pc.match_prefix("Java programming")
        assert result is None or score < 0.3

    def test_prefix_cleanup(self):
        from optimization.kv_cache import PrefixCache
        pc = PrefixCache(max_entries=2)
        pc.store("prefix1", "data1")
        pc.store("prefix2", "data2")
        pc.store("prefix3", "data3")  # should evict one
        assert pc.size <= 2
