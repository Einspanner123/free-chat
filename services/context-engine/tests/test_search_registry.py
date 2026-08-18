"""
Tests for the web-search provider registry.

Borrowed design (not code) from hermes' web_search_registry: providers
register by a stable `name`, and resolution picks the configured provider
falling back to the first available one. The registry must never do
network I/O — availability checks are cheap (import / env probes).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


class FakeProvider:
    """Minimal WebSearchProvider stand-in for registry tests."""

    name = "fake"

    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available

    def search(self, query, limit=5):
        return {"success": True, "data": {"web": [{"title": query, "url": "http://x", "description": "", "position": 1}]}}


class FakeOfflineProvider:
    name = "offline"

    def is_available(self):
        return False

    def search(self, query, limit=5):
        return {"success": False, "error": "unreachable"}


class TestSearchProviderRegistry:
    """Registry stores providers by name and resolves the active one."""

    def test_register_and_get_by_name(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        p = FakeProvider()
        reg.register(p)
        assert reg.get("fake") is p

    def test_get_unknown_returns_none(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        assert reg.get("nope") is None

    def test_list_providers(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        reg.register(FakeProvider())
        reg.register(FakeOfflineProvider())
        assert {p.name for p in reg.list()} == {"fake", "offline"}

    def test_resolve_returns_preferred_when_available(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        preferred = FakeProvider()
        reg.register(preferred)
        reg.register(FakeOfflineProvider())
        assert reg.resolve(preferred="fake") is preferred

    def test_resolve_falls_back_when_preferred_unavailable(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        reg.register(FakeOfflineProvider())
        fallback = FakeProvider()
        reg.register(fallback)
        assert reg.resolve(preferred="offline") is fallback

    def test_resolve_returns_first_available(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        offline = FakeOfflineProvider()
        online = FakeProvider()
        reg.register(offline)
        reg.register(online)
        assert reg.resolve() is online

    def test_resolve_with_no_available_returns_none(self):
        from search.registry import SearchProviderRegistry
        reg = SearchProviderRegistry()
        reg.register(FakeOfflineProvider())
        assert reg.resolve() is None


class TestModuleLevelRegistry:
    """Module-level singletons register the built-in providers."""

    def test_register_provider_round_trip(self):
        import search.registry as registry_module
        p = FakeProvider()
        registry_module.register_provider(p)
        try:
            assert registry_module.get_provider("fake") is p
            assert p in registry_module.list_providers()
        finally:
            registry_module._registry._providers.pop("fake", None)
