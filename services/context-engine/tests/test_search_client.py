"""
Tests for WebSearchClient — the facade the pipeline uses.

It resolves a provider (explicit provider, preferred name, or the first
available one), calls search(), and returns a flat list of hits. Failures
(no provider, error envelope, empty results) surface as [] so the pipeline
can degrade gracefully without try/except.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from search.client import WebSearchClient


class FakeProvider:
    name = "fake"

    def __init__(self, response=None):
        self._response = response or {"success": True, "data": {"web": [{"title": "t", "url": "http://x", "description": "d", "position": 1}]}}

    def is_available(self):
        return True

    def search(self, query, limit=5):
        self.last_query = query
        self.last_limit = limit
        return self._response


class TestWebSearchClient:
    def test_returns_hits_from_provider(self):
        client = WebSearchClient(provider=FakeProvider())
        hits = client.search("today news", limit=3)
        assert len(hits) == 1
        assert hits[0]["title"] == "t"

    def test_passes_query_and_limit_through(self):
        p = FakeProvider()
        client = WebSearchClient(provider=p)
        client.search("hello", limit=7)
        assert p.last_query == "hello"
        assert p.last_limit == 7

    def test_returns_empty_on_error_envelope(self):
        provider = FakeProvider(response={"success": False, "error": "boom"})
        client = WebSearchClient(provider=provider)
        assert client.search("q") == []

    def test_returns_empty_when_provider_missing(self):
        client = WebSearchClient(provider=None)
        # No registry provider is registered by default in tests → [].
        assert client.search("q") == []

    def test_accepts_provider_name_via_registry(self):
        from search.registry import register_provider, _registry
        p = FakeProvider()
        register_provider(p)
        try:
            client = WebSearchClient(provider="fake")
            hits = client.search("q")
            assert hits[0]["title"] == "t"
        finally:
            _registry._providers.pop("fake", None)
