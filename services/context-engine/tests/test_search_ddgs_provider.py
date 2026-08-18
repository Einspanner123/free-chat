"""
Tests for the DDGS web-search provider.

The ddgs package is lazy-imported and never touched by tests: is_available
is a cheap import probe, and search() runs in a worker thread with a hard
wall-clock timeout (mirrors hermes' ddgs provider). Responses use the
fixed hermes envelope:
    success: {"success": True, "data": {"web": [{title, url, description, position}]}}
    failure: {"success": False, "error": str}
"""

import os
import sys
import types
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import search.ddgs_provider as ddgs_provider


def _install_fake_ddgs(monkeypatch, hit=None, raise_exc=None):
    """Install a fake `ddgs` module with a DDGS client returning `hit`.

    A module-level function so both is_available() and search() see the same
    fake through sys.modules.
    """
    fake = types.ModuleType("ddgs")

    class FakeClient:
        def __init__(self, timeout=10):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=5):
            if raise_exc is not None:
                raise raise_exc
            yield {"title": "T", "href": "http://example.com/a", "body": "snippet"}

    fake.DDGS = FakeClient
    monkeypatch.setitem(sys.modules, "ddgs", fake)


class TestAvailability:
    def test_is_available_true_when_ddgs_importable(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        assert ddgs_provider.DDGSWebSearchProvider().is_available() is True

    def test_is_available_false_without_ddgs(self, monkeypatch):
        monkeypatch.setattr(ddgs_provider, "_has_ddgs", lambda: False)
        assert ddgs_provider.DDGSWebSearchProvider().is_available() is False


class TestSearch:
    def test_normalizes_hits_into_envelope(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        resp = ddgs_provider.DDGSWebSearchProvider().search("today news", limit=3)
        assert resp["success"] is True
        web = resp["data"]["web"]
        assert len(web) == 1
        assert web[0] == {
            "title": "T",
            "url": "http://example.com/a",
            "description": "snippet",
            "position": 1,
        }

    def test_returns_error_envelope_when_ddgs_missing(self, monkeypatch):
        monkeypatch.setattr(ddgs_provider, "_has_ddgs", lambda: False)
        resp = ddgs_provider.DDGSWebSearchProvider().search("anything")
        assert resp["success"] is False
        assert "not installed" in resp["error"]

    def test_returns_error_envelope_on_provider_exception(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, raise_exc=RuntimeError("rate limited"))
        resp = ddgs_provider.DDGSWebSearchProvider().search("anything")
        assert resp["success"] is False
        assert "rate limited" in resp["error"]

    def test_times_out_after_wall_clock_cap(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)

        def slow_search(query, limit):
            time.sleep(1.0)  # block past the patched timeout
            return []

        monkeypatch.setattr(ddgs_provider, "_run_ddgs_search", slow_search)
        monkeypatch.setattr(ddgs_provider, "_SEARCH_TIMEOUT_SECS", 0.1)
        resp = ddgs_provider.DDGSWebSearchProvider().search("anything")
        assert resp["success"] is False
        assert "timed out" in resp["error"]
