"""
DuckDuckGo search provider via the `ddgs` package.

No API key. The ddgs package is lazy-imported so this module imports cleanly
even when ddgs is not installed — is_available() reflects importability and
search() degrades to an error envelope. The blocking ddgs call runs in a
worker thread with a hard wall-clock cap so a rate-limited or hung search
cannot stall the pipeline (pattern borrowed from hermes' ddgs provider).
"""

import concurrent.futures
from typing import Any, Dict, List

from search.provider import WebSearchProvider

_SEARCH_TIMEOUT_SECS = 30


def _has_ddgs() -> bool:
    """Cheap import probe — no network I/O. Module-level so tests can patch it."""
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        return False


def _run_ddgs_search(query: str, safe_limit: int) -> List[Dict[str, Any]]:
    """Run the blocking ddgs query and return normalized hits.

    Module-level so tests can patch it directly without spawning a real
    network call / worker thread.
    """
    from ddgs import DDGS

    results: List[Dict[str, Any]] = []
    with DDGS(timeout=10) as client:
        for i, hit in enumerate(client.text(query, max_results=safe_limit)):
            if i >= safe_limit:
                break
            results.append(
                {
                    "title": str(hit.get("title", "")),
                    "url": str(hit.get("href") or hit.get("url") or ""),
                    "description": str(hit.get("body", "")),
                    "position": i + 1,
                }
            )
    return results


class DDGSWebSearchProvider(WebSearchProvider):
    """DuckDuckGo search via the ddgs package — free, no API key."""

    @property
    def name(self) -> str:
        return "ddgs"

    @property
    def display_name(self) -> str:
        return "DuckDuckGo (ddgs)"

    def is_available(self) -> bool:
        return _has_ddgs()

    def search(self, query: str, limit: int = 5) -> Dict:
        if not _has_ddgs():
            return {"success": False, "error": "ddgs package is not installed"}

        safe_limit = max(1, int(limit))
        # A fresh single-worker pool per call: a hung ddgs call cannot be
        # cancelled, so a shared pool would serialise every later search
        # behind it. Per-call pools isolate each search.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_run_ddgs_search, query, safe_limit)
            try:
                web = future.result(timeout=_SEARCH_TIMEOUT_SECS)
            except concurrent.futures.TimeoutError:
                return {
                    "success": False,
                    "error": f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s",
                }
        except Exception as exc:
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}
        finally:
            # On timeout the worker runs to completion on its own; it writes
            # nothing shared, so leaking it is safe.
            pool.shutdown(wait=False, cancel_futures=True)

        return {"success": True, "data": {"web": web}}
