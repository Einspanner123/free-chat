"""
WebSearchClient — the facade the pipeline calls.

Resolves a provider (explicit instance, preferred name, or the first
available one), runs search(), and returns a flat list of hits. Any failure
(no provider, error envelope, empty results) surfaces as [] so the pipeline
degrades gracefully without try/except.
"""

from typing import Dict, List, Optional

from search.registry import register_builtin_providers, resolve_search_provider
from search.provider import WebSearchProvider


class WebSearchClient:
    """Search facade. `provider` is a name string or a provider instance."""

    def __init__(self, provider: Optional[object] = None):
        self._provider = provider

    def _resolve(self) -> Optional[WebSearchProvider]:
        register_builtin_providers()
        if self._provider is None:
            return resolve_search_provider(None)
        if isinstance(self._provider, str):
            return resolve_search_provider(self._provider)
        return self._provider

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        provider = self._resolve()
        if provider is None:
            return []
        resp = provider.search(query, limit)
        if not resp or not resp.get("success"):
            return []
        return resp.get("data", {}).get("web", [])
