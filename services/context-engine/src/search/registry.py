"""
Web-search provider registry.

Design borrowed from hermes' agent/web_search_registry.py: providers
register by a stable `name`; resolution honors an explicit preference and
falls back to the first available provider. Never does network I/O.
"""

from typing import List, Optional

from search.provider import WebSearchProvider


class SearchProviderRegistry:
    """Stores providers by name and resolves the active one."""

    def __init__(self):
        self._providers: dict = {}

    def register(self, provider: WebSearchProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> Optional[WebSearchProvider]:
        return self._providers.get(name)

    def list(self) -> List[WebSearchProvider]:
        return list(self._providers.values())

    def resolve(self, preferred: Optional[str] = None) -> Optional[WebSearchProvider]:
        """Explicit preference wins when registered AND available; else the
        first available provider; else None."""
        if preferred:
            p = self._providers.get(preferred)
            if p is not None and p.is_available():
                return p
        for p in self._providers.values():
            if p.is_available():
                return p
        return None


_registry = SearchProviderRegistry()


def register_provider(provider: WebSearchProvider) -> None:
    _registry.register(provider)


def get_provider(name: str) -> Optional[WebSearchProvider]:
    return _registry.get(name)


def list_providers() -> List[WebSearchProvider]:
    return _registry.list()


def resolve_search_provider(preferred: Optional[str] = None) -> Optional[WebSearchProvider]:
    return _registry.resolve(preferred)


def register_builtin_providers() -> None:
    """Register the shipped providers. Idempotent."""
    from search.ddgs_provider import DDGSWebSearchProvider
    if _registry.get("ddgs") is None:
        register_provider(DDGSWebSearchProvider())
