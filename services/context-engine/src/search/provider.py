"""
WebSearchProvider ABC.

Design borrowed from hermes' agent/web_search_provider.py: every backend
implements a fixed contract and returns a fixed response envelope, so the
caller (registry / client / pipeline) never translates between backends.

Response envelopes:
    search success: {"success": True, "data": {"web": [{title, url, description, position}]}}
    any failure:    {"success": False, "error": "human-readable message"}

is_available() must be a cheap probe (import / env presence). It MUST NOT
perform network I/O — it is called on every resolution.
"""

from abc import ABC, abstractmethod
from typing import Dict


class WebSearchProvider(ABC):
    """A search backend. Subclass and override name / is_available / search."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable id used in config (search_provider). Lowercase, no spaces."""

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap availability gate — no network I/O."""

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> Dict:
        """Search and return a fixed response envelope (see module docstring)."""
