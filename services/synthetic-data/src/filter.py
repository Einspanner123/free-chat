"""Quality filtering for synthetic data."""

import re
from typing import List, Dict, Any, Optional
from collections import Counter


class QualityFilter:
    """Filter synthetic data by quality criteria."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 2048,
        tokenizer=None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.tokenizer = tokenizer

    def filter(
        self,
        data: List[Dict],
        deduplicate: bool = True,
        remove_html: bool = True,
    ) -> List[Dict]:
        """Filter a dataset.

        Args:
            data: List of {instruction, output} or {messages} dicts.
            deduplicate: Remove exact duplicates.
            remove_html: Strip HTML tags from text.

        Returns:
            Filtered dataset.
        """
        if not data:
            return []

        if remove_html:
            data = [self._strip_html(item) for item in data]

        filtered = []
        seen = set()

        for item in data:
            if self._should_skip(item):
                continue

            key = self._get_key(item)
            if deduplicate and key in seen:
                continue
            seen.add(key)

            filtered.append(item)

        return filtered

    def _should_skip(self, item: Dict) -> bool:
        """Check if an item should be skipped."""
        text = self._get_text(item)

        # Length check
        if len(text) < self.min_length:
            return True
        if len(text) > self.max_length:
            return True

        # Token length check
        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_length:
                return True

        # Empty input or output
        instruction = item.get("instruction", item.get("input", ""))
        output = item.get("output", "")
        messages = item.get("messages", [])
        if not messages and (not instruction or not output):
            return True

        # Repetition check
        if self._is_repetitive(text):
            return True

        # Input-output overlap check
        if instruction and output:
            overlap = self._compute_overlap(instruction, output)
            if overlap > 0.8:
                return True

        return False

    def _is_repetitive(self, text: str) -> bool:
        """Check if text is repetitive (n-gram repetition)."""
        words = text.split()
        if len(words) < 10:
            return False
        # Check for repeated 3-grams
        tri_grams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if not tri_grams:
            return False
        counts = Counter(tri_grams)
        most_common = counts.most_common(1)
        if most_common and most_common[0][1] > len(tri_grams) * 0.5:
            return True
        return False

    def _compute_overlap(self, text_a: str, text_b: str) -> float:
        """Compute token overlap ratio between two texts."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        return len(intersection) / max(len(tokens_a), len(tokens_b))

    def _strip_html(self, item: Dict) -> Dict:
        """Remove HTML tags from all text fields."""
        result = {}
        for key, value in item.items():
            if isinstance(value, str):
                result[key] = re.sub(r'<[^>]+>', '', value)
            elif isinstance(value, list):
                result[key] = value
            else:
                result[key] = value
        return result

    def _get_text(self, item: Dict) -> str:
        """Get concatenated text from item."""
        parts = []
        if "messages" in item:
            for m in item["messages"]:
                parts.append(str(m.get("content", "")))
        else:
            parts.append(str(item.get("instruction", "")))
            parts.append(str(item.get("input", "")))
            parts.append(str(item.get("output", "")))
        return " ".join(parts)

    @staticmethod
    def _get_key(item: Dict) -> str:
        """Get a deduplication key."""
        return str(item.get("instruction", "")) + "|||" + str(item.get("output", ""))


class FilterChain:
    """Chain multiple filters together."""

    def __init__(self, filters: List[QualityFilter]):
        self.filters = filters

    def apply(self, data: List[Dict]) -> List[Dict]:
        for f in self.filters:
            data = f.filter(data)
        return data
