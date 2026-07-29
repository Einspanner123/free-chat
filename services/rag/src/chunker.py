"""Document chunking strategies."""

import re
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[str]:
        ...


class RecursiveChunker(BaseChunker):
    """Recursive chunking that splits text by separators recursively."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        if not text:
            return []
        if separators is None:
            separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
        return self._chunk_recursive(text, separators)

    def chunk_with_metadata(self, text: str) -> List[Dict]:
        chunks = self.chunk(text)
        return [{"text": c, "index": i, "len": len(c)} for i, c in enumerate(chunks)]

    def _chunk_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        result = []
        for sep in separators:
            if sep == "":
                # Character-level split
                chunks = []
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk = text[i:i + self.chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                return chunks

            parts = text.split(sep)
            if len(parts) > 1:
                current = ""
                for part in parts:
                    if not current:
                        current = part
                    elif len(current) + len(sep) + len(part) <= self.chunk_size:
                        current += sep + part
                    else:
                        if current.strip():
                            result.append(current)
                        # Overlap: keep last part of current chunk
                        overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                        current = overlap_text + sep + part if overlap_text else part

                if current.strip():
                    result.append(current)
                return result

        return [text]


class SemanticChunker(BaseChunker):
    """Semantic chunking based on sentences, paragraphs, or topics."""

    def chunk(self, text: str, **kwargs) -> List[str]:
        return self.chunk_by_sentence(text)

    def chunk_by_sentence(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    def chunk_by_paragraph(self, text: str) -> List[str]:
        paras = re.split(r'\n\s*\n', text.strip())
        return [p for p in paras if p]

    def chunk_by_topic(self, text: str) -> List[str]:
        topics = re.split(r'(?=^#\s)', text.strip(), flags=re.MULTILINE)
        return [t for t in topics if t]


class ChunkerFactory:
    @staticmethod
    def create(strategy: str, **kwargs) -> BaseChunker:
        if strategy == "recursive":
            return RecursiveChunker(**{k: v for k, v in kwargs.items() if k in ("chunk_size", "chunk_overlap")})
        elif strategy == "semantic":
            return SemanticChunker()
        else:
            raise ValueError(f"Unknown chunker strategy: {strategy}")
