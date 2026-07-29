from dataclasses import dataclass, field
from typing import List, Optional

_VALID_STRATEGIES = {"dense", "sparse", "hybrid"}

@dataclass
class ChunkerConfig:
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", "!", "?", " ", ""])

@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 384
    retrieval_strategy: str = "hybrid"
    dense_weight: float = 0.5
    max_context_tokens: int = 2048
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)

    def __post_init__(self):
        strategies = {"dense", "sparse", "hybrid"}
        if self.retrieval_strategy not in strategies:
            raise ValueError(f"strategy must be {strategies}, got '{self.retrieval_strategy}'")
