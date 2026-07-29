"""Full RAG pipeline: ingest → chunk → embed → index → retrieve → generate."""

import os
import uuid
from typing import List, Dict, Any, Optional, Iterator

from loguru import logger

from config import RAGConfig
from chunker import ChunkerFactory, RecursiveChunker
from embedding import EmbeddingModel
from retriever import DenseRetriever, BM25Retriever, HybridRetriever
from vector_store import InMemoryVectorStore


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest("Large document text...")
        result = pipeline.query("What is this about?")
        print(result["answer"])
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._chunker = ChunkerFactory.create(
            self.config.chunker.strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._embedding_model = EmbeddingModel(
            model_name=self.config.embedding_model,
            dimension=self.config.embedding_dim,
        )
        self._vector_store = InMemoryVectorStore(dimension=self.config.embedding_dim)
        self._sparse_retriever = BM25Retriever()
        self._llm_engine = None  # Set externally or via set_llm()

        # Build retrievers
        self._dense_retriever = DenseRetriever(self._vector_store, self._embedding_model)
        self._hybrid_retriever = HybridRetriever(
            dense_retriever=self._dense_retriever,
            sparse_retriever=self._sparse_retriever,
            dense_weight=self.config.dense_weight,
        )

    def set_llm(self, engine):
        """Set the LLM engine for generation."""
        self._llm_engine = engine

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, text: str, doc_id: Optional[str] = None) -> str:
        """Ingest a document: chunk, embed, and store.

        Args:
            text: Document text.
            doc_id: Optional document ID.

        Returns:
            The document ID.
        """
        if not text:
            return ""
        doc_id = doc_id or str(uuid.uuid4())
        chunks = self._chunker.chunk(text)

        if not chunks:
            return doc_id

        # Embed and store each chunk
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        vectors = self._embedding_model.embed_batch(chunks)
        metadatas = [{"doc_id": doc_id, "text": chunk, "chunk_index": i}
                     for i, chunk in enumerate(chunks)]

        self._vector_store.add_batch(chunk_ids, vectors, metadatas)

        # Also index for sparse retrieval
        sparse_docs = [{"id": cid, "text": chunk} for cid, chunk in zip(chunk_ids, chunks)]
        self._sparse_retriever.index(sparse_docs)

        logger.info(f"Ingested {len(chunks)} chunks from document {doc_id}")
        return doc_id

    def ingest_batch(self, texts: List[str]) -> List[str]:
        """Ingest multiple documents at once.

        Args:
            texts: List of document texts.

        Returns:
            List of document IDs.
        """
        return [self.ingest(t) for t in texts]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            List of result dicts.
        """
        k = k or self.config.top_k
        if self.config.retrieval_strategy == "dense":
            return self._dense_retriever.retrieve(query, k=k)
        elif self.config.retrieval_strategy == "sparse":
            return self._sparse_retriever.retrieve(query, k=k)
        else:
            return self._hybrid_retriever.retrieve(query, k=k)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """Build a RAG prompt with context.

        Args:
            query: User query.
            contexts: Retrieved context chunks.

        Returns:
            Formatted prompt string.
        """
        context_str = "\n\n".join(contexts) if contexts else "(No relevant context found.)"

        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context:
{context_str}

Question: {query}

Answer:"""
        return prompt

    def format_context(self, contexts: List[Dict], max_tokens: int = 2048) -> str:
        """Format retrieved contexts with token budget.

        Args:
            contexts: List of result dicts with metadata.text.
            max_tokens: Maximum total tokens for context.

        Returns:
            Formatted context string.
        """
        formatted = []
        total_chars = max_tokens * 2  # rough char-to-token ratio

        for ctx in contexts:
            text = ctx.get("metadata", {}).get("text", ctx.get("text", ""))
            if total_chars - len(text) < 0:
                # Truncate last context to fit
                text = text[:total_chars]
            if text:
                formatted.append(text)
                total_chars -= len(text)
            if total_chars <= 0:
                break

        return "\n\n".join(formatted)

    def query(self, query: str) -> Dict[str, Any]:
        """Run a full RAG query: retrieve + generate.

        Args:
            query: User query.

        Returns:
            Dict with answer, context, and sources.
        """
        if not query:
            return {"answer": "", "context": [], "sources": []}

        # Retrieve
        retrieved = self.retrieve(query)
        contexts = [r.get("metadata", {}).get("text", r.get("text", "")) for r in retrieved]

        # Build prompt
        prompt = self.build_prompt(query, contexts)

        # Generate
        answer = ""
        if self._llm_engine is not None:
            response = self._llm_engine.generate([{"role": "user", "content": prompt}])
            answer = response.chunk

        return {
            "answer": answer,
            "context": retrieved,
            "sources": [r.get("id", "") for r in retrieved],
        }

    def stream_query(self, query: str) -> Iterator:
        """Run RAG query with streaming generation.

        Yields:
            Generation chunks from the LLM engine.
        """
        if not query or self._llm_engine is None:
            return

        retrieved = self.retrieve(query)
        contexts = [r.get("metadata", {}).get("text", r.get("text", "")) for r in retrieved]
        prompt = self.build_prompt(query, contexts)

        yield from self._llm_engine.stream_generate([{"role": "user", "content": prompt}])

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self):
        """Clear all stored documents."""
        self._vector_store.clear()

    def document_count(self) -> int:
        return self._vector_store.count()
