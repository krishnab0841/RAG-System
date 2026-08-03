"""
In-memory vector store using HuggingFace Inference API embeddings.
Replaces FAISS + local PyTorch embeddings with a lightweight,
Vercel-compatible approach using LangChain's InMemoryVectorStore
and HuggingFace Inference API for remote embedding generation.
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import HUGGINGFACE_API_KEY, EMBEDDING_MODEL, TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages an in-memory vector store with HuggingFace API embeddings."""

    def __init__(self):
        self._embeddings = None
        self._store: Optional[InMemoryVectorStore] = None
        self._doc_registry: dict[str, dict] = {}  # doc_id -> {filename, file_type, chunk_count}
        self._runtime_hf_key: str = ""  # Can be set via settings API
        # Track all documents with their store IDs for deletion
        self._doc_store_ids: dict[str, list[str]] = {}  # doc_id -> [store_ids]

    def set_api_key(self, key: str):
        """Update the HuggingFace API key at runtime."""
        self._runtime_hf_key = key
        # Force re-initialization of embeddings with new key
        self._embeddings = None

    def _get_api_key(self) -> str:
        """Resolve HuggingFace API key from runtime settings or config."""
        if self._runtime_hf_key:
            return self._runtime_hf_key
        return HUGGINGFACE_API_KEY

    @property
    def embeddings(self) -> HuggingFaceEndpointEmbeddings:
        """Lazy-load the HuggingFace Inference API embedding model."""
        if self._embeddings is None:
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError(
                    "HuggingFace API key is required for embeddings. "
                    "Set HUGGINGFACE_API_KEY in your environment or configure it in Settings."
                )
            logger.info("Initializing HuggingFace Inference API embeddings: %s", EMBEDDING_MODEL)
            self._embeddings = HuggingFaceEndpointEmbeddings(
                model=EMBEDDING_MODEL,
                huggingfacehub_api_token=api_key,
            )
            logger.info("Embedding model initialized successfully (API-based).")
        return self._embeddings

    def _ensure_store(self):
        """Ensure the in-memory store is initialized."""
        if self._store is None:
            self._store = InMemoryVectorStore(embedding=self.embeddings)

    def add_documents(self, doc_id: str, chunks: list[Document]) -> int:
        """
        Add LangChain Document chunks to the in-memory vector store.

        Args:
            doc_id: Unique document identifier.
            chunks: List of LangChain Document objects with metadata.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        self._ensure_store()

        # Add documents and track the returned IDs
        store_ids = self._store.add_documents(documents=chunks)
        self._doc_store_ids[doc_id] = store_ids

        # Update the registry
        first_chunk = chunks[0]
        self._doc_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": first_chunk.metadata.get("filename", "unknown"),
            "file_type": first_chunk.metadata.get("file_type", "unknown"),
            "chunk_count": len(chunks),
        }

        return len(chunks)

    def similarity_search_with_score(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for relevant documents with similarity scores.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples. Score is a distance
            (lower = more similar) to maintain compatibility with
            the existing RAG engine score conversion.
        """
        if self._store is None:
            return []

        k = top_k or TOP_K

        # InMemoryVectorStore.similarity_search_with_score returns
        # (doc, similarity_score) where higher = more similar.
        # Convert to distance-style score for backward compat with
        # the RAG engine's formula: similarity = 1 / (1 + distance)
        results = self._store.similarity_search_with_score(query=query, k=k)

        converted = []
        for doc, similarity in results:
            # Convert similarity (0..1) → distance so that 1/(1+distance) ≈ similarity
            if similarity > 0:
                distance = (1.0 / max(similarity, 1e-6)) - 1.0
            else:
                distance = 100.0  # Very dissimilar
            converted.append((doc, distance))

        return converted

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Number of chunks deleted.
        """
        if self._store is None:
            return 0

        store_ids = self._doc_store_ids.get(doc_id, [])
        if not store_ids:
            return 0

        deleted_count = len(store_ids)

        # Delete by IDs from the store
        try:
            self._store.delete(ids=store_ids)
        except Exception as e:
            logger.warning("Failed to delete from store by IDs: %s", e)

        # Clean up registry
        self._doc_store_ids.pop(doc_id, None)
        self._doc_registry.pop(doc_id, None)

        return deleted_count

    def get_all_documents(self) -> list[dict]:
        """Get a summary of all unique documents in the store."""
        return list(self._doc_registry.values())

    def get_document_count(self) -> int:
        """Get total number of unique documents."""
        return len(self._doc_registry)


# Singleton instance
vector_store = VectorStore()
