"""
FAISS vector store using LangChain integration.
Uses LangChain's FAISS wrapper and HuggingFace embeddings
for document storage, retrieval, and management.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config import FAISS_INDEX_DIR, EMBEDDING_MODEL, TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages LangChain FAISS vector store for the RAG pipeline."""

    INDEX_NAME = "rag_index"

    def __init__(self):
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._store: Optional[FAISS] = None
        self._doc_registry: dict[str, dict] = {}  # doc_id -> {filename, file_type, chunk_count}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-load the FAISS store on first access."""
        if self._initialized:
            return
        self._initialized = True
        self._load_store()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the HuggingFace embedding model via LangChain."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully.")
        return self._embeddings

    def _index_path(self) -> Path:
        return FAISS_INDEX_DIR

    def _load_store(self):
        """Load existing FAISS index from disk, or start empty."""
        index_path = self._index_path() / f"{self.INDEX_NAME}.faiss"
        if index_path.exists():
            try:
                logger.info("Loading existing FAISS index from disk...")
                self._store = FAISS.load_local(
                    folder_path=str(self._index_path()),
                    embeddings=self.embeddings,
                    index_name=self.INDEX_NAME,
                    allow_dangerous_deserialization=True,
                )
                # Rebuild doc registry from stored metadata
                self._rebuild_registry()
                logger.info("FAISS index loaded with %d documents.", len(self._doc_registry))
            except Exception as e:
                logger.warning("Failed to load FAISS index: %s. Starting fresh.", e)
                self._store = None
                self._doc_registry = {}
        else:
            logger.info("No existing FAISS index found. Starting fresh.")
            self._store = None
            self._doc_registry = {}

    def _save_store(self):
        """Persist the FAISS index to disk."""
        if self._store is not None:
            self._store.save_local(
                folder_path=str(self._index_path()),
                index_name=self.INDEX_NAME,
            )

    def _rebuild_registry(self):
        """Rebuild the document registry from the FAISS store's metadata."""
        self._doc_registry = {}
        if self._store is None:
            return

        # Access the docstore to scan all documents
        for doc_id_key in self._store.docstore._dict:
            doc = self._store.docstore._dict[doc_id_key]
            if hasattr(doc, "metadata"):
                meta_doc_id = doc.metadata.get("doc_id", "unknown")
                if meta_doc_id not in self._doc_registry:
                    self._doc_registry[meta_doc_id] = {
                        "doc_id": meta_doc_id,
                        "filename": doc.metadata.get("filename", "unknown"),
                        "file_type": doc.metadata.get("file_type", "unknown"),
                        "chunk_count": 0,
                    }
                self._doc_registry[meta_doc_id]["chunk_count"] += 1

    def add_documents(self, doc_id: str, chunks: list[Document]) -> int:
        """
        Add LangChain Document chunks to the FAISS vector store.

        Args:
            doc_id: Unique document identifier.
            chunks: List of LangChain Document objects with metadata.

        Returns:
            Number of chunks added.
        """
        self._ensure_initialized()

        if not chunks:
            return 0

        if self._store is None:
            # Create a new FAISS store from the first batch
            self._store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings,
            )
        else:
            # Add to existing store
            self._store.add_documents(documents=chunks)

        self._save_store()

        # Update the registry
        first_chunk = chunks[0]
        self._doc_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": first_chunk.metadata.get("filename", "unknown"),
            "file_type": first_chunk.metadata.get("file_type", "unknown"),
            "chunk_count": len(chunks),
        }

        return len(chunks)

    def as_retriever(self, top_k: Optional[int] = None):
        """
        Return a LangChain retriever interface for the FAISS store.

        Args:
            top_k: Number of results to retrieve.

        Returns:
            LangChain VectorStoreRetriever.
        """
        self._ensure_initialized()

        if self._store is None:
            return None

        k = top_k or TOP_K
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

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
            List of (Document, score) tuples.
        """
        self._ensure_initialized()

        if self._store is None:
            return []

        k = top_k or TOP_K
        return self._store.similarity_search_with_score(query=query, k=k)

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document and rebuild the index.

        Args:
            doc_id: Document identifier.

        Returns:
            Number of chunks deleted.
        """
        self._ensure_initialized()

        if self._store is None:
            return 0

        # Find all docstore keys for this doc_id
        keys_to_delete = []
        for key, doc in self._store.docstore._dict.items():
            if hasattr(doc, "metadata") and doc.metadata.get("doc_id") == doc_id:
                keys_to_delete.append(key)

        if not keys_to_delete:
            return 0

        deleted_count = len(keys_to_delete)

        # Collect remaining documents
        remaining_docs = []
        for key, doc in self._store.docstore._dict.items():
            if key not in keys_to_delete:
                remaining_docs.append(doc)

        if remaining_docs:
            # Rebuild from remaining documents
            self._store = FAISS.from_documents(
                documents=remaining_docs,
                embedding=self.embeddings,
            )
        else:
            # No documents left — reset store
            self._store = None

        self._save_store()

        # Update registry
        self._doc_registry.pop(doc_id, None)

        return deleted_count

    def get_all_documents(self) -> list[dict]:
        """Get a summary of all unique documents in the store."""
        self._ensure_initialized()
        return list(self._doc_registry.values())

    def get_document_count(self) -> int:
        """Get total number of unique documents."""
        self._ensure_initialized()
        return len(self._doc_registry)




# Singleton instance — no heavy work happens here now
vector_store = VectorStore()

