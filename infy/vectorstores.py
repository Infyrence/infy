"""Vector store — store and search document embeddings.

LangChain equivalent: 1,873 lines across vectorstores/ (15+ implementations).
infy: ~120 lines (protocol + in-memory with Rust SIMD similarity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from infy.documents import Document


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector similarity search stores."""

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        """Add documents with pre-computed embeddings. Returns IDs."""
        ...

    def similarity_search(
        self, query_embedding: list[float], k: int = 4
    ) -> list[tuple[Document, float]]:
        """Find the k most similar documents. Returns (doc, score) pairs."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...

    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


@dataclass
class InMemoryVectorStore:
    """In-memory vector store using Rust SIMD cosine similarity.

    Perfect for prototyping and small-to-medium datasets (< 100K docs).
    For production, use a dedicated vector database.

    Usage:
        from infy.embeddings import FakeEmbeddings
        from infy.vectorstores import InMemoryVectorStore

        embeddings = FakeEmbeddings()
        store = InMemoryVectorStore()

        docs = [Document(page_content="Hello"), Document(page_content="World")]
        vectors = embeddings.embed_documents([d.page_content for d in docs])
        store.add_documents(docs, vectors)

        query_vec = embeddings.embed_query("Hello")
        results = store.similarity_search(query_vec, k=2)
    """

    _documents: list[Document] = field(default_factory=list)
    _embeddings: list[list[float]] = field(default_factory=list)
    _ids: list[str] = field(default_factory=list)

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        ids = []
        for doc, emb in zip(documents, embeddings, strict=True):
            self._documents.append(doc)
            self._embeddings.append(emb)
            self._ids.append(doc.id)
            ids.append(doc.id)
        return ids

    def similarity_search(
        self, query_embedding: list[float], k: int = 4
    ) -> list[tuple[Document, float]]:
        if not self._documents:
            return []

        # Use the batched (Rust SIMD when available) path: it ranks and truncates
        # in one call instead of crossing the Python/Rust boundary per document.
        from infy.similarity import batch_cosine_similarity

        ranked = batch_cosine_similarity(query_embedding, self._embeddings, top_k=k)
        return [(self._documents[idx], score) for idx, score in ranked]

    def delete(self, ids: list[str]) -> None:
        id_set = set(ids)
        new_docs = []
        new_embs = []
        new_ids = []
        for doc, emb, doc_id in zip(self._documents, self._embeddings, self._ids, strict=True):
            if doc_id not in id_set:
                new_docs.append(doc)
                new_embs.append(emb)
                new_ids.append(doc_id)
        self._documents = new_docs
        self._embeddings = new_embs
        self._ids = new_ids

    def count(self) -> int:
        return len(self._documents)

    def get_by_id(self, doc_id: str) -> Document | None:
        for doc, doc_id_val in zip(self._documents, self._ids, strict=True):
            if doc_id_val == doc_id:
                return doc
        return None

    def get_all(self) -> list[Document]:
        return list(self._documents)
