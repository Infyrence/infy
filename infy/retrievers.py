"""Retriever — fetch relevant documents for a query.

LangChain equivalent: 328 lines base + implementations.
infy: ~80 lines (protocol + vector store retriever).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from infy.documents import Document


@runtime_checkable
class Retriever(Protocol):
    """Interface for document retrieval."""

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Retrieve documents relevant to the query."""
        ...

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        """Async version of get_relevant_documents."""
        ...


@dataclass
class VectorStoreRetriever:
    """Retriever that searches a vector store using embeddings.

    Usage:
        from infy.retrievers import VectorStoreRetriever
        from infy.vectorstores import InMemoryVectorStore
        from infy.embeddings import FakeEmbeddings

        store = InMemoryVectorStore()
        embeddings = FakeEmbeddings()
        # ... add documents to store ...

        retriever = VectorStoreRetriever(store=store, embeddings=embeddings)
        docs = retriever.get_relevant_documents("What is LangChain?")
    """

    store: Any  # VectorStore
    embeddings: Any  # Embeddings
    k: int = 4
    score_threshold: float | None = None

    def get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.store.similarity_search(query_embedding, k=self.k)

        if self.score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= self.score_threshold]

        return [doc for doc, _ in results]

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


@dataclass
class MultiQueryRetriever:
    """Generates multiple query variations and merges results for better recall.

    Uses an LLM to generate alternative queries from the original.
    """

    retriever: VectorStoreRetriever
    model: Any = None  # ChatModel
    num_queries: int = 3

    def get_relevant_documents(self, query: str) -> list[Document]:
        if self.model is None:
            return self.retriever.get_relevant_documents(query)

        # Generate alternative queries
        from infy.messages import HumanMessage, SystemMessage

        prompt = [
            SystemMessage(
                content=(
                    "Generate alternative search queries. Return only the queries, "
                    "one per line, no numbering or bullets."
                )
            ),
            HumanMessage(
                content=(
                    f"Original query: {query}\n\nGenerate {self.num_queries} alternative queries:"
                )
            ),
        ]

        response = self.model.generate(prompt)
        alternative_queries = [q.strip() for q in response.text.strip().split("\n") if q.strip()]

        # Search with all queries and deduplicate
        all_docs: dict[str, Document] = {}
        for q in [query] + alternative_queries:
            for doc in self.retriever.get_relevant_documents(q):
                all_docs[doc.id] = doc

        return list(all_docs.values())

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
