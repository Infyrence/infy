"""RAG chain — retrieve, augment, generate.

LangChain equivalent: scattered across chains/, retrieval/, retrievers/.
infy: ~80 lines. One function that does it all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infy.documents import Document


@dataclass
class RAGResult:
    """Result of a RAG invocation."""

    response: str
    source_documents: list[Document]
    query: str
    usage: dict[str, Any] | None = None


@dataclass
class RAGChain:
    """A complete RAG pipeline: retrieve → build prompt → generate.

    Usage:
        from infy.rag import RAGChain
        from infy.retrievers import VectorStoreRetriever
        from infy.providers.openai import OpenAIChat

        rag = RAGChain(
            retriever=retriever,
            model=OpenAIChat("gpt-4o"),
        )

        result = rag.invoke("What is LangChain?")
        print(result.response)
        print(f"Sources: {[d.metadata.get('source') for d in result.source_documents]}")
    """

    retriever: Any  # Retriever
    model: Any  # ChatModel
    system_prompt: str = (
        "You are a helpful assistant. Answer the question based on the provided context.\n"
        "If the context doesn't contain enough information, say so honestly.\n\n"
        "Context:\n{context}"
    )
    k: int = 4
    include_sources: bool = True

    def invoke(self, query: str) -> RAGResult:
        """Run the full RAG pipeline."""
        from infy.messages import HumanMessage, SystemMessage

        # Step 1: Retrieve
        source_docs = self.retriever.get_relevant_documents(query)

        # Step 2: Build context
        context = self._build_context(source_docs)

        # Step 3: Generate
        prompt = [
            SystemMessage(content=self.system_prompt.format(context=context)),
            HumanMessage(content=query),
        ]

        response = self.model.generate(prompt)

        return RAGResult(
            response=response.text,
            source_documents=source_docs,
            query=query,
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            }
            if response.usage
            else None,
        )

    async def ainvoke(self, query: str) -> RAGResult:
        """Async version of invoke."""
        from infy.messages import HumanMessage, SystemMessage

        source_docs = self.retriever.get_relevant_documents(query)

        context = self._build_context(source_docs)

        prompt = [
            SystemMessage(content=self.system_prompt.format(context=context)),
            HumanMessage(content=query),
        ]

        response = await self.model.agenerate(prompt)

        return RAGResult(
            response=response.text,
            source_documents=source_docs,
            query=query,
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            }
            if response.usage
            else None,
        )

    def _build_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{doc.page_content}")

        return "\n\n".join(parts)
