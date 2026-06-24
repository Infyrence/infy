"""Embeddings — vectorize text for similarity search.

LangChain equivalent: 207 lines base + per-provider implementations.
infy: ~50 lines protocol + provider implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Embeddings(Protocol):
    """Interface for text embedding models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...


@dataclass
class FakeEmbeddings:
    """Deterministic embeddings for testing. Maps each unique string to a consistent vector."""

    dimension: int = 128

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def _hash_to_vector(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(self.dimension):
            byte_idx = i % len(h)
            vec.append((h[byte_idx] / 127.5) - 1.0)
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


@dataclass
class OpenAIEmbeddings:
    """OpenAI embedding model. Requires openai package."""

    model: str = "text-embedding-3-small"
    api_key: str | None = None
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = OpenAI(**kwargs)
        except ImportError:
            raise ImportError("OpenAI not installed. pip install infy[openai]") from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self.model)
        return list(response.data[0].embedding)


@dataclass
class OllamaEmbeddings:
    """Ollama local embedding model."""

    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import ollama as _ollama

            self._client = _ollama
        except ImportError:
            raise ImportError("Ollama not installed. pip install infy[ollama]") from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            response = self._client.embeddings(model=self.model, prompt=text)
            results.append(response["embedding"])
        return results

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings(model=self.model, prompt=text)
        return list(response["embedding"])
