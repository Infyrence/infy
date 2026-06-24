"""Document types — the unit of retrieval.

LangChain equivalent: 500 lines across documents/base.py + content blocks.
infy: ~80 lines.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A piece of content with metadata. The unit of retrieval and indexing.

    Usage:
        doc = Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "docs/intro.md", "page": 1},
        )
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if "id" not in self.metadata:
            self.metadata["id"] = self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "id": self.id,
        }

    def merge_metadata(self, other: Document) -> Document:
        """Create a new document with merged metadata."""
        return Document(
            page_content=self.page_content,
            metadata={**self.metadata, **other.metadata},
            id=self.id,
        )

    def __repr__(self) -> str:
        preview = self.page_content[:80]
        if len(self.page_content) > 80:
            preview += "..."
        return f"Document(page_content={preview!r}, metadata={self.metadata})"


@dataclass
class DocumentChunk:
    """A chunk of a document with position info for reconstruction."""

    document: Document
    chunk_index: int
    total_chunks: int
    start_char: int = 0
    end_char: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def page_content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.document.metadata,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    def to_document(self) -> Document:
        """Convert back to a plain Document."""
        return Document(
            page_content=self.page_content,
            metadata=self.metadata,
        )
