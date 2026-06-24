"""Text splitter — chunk documents for embedding and retrieval.

LangChain equivalent: 8,277 lines across text-splitters/ package.
infy: ~100 lines (3 strategies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from infy.documents import Document, DocumentChunk


@dataclass
class TextSplitter:
    """Split text into chunks by character count with overlap.

    The simplest and most common splitter. Splits on sentence boundaries
    when possible to preserve semantic units.

    Usage:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text("Long document text...")
        doc_chunks = splitter.split_documents([doc])
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None

    def __post_init__(self) -> None:
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        current = ""

        for sentence in self._split_sentences(text):
            # A single sentence larger than the budget can't fit any chunk, so
            # hard-split it rather than letting it blow past chunk_size.
            if len(sentence) > self.chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._hard_split(sentence.strip()))
                continue

            if len(current) + len(sentence) <= self.chunk_size:
                current += sentence
            else:
                if current:
                    chunks.append(current.strip())
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = chunks[-1][-self.chunk_overlap :]
                    current = overlap_text + sentence
                else:
                    current = sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Split a string into fixed-size pieces when no separator helps."""
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def split_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            total = len(text_chunks)
            for i, chunk_text in enumerate(text_chunks):
                start = doc.page_content.find(chunk_text[:50])
                chunks.append(
                    DocumentChunk(
                        document=doc,
                        chunk_index=i,
                        total_chunks=total,
                        start_char=max(0, start) if start >= 0 else 0,
                        end_char=min(len(doc.page_content), start + len(chunk_text))
                        if start >= 0
                        else len(chunk_text),
                    )
                )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, trying to preserve sentence boundaries."""
        # Try splitting on sentence-ending punctuation
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) > 1:
            return [p + " " for p in parts if p]

        # Fallback: split on the first available separator
        for sep in self.separators or []:
            if sep and sep in text:
                parts = text.split(sep)
                return [p + sep for p in parts[:-1]] + [parts[-1]]

        return [text]


@dataclass
class RecursiveTextSplitter:
    """Recursively split text using a hierarchy of separators.

    Tries largest separator first, falls back to smaller ones.
    This preserves document structure better than simple splitting.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None

    def __post_init__(self) -> None:
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        return self._split_recursive(text, self.separators or [])

    def split_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            total = len(text_chunks)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append(
                    DocumentChunk(
                        document=doc,
                        chunk_index=i,
                        total_chunks=total,
                        start_char=0,
                        end_char=len(chunk_text),
                    )
                )
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Last resort: hard split
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            test = current + sep + part if current else part
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_recursive(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks


@dataclass
class MarkdownTextSplitter:
    """Split markdown documents respecting heading structure."""

    chunk_size: int = 1000
    chunk_overlap: int = 200

    def split_text(self, text: str) -> list[str]:
        # Split on markdown headings
        sections = re.split(r"(?m)^(#{1,6}\s+.+)$", text)

        chunks: list[str] = []
        current = ""

        for section in sections:
            if len(current) + len(section) <= self.chunk_size:
                current += section
            else:
                if current:
                    chunks.append(current.strip())
                current = section

        if current.strip():
            chunks.append(current.strip())

        # Further split any oversized chunks
        if chunks and any(len(c) > self.chunk_size for c in chunks):
            splitter = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            result = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    result.extend(splitter.split_text(chunk))
                else:
                    result.append(chunk)
            return result

        return chunks

    def split_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            total = len(text_chunks)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append(
                    DocumentChunk(
                        document=doc,
                        chunk_index=i,
                        total_chunks=total,
                        start_char=0,
                        end_char=len(chunk_text),
                    )
                )
        return chunks
