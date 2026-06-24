"""Streaming utilities — helpers for working with streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def stream_text(stream: Iterator[Any]) -> str:
    """Consume a stream and return the full accumulated text."""
    parts: list[str] = []
    for chunk in stream:
        if hasattr(chunk, "text"):
            parts.append(chunk.text)
        elif hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                parts.append(content)
        elif isinstance(chunk, str):
            parts.append(chunk)
    return "".join(parts)


def stream_to_list(stream: Iterator[Any]) -> list[Any]:
    """Consume a stream into a list."""
    return list(stream)


def merge_chunks(chunks: list[Any]) -> Any:
    """Merge a list of streaming chunks into one accumulated result."""
    if not chunks:
        return None
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result + chunk
    return result


def tee(stream: Iterator[Any], n: int = 2) -> tuple[Iterator[Any], ...]:
    """Split a stream into n copies. Each copy can be consumed independently."""
    from itertools import tee as _tee

    return _tee(stream, n)
