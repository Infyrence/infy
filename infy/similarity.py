"""Similarity search — SIMD-accelerated via Rust core.

LangChain equivalent: 1,873 lines across vectorstores/ + utils.py with numpy fallback.
infy: ~40 lines, Rust SIMD when available.
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    import infy_core

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses Rust SIMD (AVX2) when available, falls back to pure Python.
    """
    if _HAS_RUST:
        return infy_core.cosine_similarity(list(a), list(b))
    return _py_cosine_similarity(a, b)


def batch_cosine_similarity(
    query: Sequence[float],
    documents: Sequence[Sequence[float]],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Find the most similar documents to a query vector.

    Returns list of (index, score) pairs sorted by score descending.
    """
    if _HAS_RUST:
        return infy_core.batch_cosine_similarity(list(query), [list(d) for d in documents], top_k)
    return _py_batch_cosine_similarity(query, documents, top_k)


def inner_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute inner product (dot product) of two vectors."""
    if _HAS_RUST:
        return infy_core.inner_product(list(a), list(b))
    return sum(x * y for x, y in zip(a, b, strict=False))


# --- Pure Python fallbacks ---


def _py_cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a)
    norm_b = sum(x * x for x in b)
    denom = (norm_a * norm_b) ** 0.5
    if denom < 1e-10:
        return 0.0
    return float(dot / denom)


def _py_batch_cosine_similarity(
    query: Sequence[float],
    documents: Sequence[Sequence[float]],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    results = [(i, _py_cosine_similarity(query, d)) for i, d in enumerate(documents)]
    results.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        results = results[:top_k]
    return results
