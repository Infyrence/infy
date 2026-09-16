"""Tool routing — keep a compact catalogue resident, promote only the schemas a turn needs.

Injecting every tool schema on every turn is the dominant context cost of a large tool
catalogue. The schemas are re-sent on every iteration of the agent loop, they crowd out the
conversation, and model accuracy degrades as the catalogue grows. A ``ToolRouter`` splits the
catalogue in two:

- a **summary pool**: one short line per tool, byte-stable across turns so it stays in the
  cacheable prefix of the prompt;
- **promoted schemas**: the full JSON Schema for only the top-k tools relevant to this turn.

Ranking is hybrid and dependency-free: BM25 over each tool's name and description, optionally
fused with dense embedding similarity by Reciprocal Rank Fusion. Embeddings are opt-in; with
none supplied the router is pure Python and allocates nothing per turn beyond the score lists.

Scope: this is a **context optimisation, not a security boundary**. A tool whose schema was
never promoted still executes if the model names it — ``infy.governance`` remains the only
authority on what is allowed to run. Withholding a schema makes a call less likely, not
impossible, and must not be relied on to prevent one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infy.models import ToolSchema
    from infy.tools import Tool

# Standard RRF smoothing constant. At k=60 the gap between rank 1 and rank 2 dominates the gap
# between rank 100 and 101, so consensus across rankers decides the head of the list.
RRF_K = 60

_BM25_K1 = 1.5
_BM25_B = 0.75
_WORD_RE = re.compile(r"[a-z0-9]+")

_POOL_HEADER = (
    "Available tools. Full parameter schemas are provided only for the tools relevant to the "
    "current request. If you need one that is listed here but whose schema you were not given, "
    "name it in your reply and the schema will be provided on the next turn.\n"
)


# ---------------------------------------------------------------------------
# Rank fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked index lists into one ranking by Reciprocal Rank Fusion.

    Each ranking contributes ``1 / (k + rank)`` to the items it ranks. Fusing by *rank* rather
    than by score is what makes it safe to combine rankers whose scores are not commensurable —
    BM25 is unbounded and positive, cosine similarity is bounded — without any normalisation
    step that would need tuning.

    Ties break on the lower index, so the output is fully deterministic for a given input.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


# ---------------------------------------------------------------------------
# BM25 over the tool catalogue
# ---------------------------------------------------------------------------


class _BM25:
    """BM25 over a small, fixed corpus (the tool catalogue), built once at construction.

    Term frequency saturation (``k1``) and length normalisation (``b``) keep a long tool
    description from outranking a short, exactly-matching one.

    Scoring walks an **inverted index**: a query term visits only the tools that actually contain
    it, instead of every tool in the catalogue. On a realistic catalogue most terms match a
    handful of tools, so this is the difference between O(tools x query terms) and O(matches).
    Per-document length norms are precomputed, leaving one multiply-divide per posting.
    """

    __slots__ = ("_count", "_idf", "_norms", "_postings")

    def __init__(self, documents: Sequence[Sequence[str]]) -> None:
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for index, document in enumerate(documents):
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            for term, freq in frequencies.items():
                postings.setdefault(term, []).append((index, freq))
            lengths.append(len(document))

        total = len(documents)
        self._count: int = total
        self._postings: dict[str, list[tuple[int, int]]] = postings
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (total - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in postings.items()
        }
        avgdl = (sum(lengths) / total) if total else 0.0
        self._norms: list[float] = [
            _BM25_K1 * (1.0 - _BM25_B + _BM25_B * length / avgdl) if avgdl > 0.0 else 0.0
            for length in lengths
        ]

    def scores(self, query_terms: Sequence[str]) -> list[float]:
        """Score every document against ``query_terms``, aligned to catalogue order."""
        out = [0.0] * self._count
        norms = self._norms
        for term in query_terms:
            entries = self._postings.get(term)
            if entries is None:
                continue
            weight = self._idf[term] * (_BM25_K1 + 1.0)
            for index, freq in entries:
                out[index] += weight * freq / (freq + norms[index])
        return out


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


@dataclass
class ToolRouter:
    """Select which tool schemas to send on a given turn.

    Usage::

        from infy import ToolRouter, create_agent

        router = ToolRouter(tools=ALL_TOOLS, top_k=5, always=["search"])
        agent = create_agent(model, ALL_TOOLS, tool_router=router)

    Args:
        tools: The full catalogue. Order is preserved in every output, which is what keeps the
            promoted list append-only and the prompt prefix stable across turns.
        top_k: How many schemas to promote per turn, on top of ``always``.
        always: Names of tools to promote on every turn regardless of the query.
        embeddings: Optional ``infy.embeddings.Embeddings``. When supplied, dense similarity is
            fused with BM25 via RRF. Tool vectors are embedded once, at construction.
        sticky: Keep previously promoted tools promoted for the rest of the run. This costs
            context but makes the tool list grow monotonically, so a provider that caches on a
            request prefix keeps hitting. Set ``False`` to re-select from scratch each turn.
        summary_chars: Budget for each tool's description in the summary pool.

    The catalogue is read **once**, at construction: the BM25 index, the name set and the
    rendered summary pool are all built in ``__post_init__`` so that per-turn selection touches
    nothing but the score arrays. Mutating ``tools`` after construction is not supported — build
    a new router instead.
    """

    tools: list[Tool]
    top_k: int = 5
    always: list[str] = field(default_factory=list)
    embeddings: Any = None  # infy.embeddings.Embeddings — Any to avoid an import cycle
    sticky: bool = True
    summary_chars: int = 96

    _bm25: _BM25 = field(init=False, repr=False)
    _names: set[str] = field(init=False, repr=False)
    _pool: str = field(init=False, repr=False)
    _pinned: set[str] = field(init=False, repr=False)
    _vectors: list[list[float]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._bm25 = _BM25([_tool_terms(t) for t in self.tools])
        self._names = {t.name for t in self.tools}
        self._pinned = {name for name in self.always if name in self._names}
        self._pool = _POOL_HEADER + "\n".join(
            _summary_line(t, self.summary_chars) for t in self.tools
        )
        if self.embeddings is not None and self.tools:
            self._vectors = self.embeddings.embed_documents(
                [f"{t.name}: {t.description}" for t in self.tools]
            )

    # --- public API -------------------------------------------------------

    def summary_pool(self) -> str:
        """The compact catalogue block to keep resident in the system prompt.

        Depends only on the catalogue, so it is rendered once and is byte-identical on every
        turn — which is what keeps it inside a provider's cacheable prefix.
        """
        return self._pool

    def select(self, query: str, *, promoted: Iterable[str] = ()) -> list[Tool]:
        """Return the tools whose full schema should be sent for this turn.

        The result is always in catalogue order, never ranked order: the ranking decides
        *membership*, and a stable order keeps the serialised tool list from churning between
        turns for reasons the model cannot see.
        """
        if len(self.tools) <= self.top_k:
            return list(self.tools)

        keep: set[str] = set(self._pinned)
        if self.sticky:
            keep.update(name for name in promoted if name in self._names)

        terms = _terms(query)
        if terms:
            for index in self._rank(terms, query)[: self.top_k]:
                keep.add(self.tools[index].name)
        elif not keep:
            # No query signal and nothing pinned: send a bounded prefix rather than everything.
            return list(self.tools[: self.top_k])

        return [t for t in self.tools if t.name in keep]

    def schemas(self, query: str, *, promoted: Iterable[str] = ()) -> list[ToolSchema]:
        """``select``, already converted to the wire schemas the model call expects."""
        return [t.to_schema() for t in self.select(query, promoted=promoted)]

    # --- internals --------------------------------------------------------

    def _rank(self, terms: Sequence[str], query: str) -> list[int]:
        """Indices of ``self.tools``, best match first."""
        lexical = self._bm25.scores(terms)
        dense = self._dense_scores(query)
        if dense is None:
            # Lexical alone: a zero score means no term overlap at all. Promoting those would
            # pad the context with schemas the turn has no reason to want.
            return [i for i in _order(lexical) if lexical[i] > 0.0]
        return [index for index, _ in reciprocal_rank_fusion([_order(lexical), _order(dense)])]

    def _dense_scores(self, query: str) -> list[float] | None:
        vectors = self._vectors
        if vectors is None:
            return None
        from infy.similarity import batch_cosine_similarity

        query_vector = self.embeddings.embed_query(query)
        scores = [0.0] * len(vectors)
        for index, score in batch_cosine_similarity(query_vector, vectors):
            scores[index] = score
        return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _tool_terms(tool: Tool) -> list[str]:
    """Searchable terms for a tool: its name (snake_case split) plus its description."""
    return _terms(tool.name.replace("_", " ")) + _terms(tool.description)


def _summary_line(tool: Tool, limit: int) -> str:
    description = " ".join(tool.description.split())
    if len(description) > limit:
        description = description[: max(0, limit - 3)].rstrip() + "..."
    return f"- {tool.name}: {description}" if description else f"- {tool.name}"


def _order(scores: Sequence[float]) -> list[int]:
    """Indices sorted by score descending, ties broken on the lower index."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))
