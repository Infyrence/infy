"""Bi-temporal memory — facts that know when they were true and when we believed them.

A conversational agent that stores "the user lives in New York" and later learns "the user
moved to London" has three bad options if its memory is a pile of text: keep both and
contradict itself, overwrite and lose the fact that New York was ever true, or ask the model
to reconcile them and hope. All three are how long-running agents end up confidently wrong.

The fix is to stop storing strings and store facts on **two independent time axes**:

- **Valid time** — when the fact was true *in the world*. New York was genuinely true until
  the move; that does not stop being so.
- **Transaction time** — when this store *believed* it. If we recorded London a week late,
  valid time starts at the move and transaction time starts at the recording.

Two axes are what make the crucial distinction expressible, and it is a distinction a single
timeline simply cannot hold:

- :meth:`BiTemporalStore.retract` — *the world changed.* The old fact stays true for its
  interval. Someone asking "where did they live in March?" still gets New York.
- :meth:`BiTemporalStore.correct` — *we were wrong.* The old record was never true; it is
  closed on the transaction axis so it vanishes from current belief, but it stays in the
  history so "what did we think in March, and when did we find out we were wrong?" is
  answerable. That question is an audit question, and an agent with real authority gets asked
  it.

The result is a memory you can query as of any point on either axis, which is what makes
"why did the agent do that?" answerable after the fact rather than a matter of reconstruction.

Zero dependencies, in-memory, and deliberately a plain protocol seam: ``BiTemporalStore`` is
an interface a durable backend can implement without the calling code changing, in the same
way ``PolicyEngine`` and ``Approver`` are seams in ``infy.governance``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

# A time far enough out to mean "still open" while remaining a real, comparable number, so
# interval maths never has to special-case None.
FOREVER = float("inf")


@dataclass(frozen=True)
class Fact:
    """One belief, on both time axes.

    ``(subject, predicate)`` is the *logical identity*: asserting a new object for the same
    pair is what closes the previous one. The object is the value that changes.
    """

    subject: str
    predicate: str
    object: Any

    valid_from: float = 0.0
    valid_to: float = FOREVER  # when it stopped being true in the world
    recorded_at: float = 0.0
    superseded_at: float = FOREVER  # when we stopped believing we had recorded it correctly

    confidence: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, str]:
        return (self.subject, self.predicate)

    @property
    def believed(self) -> bool:
        """Still part of current belief (never corrected away)."""
        return self.superseded_at == FOREVER

    @property
    def current(self) -> bool:
        """Believed *and* still true in the world."""
        return self.believed and self.valid_to == FOREVER

    def holds_at(self, valid_time: float, transaction_time: float = FOREVER) -> bool:
        """Was this fact true at ``valid_time``, according to what we believed at
        ``transaction_time``? Intervals are half-open: ``[from, to)``.

        ``transaction_time=FOREVER`` means "latest belief". It cannot go through the
        half-open check — an open record has ``superseded_at == FOREVER`` and ``inf < inf``
        is false, which would make every fact invisible — so it is answered as a belief
        question directly.
        """
        if not (self.valid_from <= valid_time < self.valid_to):
            return False
        if transaction_time == FOREVER:
            return self.believed
        return self.recorded_at <= transaction_time < self.superseded_at

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object!r}"


@runtime_checkable
class BiTemporalStore(Protocol):
    """The seam. Implement this over a real database to outlive the process."""

    def add(self, fact: Fact) -> None: ...

    def all(self) -> list[Fact]: ...

    def replace_fact(self, old: Fact, new: Fact) -> None: ...


@dataclass
class InMemoryBiTemporalStore:
    """Append-mostly list of facts. Small by construction: this holds an agent's beliefs,
    not its transcript."""

    facts: list[Fact] = field(default_factory=list)

    def add(self, fact: Fact) -> None:
        self.facts.append(fact)

    def all(self) -> list[Fact]:
        return list(self.facts)

    def replace_fact(self, old: Fact, new: Fact) -> None:
        self.facts[self.facts.index(old)] = new


@dataclass
class BiTemporalMemory:
    """Facts with valid-time and transaction-time, and the operators that keep them honest.

    Usage::

        from infy.temporal import BiTemporalMemory

        mem = BiTemporalMemory()
        mem.assert_fact("user", "lives_in", "New York", valid_from=t0)
        mem.assert_fact("user", "lives_in", "London", valid_from=t1)   # the world changed

        mem.current()                      # -> London
        mem.as_of(valid_time=t0 + 1)       # -> New York, still true for its interval

        mem.correct("user", "employer", "Acme")   # we had recorded the wrong value
        mem.history("user", "employer")           # the error is still on the record

    ``now`` is injectable so runs are reproducible and tests do not sleep.
    """

    store: BiTemporalStore = field(default_factory=InMemoryBiTemporalStore)
    clock: Any = time.time  # () -> float

    # --- writing ----------------------------------------------------------

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        object: Any,
        *,
        valid_from: float | None = None,
        confidence: float = 1.0,
        source: str = "",
        **metadata: Any,
    ) -> Fact:
        """Record a new belief, closing any prior belief for the same identity.

        This is "the world moved on": the previous fact keeps its interval and stays true for
        it. Asserting the value it already holds is a no-op that returns the existing fact,
        so a model repeating itself does not shred the timeline into adjacent duplicates.
        """
        now = float(self.clock())
        start = now if valid_from is None else float(valid_from)

        existing = self._open(subject, predicate)
        if existing is not None:
            if existing.object == object:
                return existing
            # Half-open intervals: the old fact ends exactly where the new one begins, so no
            # instant is covered by both and none is left uncovered.
            self.store.replace_fact(existing, replace(existing, valid_to=start))

        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            valid_from=start,
            recorded_at=now,
            confidence=confidence,
            source=source,
            metadata=dict(metadata),
        )
        self.store.add(fact)
        return fact

    def retract(
        self, subject: str, predicate: str, *, valid_from: float | None = None
    ) -> Fact | None:
        """The world changed and there is no replacement value: close the valid interval.

        The fact remains historically true — ``as_of`` a time inside its interval still
        returns it. Use this for "they no longer work here", not for "we had it wrong".
        """
        existing = self._open(subject, predicate)
        if existing is None:
            return None
        end = float(self.clock()) if valid_from is None else float(valid_from)
        closed = replace(existing, valid_to=end)
        self.store.replace_fact(existing, closed)
        return closed

    def correct(
        self,
        subject: str,
        predicate: str,
        object: Any,
        *,
        source: str = "",
    ) -> Fact:
        """We recorded the wrong thing: supersede the record without rewriting history.

        The mistaken fact is closed on the *transaction* axis, so it leaves current belief
        entirely — unlike :meth:`retract`, it was never true, so it must not answer a
        valid-time query. It stays in :meth:`history`, because an agent that can quietly
        erase its own mistakes cannot be audited.
        """
        now = float(self.clock())
        existing = self._open(subject, predicate)
        valid_from = existing.valid_from if existing is not None else now
        if existing is not None:
            self.store.replace_fact(existing, replace(existing, superseded_at=now))
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            valid_from=valid_from,
            recorded_at=now,
            source=source,
            metadata={"corrects": str(existing) if existing else ""},
        )
        self.store.add(fact)
        return fact

    # --- reading ----------------------------------------------------------

    def current(self, subject: str | None = None) -> list[Fact]:
        """Everything believed now and true now."""
        return [
            f for f in self.store.all() if f.current and (subject is None or f.subject == subject)
        ]

    def get(self, subject: str, predicate: str) -> Any:
        """The current value for one identity, or ``None``."""
        fact = self._open(subject, predicate)
        return fact.object if fact is not None else None

    def as_of(
        self,
        valid_time: float | None = None,
        transaction_time: float | None = None,
        *,
        subject: str | None = None,
    ) -> list[Fact]:
        """Reconstruct belief at a point on either axis.

        ``valid_time`` asks what was true in the world then; ``transaction_time`` asks what
        this store thought at that moment. Together they answer the question that actually
        settles arguments: *what did we believe, at the time we acted?*
        """
        vt = float(self.clock()) if valid_time is None else float(valid_time)
        tt = FOREVER if transaction_time is None else float(transaction_time)
        return [
            f
            for f in self.store.all()
            if f.holds_at(vt, tt) and (subject is None or f.subject == subject)
        ]

    def history(self, subject: str, predicate: str) -> list[Fact]:
        """Every record for one identity, oldest first — including corrected-away mistakes."""
        return sorted(
            (f for f in self.store.all() if f.identity == (subject, predicate)),
            key=lambda f: (f.recorded_at, f.valid_from),
        )

    def render(self, subject: str | None = None) -> str:
        """Current beliefs as a compact block to prepend to a prompt.

        Deliberately flat text: this is the part the model reads, and structure it cannot
        parse reliably is structure that costs tokens for nothing.
        """
        facts = self.current(subject)
        if not facts:
            return ""
        lines = ["Known facts:"]
        lines.extend(f"  - {f}" for f in sorted(facts, key=lambda f: (f.subject, f.predicate)))
        return "\n".join(lines)

    # --- internals --------------------------------------------------------

    def _open(self, subject: str, predicate: str) -> Fact | None:
        """The one believed, still-valid fact for an identity, if any."""
        for fact in reversed(self.store.all()):
            if fact.identity == (subject, predicate) and fact.current:
                return fact
        return None
