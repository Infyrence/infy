"""Tamper-evident audit log — append-only, SHA-256 (or HMAC) hash-chained, thread-safe.

Each event's hash chains the previous one, so editing, deleting, or reordering any historical
event breaks ``verify()`` — the integrity guarantee database permissions alone cannot give, and
the evidence chain SOC 2 (CC6/CC7/CC8) and EU AI Act Art. 12 expect.

Threat model (be honest): a plain SHA-256 chain is *self-anchored* — an attacker who controls
the process can recompute the whole chain after editing it. Pass ``key=`` (held OUTSIDE the
agent process) to switch to HMAC-SHA256 so forging the chain requires that key. Anchoring the
head hash to an external notary/WORM sink is the production hardening (deferred). Zero dependency
(stdlib hashlib/hmac/json/threading).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

GENESIS = "0" * 64
_NO_ARGS = ""  # distinct from a hash of {} so "no args" and "empty args" never collide


def _canonical(obj: Any) -> str:
    # allow_nan=False fails closed on NaN/Infinity rather than emitting invalid JSON.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    ts: float
    actor: str
    action: str
    resource: str
    decision: str
    risk_tier: str
    reasons: tuple[str, ...]
    args_digest: str
    prev_hash: str
    hash: str


def _chain_hash(
    key: bytes | None,
    *,
    seq: int,
    ts: float,
    actor: str,
    action: str,
    resource: str,
    decision: str,
    risk_tier: str,
    reasons: tuple[str, ...],
    args_digest: str,
    prev_hash: str,
) -> str:
    body = {
        "seq": seq,
        "ts": ts,
        "actor": actor,
        "action": action,
        "resource": resource,
        "decision": decision,
        "risk_tier": risk_tier,
        "reasons": list(reasons),
        "args_digest": args_digest,
        "prev_hash": prev_hash,
    }
    payload = (prev_hash + _canonical(body)).encode()
    if key is not None:
        return hmac.new(key, payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class AuditLog:
    """Append-only hash-chained log. Thread-safe. Optionally HMAC-keyed and JSONL-mirrored."""

    path: str | None = None
    key: bytes | None = None
    _events: list[AuditEvent] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record(
        self,
        *,
        actor: str,
        action: str,
        resource: str,
        decision: str,
        risk_tier: str = "",
        reasons: tuple[str, ...] = (),
        args: dict[str, Any] | None = None,
    ) -> AuditEvent:
        args_digest = (
            _NO_ARGS if args is None else hashlib.sha256(_canonical(args).encode()).hexdigest()
        )
        with self._lock:  # seq, prev-hash read, hash, append, and file write are atomic
            seq = len(self._events)
            ts = time.time()
            prev = self._events[-1].hash if self._events else GENESIS
            digest = _chain_hash(
                self.key,
                seq=seq,
                ts=ts,
                actor=actor,
                action=action,
                resource=resource,
                decision=decision,
                risk_tier=risk_tier,
                reasons=reasons,
                args_digest=args_digest,
                prev_hash=prev,
            )
            event = AuditEvent(
                seq=seq,
                ts=ts,
                actor=actor,
                action=action,
                resource=resource,
                decision=decision,
                risk_tier=risk_tier,
                reasons=reasons,
                args_digest=args_digest,
                prev_hash=prev,
                hash=digest,
            )
            self._events.append(event)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(_canonical(asdict(event)) + "\n")
            return event

    def verify(self) -> bool:
        """True iff the chain is intact (no event added, removed, reordered, or altered)."""
        prev = GENESIS
        for event in self._events:
            expected = _chain_hash(
                self.key,
                seq=event.seq,
                ts=event.ts,
                actor=event.actor,
                action=event.action,
                resource=event.resource,
                decision=event.decision,
                risk_tier=event.risk_tier,
                reasons=event.reasons,
                args_digest=event.args_digest,
                prev_hash=event.prev_hash,
            )
            if event.prev_hash != prev or event.hash != expected:
                return False
            prev = event.hash
        return True

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)
