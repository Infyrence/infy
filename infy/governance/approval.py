"""Approval gate — the human-in-the-loop decision for actions a policy escalates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from infy.messages import Message


@dataclass(frozen=True)
class ApprovalRequest:
    tool: str
    args: dict[str, Any]
    risk_tier: str
    reason: str


@runtime_checkable
class Approver(Protocol):
    def review(self, request: ApprovalRequest) -> bool: ...


@dataclass
class CallbackApprover:
    """Delegate the decision to a supplied callable (console prompt, Slack, queue, ...)."""

    callback: Callable[[ApprovalRequest], bool]

    def review(self, request: ApprovalRequest) -> bool:
        return bool(self.callback(request))


class AutoApprove:
    """Approve everything. Development and testing only — never production."""

    def review(self, request: ApprovalRequest) -> bool:
        return True


class DenyAll:
    """Reject everything. The fail-closed default when no approver is configured."""

    def review(self, request: ApprovalRequest) -> bool:
        return False


# ── durable, out-of-band approval ─────────────────────────────────────────────
# The synchronous approvers above must answer immediately. A real human sign-off can take
# minutes or hours, which no thread can wait on. These pieces let a run SUSPEND at the approval
# point, persist itself, and RESUME later once a human has decided — without blocking anything.


def fingerprint(tool: str, args: dict[str, Any]) -> str:
    """A stable id binding an approval to the EXACT action (tool + args). A recorded verdict only
    applies to a call with the same fingerprint, so it cannot be replayed onto a different action
    (defeats a TOCTOU swap between approval and execution)."""
    canon = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{tool}\x00{canon}".encode()).hexdigest()


@dataclass(frozen=True)
class PendingApproval:
    """An approval a suspended run is waiting on. ``fingerprint`` is what a decision is keyed by."""

    request: ApprovalRequest
    fingerprint: str


class ApprovalRequired(Exception):  # noqa: N818 - a suspend/resume signal, not an error
    """Raised by a ``DurableApprover`` when no decision exists yet: the run must SUSPEND and wait
    for a human rather than block. Carries the request and its fingerprint so the caller can
    surface it to an approval inbox and, later, record the verdict against the exact action."""

    def __init__(self, request: ApprovalRequest, fingerprint: str) -> None:
        super().__init__(f"approval required for {request.tool}")
        self.request = request
        self.fingerprint = fingerprint


@runtime_checkable
class ApprovalStore(Protocol):
    """Durable state for out-of-band approval: recorded decisions, pending requests, and the
    suspended run's messages. Back it with anything — in-memory for dev, Postgres for prod."""

    def get_decision(self, run_id: str, fingerprint: str) -> bool | None: ...
    def put_decision(self, run_id: str, fingerprint: str, approved: bool) -> None: ...
    def record_pending(self, run_id: str, pending: PendingApproval) -> None: ...
    def save_state(self, run_id: str, messages: list[Message]) -> None: ...
    def load_state(self, run_id: str) -> list[Message] | None: ...
    def clear(self, run_id: str) -> None: ...


class InMemoryApprovalStore:
    """In-memory ``ApprovalStore`` for development and tests. Not durable across processes."""

    def __init__(self) -> None:
        self._decisions: dict[tuple[str, str], bool] = {}
        self._pending: dict[str, list[PendingApproval]] = {}
        self._state: dict[str, list[Message]] = {}

    def get_decision(self, run_id: str, fingerprint: str) -> bool | None:
        return self._decisions.get((run_id, fingerprint))

    def put_decision(self, run_id: str, fingerprint: str, approved: bool) -> None:
        self._decisions[(run_id, fingerprint)] = approved

    def record_pending(self, run_id: str, pending: PendingApproval) -> None:
        bucket = self._pending.setdefault(run_id, [])
        if all(p.fingerprint != pending.fingerprint for p in bucket):
            bucket.append(pending)

    def pending(self, run_id: str) -> list[PendingApproval]:
        return list(self._pending.get(run_id, []))

    def save_state(self, run_id: str, messages: list[Message]) -> None:
        self._state[run_id] = list(messages)

    def load_state(self, run_id: str) -> list[Message] | None:
        stored = self._state.get(run_id)
        return list(stored) if stored is not None else None

    def clear(self, run_id: str) -> None:
        self._state.pop(run_id, None)
        self._pending.pop(run_id, None)
        # Recorded decisions are kept as a small record of what was decided; harmless to retain.


_run_id_var: ContextVar[str | None] = ContextVar("infy_approval_run_id", default=None)


@contextmanager
def run_scope(run_id: str) -> Iterator[None]:
    """Bind the current run id so a ``DurableApprover`` (which only sees the ApprovalRequest) can
    key its decisions and pending records by run. A ``DurableAgent`` sets this around its loop."""
    token = _run_id_var.set(run_id)
    try:
        yield
    finally:
        _run_id_var.reset(token)


@dataclass
class DurableApprover:
    """An approver that never blocks. It looks up a prior human decision in an ``ApprovalStore``;
    if none exists it records the pending request and raises ``ApprovalRequired`` so the run
    suspends. Fail-closed: an action with no recorded decision never runs."""

    store: ApprovalStore

    def review(self, request: ApprovalRequest) -> bool:
        run_id = _run_id_var.get()
        if run_id is None:
            raise RuntimeError("DurableApprover used outside a DurableAgent run (no run scope)")
        fp = fingerprint(request.tool, request.args)
        decision = self.store.get_decision(run_id, fp)
        if decision is not None:
            return decision
        self.store.record_pending(run_id, PendingApproval(request, fp))
        raise ApprovalRequired(request, fp)
