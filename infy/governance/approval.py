"""Approval gate — the human-in-the-loop decision for actions a policy escalates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


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
