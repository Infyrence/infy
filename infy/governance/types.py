"""Core governance types: risk tiers, policy effects, obligations, decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_TIER_RANK = {RiskTier.LOW: 0, RiskTier.MEDIUM: 1, RiskTier.HIGH: 2, RiskTier.CRITICAL: 3}


def tier_at_least(tier: RiskTier, threshold: RiskTier) -> bool:
    """True if ``tier`` is at or above ``threshold`` in severity."""
    return _TIER_RANK[tier] >= _TIER_RANK[threshold]


class Effect(str, Enum):
    PERMIT = "permit"
    FORBID = "forbid"


class Obligation(str, Enum):
    REQUIRE_APPROVAL = "require_approval"
    # REDACT and other obligations are intentionally not shipped until enforced
    # (a declared-but-unenforced control is worse than no control).


@dataclass(frozen=True)
class AuthRequest:
    """One authorization request evaluated at a chokepoint."""

    principal: str
    action: str  # "invoke_tool" | "model_call"
    tool: str
    args: dict[str, Any]
    risk_tier: RiskTier
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """A policy decision: permit/forbid plus any obligations and reasons."""

    effect: Effect
    obligations: tuple[Obligation, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def permitted(self) -> bool:
        return self.effect is Effect.PERMIT


@dataclass(frozen=True)
class ToolDecision:
    """The agent-loop-facing outcome of governing a single tool call."""

    allowed: bool
    message: str = ""  # error returned to the model when not allowed
