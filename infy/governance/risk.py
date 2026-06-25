"""Risk engine — assign a risk tier to a tool call from the tool's static profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from infy.governance.types import RiskTier

if TYPE_CHECKING:
    from infy.tools import Tool

_VERB_TIER: dict[str, RiskTier] = {
    "READ": RiskTier.LOW,
    "API_GET": RiskTier.LOW,
    "NETWORK": RiskTier.MEDIUM,
    "EXTERNAL_API": RiskTier.MEDIUM,
    "WRITE": RiskTier.HIGH,
    "DB": RiskTier.HIGH,
    "FS": RiskTier.HIGH,
    "EXECUTE": RiskTier.HIGH,
    "SENSITIVE_DATA": RiskTier.HIGH,
    "FINANCIAL": RiskTier.CRITICAL,
}


@dataclass
class RiskEngine:
    """Resolve a tool call's risk tier.

    Order: explicit ``tool.risk_tier`` -> ``tool.verb`` mapping -> conservative default
    (an UNPROFILED side-effecting tool is HIGH; a pure/read tool is LOW). ``args`` is
    accepted now so per-argument escalation can be added without changing call sites.
    """

    def assess(self, tool: Tool, args: dict[str, Any]) -> RiskTier:
        if tool.risk_tier is not None:
            return RiskTier(tool.risk_tier)
        if tool.verb is not None and tool.verb.upper() in _VERB_TIER:
            return _VERB_TIER[tool.verb.upper()]
        if tool.side_effect:
            return RiskTier.HIGH
        return RiskTier.LOW
