"""Govern an OpenHands agent with infy through its SecurityAnalyzer hook.

OpenHands' Software Agent SDK evaluates every action through a ``SecurityAnalyzer`` that returns a
``SecurityRisk`` (LOW/MEDIUM/HIGH/UNKNOWN), gated by a ``ConfirmationPolicy``. This adapter
replaces OpenHands' LLM-guessed risk with infy's DETERMINISTIC
policy and risk engine, and writes every action to infy's tamper-evident, hash-chained audit trail,
which OpenHands does not provide.

Two layers, on purpose:

* ``assess_action`` is the SDK-free core: it runs one action through infy's policy, risk, and audit,
  and returns the OpenHands risk plus whether the policy permits it. It is fully unit-tested here.
* ``build_analyzer`` is the thin wrapper that plugs the core into a real OpenHands
  ``SecurityAnalyzerBase``. It imports ``openhands.sdk`` lazily (a heavy optional dependency), so
  importing this module never requires the SDK. Smoke-test ``build_analyzer`` on a host where
  ``pip install openhands-sdk`` succeeds; the mapping it delegates to is already covered by tests.

Note: the risk-to-confirmation gate is OpenHands' own (its ConfirmationPolicy). infy contributes the
deterministic classification and the cryptographic audit. A policy FORBID is surfaced as HIGH so the
confirmation policy stops it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from infy.governance import AuthRequest, Effect

if TYPE_CHECKING:
    from infy.governance import Governance

# infy risk tier -> OpenHands SecurityRisk value. CRITICAL folds into HIGH (OpenHands' top tier).
_TIER_TO_RISK: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "critical": "HIGH",
}


@dataclass
class _ToolMeta:
    """The tool metadata infy's risk engine reads (verb/risk_tier/side_effect) plus the name."""

    name: str
    verb: str | None = None
    risk_tier: str | None = None
    side_effect: bool = True


def assess_action(
    governance: Governance,
    *,
    tool_name: str,
    args: dict[str, Any],
    verb: str | None = None,
    risk_tier: str | None = None,
    side_effect: bool = True,
) -> tuple[str, bool]:
    """Classify one OpenHands action with infy governance.

    Runs the action through infy's deterministic policy and risk engine, writes the decision to the
    tamper-evident audit chain, and returns ``(openhands_risk, allowed)``. Uses policy and risk
    directly (not the human approver), because OpenHands owns the confirmation gate. A policy denial
    is surfaced as ``"HIGH"`` so OpenHands' ConfirmationPolicy stops it, and is audited as a deny.
    """
    meta = _ToolMeta(name=tool_name, verb=verb, risk_tier=risk_tier, side_effect=side_effect)
    tier = governance.risk.assess(cast(Any, meta), dict(args))
    request = AuthRequest(governance.principal, "invoke_tool", tool_name, dict(args), tier)
    decision = governance.policy.evaluate(request)
    allowed = decision.effect is not Effect.FORBID
    governance.audit.record(
        actor=governance.principal,
        action="invoke_tool",
        resource=tool_name,
        decision="allow" if allowed else "deny",
        risk_tier=tier.value,
        reasons=decision.reasons,
        args=dict(args),
    )
    risk = _TIER_TO_RISK.get(tier.value, "HIGH") if allowed else "HIGH"
    return risk, allowed


def _default_extract(action: Any) -> tuple[str, dict[str, Any]]:
    """Best-effort extraction of ``(tool_name, args)`` from an OpenHands action event.

    OpenHands action shapes vary by version, so this tries the common fields and falls back to the
    class name. Pass your own ``extract`` to ``build_analyzer`` if your SDK version differs.
    """
    name = (
        getattr(action, "tool_name", None)
        or getattr(action, "action", None)
        or type(action).__name__
    )
    args = getattr(action, "arguments", None)
    if args is None:
        args = getattr(action, "args", None)
    if not isinstance(args, dict):
        args = {"value": str(args)} if args is not None else {}
    return str(name), args


def build_analyzer(
    governance: Governance,
    *,
    risk: dict[str, dict[str, Any]] | None = None,
    extract: Callable[[Any], tuple[str, dict[str, Any]]] | None = None,
) -> Any:
    """Build an OpenHands ``SecurityAnalyzer`` backed by infy governance.

    Register the returned analyzer with an OpenHands conversation or agent as its security analyzer,
    alongside a ConfirmationPolicy (for example ``ConfirmRisky``). ``risk`` maps a tool name to
    ``{"verb", "risk_tier", "side_effect"}``; ``extract`` overrides how ``(tool_name, args)`` are
    read from an action. Imports ``openhands.sdk`` lazily, so it is only needed when you call this.
    """
    from openhands.sdk.security.analyzer import SecurityAnalyzerBase
    from openhands.sdk.security.risk import SecurityRisk

    profiles = risk or {}
    read = extract or _default_extract

    class InfyrenceSecurityAnalyzer(SecurityAnalyzerBase):  # type: ignore[misc]  # untyped SDK base
        """A SecurityAnalyzer that classifies and audits every action with infy governance."""

        def security_risk(self, action: Any) -> Any:
            name, args = read(action)
            meta = profiles.get(name, {})
            risk_str, _allowed = assess_action(
                governance,
                tool_name=name,
                args=args,
                verb=meta.get("verb"),
                risk_tier=meta.get("risk_tier"),
                side_effect=meta.get("side_effect", True),
            )
            return SecurityRisk(risk_str)

    return InfyrenceSecurityAnalyzer()
