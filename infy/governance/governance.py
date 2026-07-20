"""Governance facade — the in-process control wired into the agent loop.

Enforcement runs IN-PROCESS at the existing chokepoints (a plain function call, microseconds),
never an out-of-band service on the hot path. Every decision path FAILS CLOSED: any error in
risk scoring, policy evaluation, or approval results in a deny, and is still audited. Attach it
via ``create_agent(..., governance=Governance(...))``; when absent the loop is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from infy.governance.approval import ApprovalRequest, ApprovalRequired, Approver, DenyAll
from infy.governance.audit import AuditLog
from infy.governance.engine import PolicyEngine
from infy.governance.risk import RiskEngine
from infy.governance.types import (
    AuthRequest,
    Effect,
    Obligation,
    RiskTier,
    ToolDecision,
    tier_at_least,
)

if TYPE_CHECKING:
    from infy.messages import AIMessage, Message, ToolMessage
    from infy.tools import Tool


@dataclass
class Governance:
    """Holds the policy engine, risk engine, approval handler, and audit log, and exposes the
    four hook points the agent loop calls (before/after model, before/after tool).

    ``escalate_at`` makes RISK gate the decision, not just annotate it: any permitted action at
    or above this tier is sent to the approver (default HIGH), so an unprofiled high-risk or
    side-effecting tool is denied-by-default unless a human approves. Set to None to disable.
    """

    policy: PolicyEngine
    risk: RiskEngine = field(default_factory=RiskEngine)
    audit: AuditLog = field(default_factory=AuditLog)
    approver: Approver = field(default_factory=DenyAll)
    principal: str = "agent"
    escalate_at: RiskTier | None = RiskTier.HIGH

    # -- tool chokepoint (the critical security boundary) ------------------

    def before_tool(self, tool: Tool, args: dict[str, Any]) -> ToolDecision:
        try:
            return self._authorize(tool, args)
        except (
            ApprovalRequired
        ):  # not an error: the run must SUSPEND for a human. Let it propagate.
            raise
        except Exception as exc:  # ANY governance error fails closed (deny), and is audited
            self._log(
                "invoke_tool",
                tool.name,
                "error",
                "",
                (f"governance-error:{type(exc).__name__}",),
                args,
            )
            return ToolDecision(False, f"Blocked: governance error [{type(exc).__name__}]")

    def _authorize(self, tool: Tool, args: dict[str, Any]) -> ToolDecision:
        tier = self.risk.assess(tool, args)
        request = AuthRequest(self.principal, "invoke_tool", tool.name, args, tier)
        decision = self.policy.evaluate(request)

        if decision.effect is Effect.FORBID:
            self._log("invoke_tool", tool.name, "deny", tier.value, decision.reasons, args)
            joined = ", ".join(decision.reasons)
            return ToolDecision(False, f"Blocked by policy: {tool.name} [{joined}]")

        needs_approval = Obligation.REQUIRE_APPROVAL in decision.obligations or (
            self.escalate_at is not None and tier_at_least(tier, self.escalate_at)
        )
        if needs_approval:
            try:
                approved = self.approver.review(
                    ApprovalRequest(tool.name, args, tier.value, "policy/risk requires approval")
                )
            except ApprovalRequired:  # a durable approver defers to a human: suspend, do not reject
                raise
            except Exception as exc:  # an approver that errors is treated as a rejection
                self._log(
                    "invoke_tool",
                    tool.name,
                    "rejected",
                    tier.value,
                    (f"approver-error:{type(exc).__name__}",),
                    args,
                )
                return ToolDecision(False, f"Approval failed for {tool.name}")
            verdict = "approved" if approved else "rejected"
            self._log("invoke_tool", tool.name, verdict, tier.value, decision.reasons, args)
            if not approved:
                return ToolDecision(False, f"Approval rejected for {tool.name}")
            return ToolDecision(True)

        self._log("invoke_tool", tool.name, "allow", tier.value, decision.reasons, args)
        return ToolDecision(True)

    def after_tool(self, tool: Tool, result: ToolMessage) -> ToolMessage:
        self._log("tool_result", tool.name, getattr(result, "status", "ok") or "ok")
        return result

    def on_unknown_tool(self, name: str) -> None:
        """Record an attempt to invoke an unregistered tool (probing is observable)."""
        self._log("invoke_tool", name, "deny", "", ("unknown-tool",))

    # -- model chokepoint (audit; semantic firewall deferred) --------------

    def before_model(self, messages: list[Message]) -> None:
        self._log("model_call", "model", "logged", reasons=(f"messages={len(messages)}",))

    def after_model(self, response: AIMessage) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            tok = f"in={getattr(usage, 'input_tokens', 0)},out={getattr(usage, 'output_tokens', 0)}"
        else:
            tok = "tokens=?"
        self._log("model_result", "model", "logged", reasons=(tok,))

    # -- internals ---------------------------------------------------------

    def _log(
        self,
        action: str,
        resource: str,
        decision: str,
        risk_tier: str = "",
        reasons: tuple[str, ...] = (),
        args: dict[str, Any] | None = None,
    ) -> None:
        self.audit.record(
            actor=self.principal,
            action=action,
            resource=resource,
            decision=decision,
            risk_tier=risk_tier,
            reasons=reasons,
            args=args,
        )
