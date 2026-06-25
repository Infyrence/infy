"""Tests for the in-process governance layer."""

import dataclasses

from infy.agents import create_agent, create_async_agent
from infy.governance import (
    AuditLog,
    AuthRequest,
    AutoApprove,
    CallbackApprover,
    Effect,
    Governance,
    Policy,
    PythonPolicyEngine,
    RiskEngine,
    RiskTier,
    Rule,
)
from infy.governance.types import Obligation
from infy.messages import AIMessage, ToolCall, ToolMessage
from infy.tools import Tool, tool

# --------------------------------------------------------------------------- fakes


class FakeModel:
    model_name = "fake"

    def __init__(self, tool_name: str, tool_args: dict) -> None:
        self.calls = 0
        self._tn = tool_name
        self._ta = tool_args

    def generate(self, messages, *, tools=None, **kw):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="", tool_calls=[ToolCall(name=self._tn, args=self._ta, id="c1")]
            )
        return AIMessage(content="final answer")

    async def agenerate(self, messages, *, tools=None, **kw):
        return self.generate(messages, tools=tools, **kw)


@tool(verb="WRITE", side_effect=True)
def write_db(value: str) -> str:
    """Write a value to the database."""
    return f"wrote:{value}"


@tool(verb="READ")
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"results:{query}"


def _tool_messages(result):
    return [m for m in result.messages if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- engine


class TestPolicyEngine:
    def _req(self, name="search", tier=RiskTier.LOW):
        return AuthRequest("agent", "invoke_tool", name, {}, tier)

    def test_default_deny(self):
        assert PythonPolicyEngine([]).evaluate(self._req()).effect is Effect.FORBID

    def test_permit_only_matching(self):
        eng = PythonPolicyEngine([Rule(Effect.PERMIT, tools=("search",))])
        assert eng.evaluate(self._req("search")).effect is Effect.PERMIT
        assert eng.evaluate(self._req("other")).effect is Effect.FORBID

    def test_forbid_overrides_permit(self):
        eng = PythonPolicyEngine(
            [Rule(Effect.PERMIT, name="p"), Rule(Effect.FORBID, tools=("shell",), name="f")]
        )
        assert eng.evaluate(self._req("shell")).effect is Effect.FORBID
        assert eng.evaluate(self._req("search")).effect is Effect.PERMIT

    def test_fail_closed_on_condition_error(self):
        def boom(req):
            raise RuntimeError("boom")

        eng = PythonPolicyEngine([Rule(Effect.PERMIT, condition=boom)])
        decision = eng.evaluate(self._req())
        assert decision.effect is Effect.FORBID
        assert any("fail-closed" in r for r in decision.reasons)

    def test_policy_require_approval_by_name(self):
        decision = Policy(require_approval=["write_db"]).evaluate(
            self._req("write_db", RiskTier.HIGH)
        )
        assert decision.effect is Effect.PERMIT
        assert Obligation.REQUIRE_APPROVAL in decision.obligations

    def test_policy_approve_when_predicate(self):
        pol = Policy(approve_when=lambda r: r.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL))
        assert (
            Obligation.REQUIRE_APPROVAL in pol.evaluate(self._req("x", RiskTier.HIGH)).obligations
        )
        assert (
            Obligation.REQUIRE_APPROVAL
            not in pol.evaluate(self._req("x", RiskTier.LOW)).obligations
        )


# --------------------------------------------------------------------------- risk


class TestRiskEngine:
    def test_explicit_tier_wins(self):
        @tool(risk_tier="critical", verb="READ")
        def t() -> str:
            """x."""
            return ""

        assert RiskEngine().assess(t, {}) is RiskTier.CRITICAL

    def test_verb_mapping(self):
        assert RiskEngine().assess(write_db, {}) is RiskTier.HIGH
        assert RiskEngine().assess(search, {}) is RiskTier.LOW

    def test_unprofiled_side_effect_is_high(self):
        @tool(side_effect=True)
        def t() -> str:
            """x."""
            return ""

        assert RiskEngine().assess(t, {}) is RiskTier.HIGH

    def test_unprofiled_pure_is_low(self):
        @tool
        def t() -> str:
            """x."""
            return ""

        assert RiskEngine().assess(t, {}) is RiskTier.LOW


# --------------------------------------------------------------------------- audit


class TestAudit:
    def test_chain_verifies(self):
        log = AuditLog()
        for i in range(3):
            log.record(actor="a", action="invoke_tool", resource=f"t{i}", decision="allow")
        assert len(log) == 3
        assert log.verify() is True

    def test_tamper_is_detected(self):
        log = AuditLog()
        log.record(actor="a", action="invoke_tool", resource="t0", decision="allow")
        log.record(actor="a", action="invoke_tool", resource="t1", decision="deny")
        log._events[0] = dataclasses.replace(log._events[0], decision="deny")  # flip a verdict
        assert log.verify() is False

    def test_empty_verifies(self):
        assert AuditLog().verify() is True


# --------------------------------------------------------------------------- facade


class TestGovernanceFacade:
    def test_deny_blocks(self):
        decision = Governance(policy=Policy(deny=["write_db"])).before_tool(
            write_db, {"value": "x"}
        )
        assert decision.allowed is False
        assert "Blocked by policy" in decision.message

    def test_allow_passes(self):
        assert (
            Governance(policy=Policy(allow=["search"])).before_tool(search, {"query": "x"}).allowed
        )

    def test_require_approval_approved(self):
        gov = Governance(policy=Policy(require_approval=["write_db"]), approver=AutoApprove())
        assert gov.before_tool(write_db, {"value": "x"}).allowed is True

    def test_require_approval_default_denyall_rejects(self):
        gov = Governance(policy=Policy(require_approval=["write_db"]))  # default approver = DenyAll
        decision = gov.before_tool(write_db, {"value": "x"})
        assert decision.allowed is False
        assert "Approval rejected" in decision.message

    def test_callback_approver_receives_request(self):
        seen: list = []
        gov = Governance(
            policy=Policy(require_approval=["write_db"]),
            approver=CallbackApprover(lambda req: bool(seen.append(req)) or True),
        )
        assert gov.before_tool(write_db, {"value": "x"}).allowed is True
        assert seen[0].tool == "write_db"
        assert seen[0].risk_tier == "high"

    def test_audit_records_and_verifies(self):
        gov = Governance(policy=Policy(deny=["write_db"]))
        gov.before_tool(write_db, {"value": "x"})
        assert len(gov.audit) == 1
        assert gov.audit.events[0].decision == "deny"
        assert gov.audit.verify() is True


# --------------------------------------------------------------------------- agent loop


class TestAgentIntegration:
    def test_noop_when_governance_absent(self):
        result = create_agent(FakeModel("search", {"query": "x"}), tools=[search])("hi")
        assert _tool_messages(result)[0].content == "results:x"

    def test_deny_blocks_and_agent_recovers(self):
        gov = Governance(policy=Policy(deny=["write_db"]))
        agent = create_agent(
            FakeModel("write_db", {"value": "x"}), tools=[write_db], governance=gov
        )
        result = agent("hi")
        tm = _tool_messages(result)[0]
        assert tm.status == "error"
        assert "Blocked by policy" in tm.content
        assert result.response.text == "final answer"  # the loop kept going
        assert gov.audit.verify() is True

    def test_approval_gate_executes_on_approve(self):
        gov = Governance(policy=Policy(require_approval=["write_db"]), approver=AutoApprove())
        agent = create_agent(
            FakeModel("write_db", {"value": "x"}), tools=[write_db], governance=gov
        )
        result = agent("hi")
        assert _tool_messages(result)[0].content == "wrote:x"

    async def test_async_deny_path(self):
        gov = Governance(policy=Policy(deny=["write_db"]))
        agent = create_async_agent(
            FakeModel("write_db", {"value": "x"}), tools=[write_db], governance=gov
        )
        result = await agent("hi")
        tm = _tool_messages(result)[0]
        assert tm.status == "error"
        assert "Blocked by policy" in tm.content


# ----------------------------------------------------- security hardening (from review)


class TestGovernanceHardening:
    def test_governance_error_fails_closed(self):
        # A malformed profile makes RiskEngine raise; before_tool must DENY, not crash.
        bad = Tool(name="bad", description="", func=lambda: None, risk_tier="bogus")
        gov = Governance(policy=Policy())
        decision = gov.before_tool(bad, {})
        assert decision.allowed is False
        assert "governance error" in decision.message
        assert gov.audit.events[-1].decision == "error"

    def test_approver_exception_is_rejection(self):
        def boom(req):
            raise RuntimeError("approver backend down")

        gov = Governance(
            policy=Policy(require_approval=["write_db"]), approver=CallbackApprover(boom)
        )
        decision = gov.before_tool(write_db, {"value": "x"})
        assert decision.allowed is False
        assert gov.audit.events[-1].decision == "rejected"

    def test_callback_rejects(self):
        gov = Governance(
            policy=Policy(require_approval=["write_db"]),
            approver=CallbackApprover(lambda r: False),
        )
        decision = gov.before_tool(write_db, {"value": "x"})
        assert decision.allowed is False
        assert "Approval rejected" in decision.message

    def test_risk_escalation_denies_unprofiled_high_risk(self):
        # Unprofiled side-effecting -> HIGH -> escalated to approval -> default DenyAll -> deny.
        danger = Tool(name="danger", description="", func=lambda: "done", side_effect=True)
        assert Governance(policy=Policy()).before_tool(danger, {}).allowed is False

    def test_escalate_at_none_disables_risk_gate(self):
        danger = Tool(name="danger", description="", func=lambda: "done", side_effect=True)
        assert Governance(policy=Policy(), escalate_at=None).before_tool(danger, {}).allowed is True

    def test_allow_plus_require_approval_does_not_widen(self):
        # write_db is require_approval but NOT allowlisted -> must stay FORBID.
        pol = Policy(allow=["search"], require_approval=["write_db"])
        decision = pol.evaluate(AuthRequest("agent", "invoke_tool", "write_db", {}, RiskTier.HIGH))
        assert decision.effect is Effect.FORBID

    def test_unknown_tool_is_audited(self):
        gov = Governance(policy=Policy(allow=["search"]))
        create_agent(FakeModel("ghost", {}), tools=[search], governance=gov)("hi")
        assert any(e.resource == "ghost" and "unknown-tool" in e.reasons for e in gov.audit.events)

    def test_audit_is_thread_safe(self):
        import threading

        log = AuditLog()

        def worker():
            for _ in range(500):
                log.record(actor="a", action="invoke_tool", resource="t", decision="allow")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(log) == 8 * 500
        assert log.verify() is True

    def test_hmac_keyed_chain_detects_tampering(self):
        log = AuditLog(key=b"operator-secret")
        log.record(actor="a", action="invoke_tool", resource="t", decision="allow")
        log.record(actor="a", action="invoke_tool", resource="t2", decision="deny")
        assert log.verify() is True
        log._events[0] = dataclasses.replace(log._events[0], decision="deny")
        assert log.verify() is False
