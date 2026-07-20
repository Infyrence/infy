"""Tests for durable, out-of-band human approval: a governed run suspends at the approval
chokepoint, persists itself, and resumes minutes or hours later once a human has decided."""

import pytest

from infy.agents import create_agent
from infy.governance import (
    AuditLog,
    CallbackApprover,
    DurableAgent,
    DurableApprover,
    Governance,
    InMemoryApprovalStore,
    Policy,
    fingerprint,
)
from infy.governance.approval import ApprovalRequest
from infy.messages import AIMessage, ToolCall, ToolMessage
from infy.tools import tool

# --------------------------------------------------------------------------- fakes


class ScriptModel:
    """Returns a scripted sequence of responses, then a final (no-tool) message."""

    model_name = "script"

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages, *, tools=None, **kw):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return AIMessage(content="done")


EXECUTED: list[str] = []


@tool(verb="READ")
def read_ledger(q: str) -> str:
    """Read the ledger."""
    EXECUTED.append(f"read:{q}")
    return "balance ok"


@tool(verb="WRITE", side_effect=True)
def pay(amount: int, to: str) -> str:
    """Release a payment (the gated, side-effecting action)."""
    EXECUTED.append(f"pay:{amount}:{to}")
    return f"paid {amount} to {to}"


def setup_function(_) -> None:
    EXECUTED.clear()


def _build(responses, store=None):
    store = store or InMemoryApprovalStore()
    audit = AuditLog()
    gov = Governance(
        # deny-by-default allowlist; `pay` needs approval; escalate_at=None so ONLY the
        # approval-list gates (risk does not auto-escalate), for a precise test.
        policy=Policy(allow=["read_ledger", "pay"], require_approval=["pay"]),
        approver=DurableApprover(store),
        audit=audit,
        escalate_at=None,
    )
    agent = DurableAgent(ScriptModel(responses), [read_ledger, pay], governance=gov, store=store)
    return agent, store, audit


def _pay_call(amount=100, to="acme", cid="c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[ToolCall(name="pay", args={"amount": amount, "to": to}, id=cid)]
    )


def _tool_msgs(result):
    return [m for m in result.messages if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- suspend


def test_suspends_on_gated_tool():
    agent, store, _ = _build([_pay_call()])
    result = agent.run("pay acme 100", run_id="r1")

    assert result.status == "suspended"
    assert result.run_id == "r1"
    assert len(result.pending_approvals) == 1
    assert result.pending_approvals[0].request.tool == "pay"
    assert result.pending_approvals[0].fingerprint  # non-empty
    # fail-closed: the side effect did NOT run and no tool result was produced
    assert EXECUTED == []
    assert _tool_msgs(result) == []
    # the run is persisted so it can be resumed
    assert store.load_state("r1") is not None


def test_reads_do_not_suspend():
    read_first = AIMessage(
        content="", tool_calls=[ToolCall(name="read_ledger", args={"q": "x"}, id="c1")]
    )
    agent, _, _ = _build([read_first])
    result = agent.run("check", run_id="r1")

    assert result.status == "completed"
    assert result.pending_approvals == []
    assert EXECUTED == ["read:x"]


# --------------------------------------------------------------------------- resume


def test_resume_approve_executes_and_completes():
    agent, store, audit = _build([_pay_call()])
    suspended = agent.run("pay", run_id="r1")
    fp = suspended.pending_approvals[0].fingerprint

    done = agent.resume("r1", {fp: True})

    assert done.status == "completed"
    assert EXECUTED == ["pay:100:acme"]
    assert any(m.content.startswith("paid") for m in _tool_msgs(done))
    assert any(e.decision == "approved" for e in audit.events)
    assert audit.verify()
    # completed runs clear their persisted state
    assert store.load_state("r1") is None


def test_resume_deny_blocks_and_records():
    agent, store, audit = _build([_pay_call()])
    suspended = agent.run("pay", run_id="r1")
    fp = suspended.pending_approvals[0].fingerprint

    done = agent.resume("r1", {fp: False})

    assert done.status == "completed"
    assert EXECUTED == []  # denied action never ran
    # the model saw a blocked tool result
    assert any(m.status == "error" for m in _tool_msgs(done))
    assert any(e.decision == "rejected" for e in audit.events)


def test_resume_unknown_run_raises():
    agent, _, _ = _build([_pay_call()])
    with pytest.raises(KeyError):
        agent.resume("never-suspended", {})


# --------------------------------------------------------------------------- binding / TOCTOU


def test_fingerprint_binds_tool_and_args():
    assert fingerprint("pay", {"amount": 100}) == fingerprint("pay", {"amount": 100})
    assert fingerprint("pay", {"amount": 100}) != fingerprint("pay", {"amount": 200})
    assert fingerprint("pay", {"amount": 100}) != fingerprint("wire", {"amount": 100})
    # arg order does not matter (canonicalized)
    assert fingerprint("pay", {"a": 1, "b": 2}) == fingerprint("pay", {"b": 2, "a": 1})


def test_wrong_fingerprint_does_not_approve():
    agent, store, _ = _build([_pay_call()])
    suspended = agent.run("pay", run_id="r1")
    real_fp = suspended.pending_approvals[0].fingerprint

    # A verdict recorded against a DIFFERENT fingerprint must not release this action.
    again = agent.resume("r1", {"deadbeef" + real_fp[8:]: True})
    assert again.status == "suspended"
    assert EXECUTED == []


# --------------------------------------------------------------------------- batch / multi


def test_pre_authorize_no_sibling_side_effect():
    # One batch: a read AND a gated pay. The pending approval must suspend the WHOLE batch
    # before the read's side effect runs.
    batch = AIMessage(
        content="",
        tool_calls=[
            ToolCall(name="read_ledger", args={"q": "x"}, id="c1"),
            ToolCall(name="pay", args={"amount": 100, "to": "acme"}, id="c2"),
        ],
    )
    agent, store, _ = _build([batch])
    suspended = agent.run("do both", run_id="r1")

    assert suspended.status == "suspended"
    assert EXECUTED == []  # the read did NOT run ahead of the pending approval

    fp = suspended.pending_approvals[0].fingerprint
    done = agent.resume("r1", {fp: True})
    assert done.status == "completed"
    assert EXECUTED == ["read:x", "pay:100:acme"]


def test_multi_step_two_approvals():
    agent, store, _ = _build([_pay_call(amount=1, cid="a"), _pay_call(amount=2, cid="b")])

    s1 = agent.run("two payments", run_id="r1")
    assert s1.status == "suspended"
    d1 = agent.resume("r1", {s1.pending_approvals[0].fingerprint: True})

    assert d1.status == "suspended"  # a second gated action appeared
    assert EXECUTED == ["pay:1:acme"]
    d2 = agent.resume("r1", {d1.pending_approvals[0].fingerprint: True})

    assert d2.status == "completed"
    assert EXECUTED == ["pay:1:acme", "pay:2:acme"]


# --------------------------------------------------------------------------- fail-closed / default


def test_durable_approver_outside_run_scope_raises():
    approver = DurableApprover(InMemoryApprovalStore())
    with pytest.raises(RuntimeError):
        approver.review(ApprovalRequest("pay", {"amount": 1}, "high", "x"))


def test_default_create_agent_still_synchronous():
    # The classic sync path is untouched: a CallbackApprover resolves inline, no suspend.
    gov = Governance(
        policy=Policy(allow=["pay"], require_approval=["pay"]),
        approver=CallbackApprover(lambda req: True),
        escalate_at=None,
    )
    agent = create_agent(ScriptModel([_pay_call()]), [pay], governance=gov, parallel_tools=False)
    result = agent("pay")

    assert result.status == "completed"
    assert result.pending_approvals == []
    assert EXECUTED == ["pay:100:acme"]
