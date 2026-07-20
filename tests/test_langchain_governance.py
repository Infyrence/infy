"""Governing a real LangChain agent's tools with infy: a denied or unapproved action never runs,
and every decision lands in the tamper-evident audit chain. Skipped without langchain-core."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool as lc_tool  # noqa: E402

from infy.governance import AuditLog, CallbackApprover, Governance, Policy  # noqa: E402
from infy.integrations.langchain import govern  # noqa: E402

CALLS: list[str] = []


@lc_tool
def run_shell(command: str) -> str:
    """Run a shell command."""
    CALLS.append(f"shell:{command}")
    return f"ran {command}"


@lc_tool
def read_file(path: str) -> str:
    """Read a file."""
    CALLS.append(f"read:{path}")
    return f"contents of {path}"


@lc_tool
def delete_all(confirm: bool) -> str:
    """Delete everything, destructive and irreversible."""
    CALLS.append("delete_all")
    return "deleted"


RISK = {
    "run_shell": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
    "read_file": {"verb": "READ", "risk_tier": "low", "side_effect": False},
    "delete_all": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
}


def setup_function(_) -> None:
    CALLS.clear()


def _governed(approve: bool):
    audit = AuditLog()
    gov = Governance(
        policy=Policy(
            allow=["read_file", "run_shell"],
            deny=["delete_all"],
            require_approval=["run_shell"],
        ),
        approver=CallbackApprover(lambda req: approve),
        audit=audit,
        escalate_at=None,
    )
    tools = {t.name: t for t in govern([run_shell, read_file, delete_all], gov, risk=RISK)}
    return tools, audit


def _decisions(audit):
    return [(e.resource, e.decision) for e in audit.events if e.action == "invoke_tool"]


def test_allowed_tool_runs_and_is_audited():
    tools, audit = _governed(approve=True)
    out = tools["read_file"].invoke({"path": "config.yaml"})
    assert out == "contents of config.yaml"
    assert CALLS == ["read:config.yaml"]
    assert ("read_file", "allow") in _decisions(audit)


def test_denied_tool_is_blocked_and_never_runs():
    tools, audit = _governed(approve=True)
    out = tools["delete_all"].invoke({"confirm": True})
    assert "governance" in out.lower() and "block" in out.lower()
    assert CALLS == []  # the destructive tool never executed
    assert ("delete_all", "deny") in _decisions(audit)


def test_approval_rejected_blocks_the_action():
    tools, audit = _governed(approve=False)  # the human says no
    out = tools["run_shell"].invoke({"command": "rm -rf /"})
    assert "governance" in out.lower()
    assert CALLS == []
    assert ("run_shell", "rejected") in _decisions(audit)


def test_approval_granted_lets_the_action_run():
    tools, audit = _governed(approve=True)  # the human says yes
    out = tools["run_shell"].invoke({"command": "ls -la"})
    assert out == "ran ls -la"
    assert CALLS == ["shell:ls -la"]
    assert ("run_shell", "approved") in _decisions(audit)


def test_audit_chain_is_tamper_evident():
    tools, audit = _governed(approve=True)
    tools["read_file"].invoke({"path": "a.txt"})
    tools["delete_all"].invoke({"confirm": True})
    assert audit.verify()
    assert len(_decisions(audit)) == 2


def test_governed_tools_stay_drop_in_compatible():
    tools, _ = _governed(approve=True)
    t = tools["run_shell"]
    assert t.name == "run_shell"
    assert t.args_schema is not None  # the LangChain schema is preserved for the agent
    assert callable(t.invoke)
