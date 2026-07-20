"""Governing a real smolagents agent's tools with infy: a denied or unapproved action never runs,
and every decision lands in the tamper-evident audit chain. Skipped when smolagents is absent."""

import pytest

pytest.importorskip("smolagents")

from smolagents import tool as sa_tool  # noqa: E402

from infy.governance import AuditLog, CallbackApprover, Governance, Policy  # noqa: E402
from infy.integrations.smolagents import govern  # noqa: E402

# Suppress smolagents' benign "decorators other than @tool" serialization warning for these fakes.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CALLS: list[str] = []


@sa_tool
def run_shell(command: str) -> str:
    """Run a shell command.

    Args:
        command: the command to run
    """
    CALLS.append(f"shell:{command}")
    return f"ran {command}"


@sa_tool
def read_file(path: str) -> str:
    """Read a file.

    Args:
        path: the file path
    """
    CALLS.append(f"read:{path}")
    return f"contents of {path}"


@sa_tool
def delete_all(confirm: bool) -> str:
    """Delete everything, destructive and irreversible.

    Args:
        confirm: confirmation flag
    """
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


def _call(tool, **kwargs):
    # Invoke exactly as smolagents' execute_tool_call does.
    return tool(**kwargs, sanitize_inputs_outputs=True)


def _decisions(audit):
    return [(e.resource, e.decision) for e in audit.events if e.action == "invoke_tool"]


def test_allowed_tool_runs_and_is_audited():
    tools, audit = _governed(approve=True)
    out = _call(tools["read_file"], path="config.yaml")
    assert out == "contents of config.yaml"
    assert CALLS == ["read:config.yaml"]
    assert ("read_file", "allow") in _decisions(audit)


def test_denied_tool_is_blocked_and_never_runs():
    tools, audit = _governed(approve=True)
    out = _call(tools["delete_all"], confirm=True)
    assert "governance" in out.lower() and "block" in out.lower()
    assert CALLS == []  # the destructive tool never executed
    assert ("delete_all", "deny") in _decisions(audit)


def test_approval_rejected_blocks_the_action():
    tools, audit = _governed(approve=False)  # the human says no
    out = _call(tools["run_shell"], command="rm -rf /")
    assert "governance" in out.lower()
    assert CALLS == []  # nothing ran
    assert ("run_shell", "rejected") in _decisions(audit)


def test_approval_granted_lets_the_action_run():
    tools, audit = _governed(approve=True)  # the human says yes
    out = _call(tools["run_shell"], command="ls -la")
    assert out == "ran ls -la"
    assert CALLS == ["shell:ls -la"]
    assert ("run_shell", "approved") in _decisions(audit)


def test_audit_chain_is_tamper_evident():
    tools, audit = _governed(approve=True)
    _call(tools["read_file"], path="a.txt")
    _call(tools["delete_all"], confirm=True)
    assert audit.verify()
    # a full trail exists: one governed decision per attempted action
    assert len(_decisions(audit)) == 2


def test_governed_tools_stay_drop_in_compatible():
    # The wrapped tools keep the smolagents interface, so the agent treats them identically.
    tools, _ = _governed(approve=True)
    t = tools["run_shell"]
    assert t.name == "run_shell"
    assert "command" in t.inputs
    assert t.output_type == "string"
