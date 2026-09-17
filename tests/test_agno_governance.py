"""Governing a real Agno agent's tools with infy: a denied or unapproved action never runs, and
every decision lands in the tamper-evident audit chain. Skipped when agno is absent.

The tests drive ``FunctionCall.execute`` / ``aexecute`` directly, which is the same path Agno's
model layer takes when it dispatches a tool call, so the hook is exercised exactly as in a run.
"""

import pytest

pytest.importorskip("agno")

from agno.tools.function import Function, FunctionCall  # noqa: E402

from infy.governance import AuditLog, CallbackApprover, Governance, Policy  # noqa: E402
from infy.governance.types import RiskTier  # noqa: E402
from infy.integrations.agno import agovern_hook, govern_hook  # noqa: E402

CALLS: list[str] = []


def run_shell(command: str) -> str:
    """Run a shell command."""
    CALLS.append(f"shell:{command}")
    return f"ran {command}"


def read_file(path: str) -> str:
    """Read a file."""
    CALLS.append(f"read:{path}")
    return f"contents of {path}"


def delete_all(confirm: bool) -> str:
    """Delete everything, destructive and irreversible."""
    CALLS.append("delete_all")
    return "deleted"


def unprofiled(payload: str) -> str:
    """A tool with no risk profile, so the conservative default applies."""
    CALLS.append(f"unprofiled:{payload}")
    return "done"


RISK = {
    "run_shell": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
    "read_file": {"verb": "READ", "risk_tier": "low", "side_effect": False},
    "delete_all": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
}


def setup_function(_) -> None:
    CALLS.clear()


def _governed(approve: bool, *, escalate_at=None, asynchronous: bool = False):
    audit = AuditLog()
    gov = Governance(
        policy=Policy(
            allow=["read_file", "run_shell", "unprofiled"],
            deny=["delete_all"],
            require_approval=["run_shell"],
        ),
        approver=CallbackApprover(lambda req: approve),
        audit=audit,
        escalate_at=escalate_at,
    )
    build = agovern_hook if asynchronous else govern_hook
    hook = build(gov, risk=RISK)
    tools = {}
    for fn in (run_shell, read_file, delete_all, unprofiled):
        f = Function.from_callable(fn)
        f.tool_hooks = [hook]
        tools[f.name] = f
    return tools, audit


def _call(tool, **kwargs):
    # Dispatch exactly as Agno's model layer does when it executes a tool call.
    return FunctionCall(function=tool, arguments=kwargs).execute().result


async def _acall(tool, **kwargs):
    return (await FunctionCall(function=tool, arguments=kwargs).aexecute()).result


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


def test_unprofiled_tool_escalates_by_default():
    # No entry in RISK, so the conservative default (high risk, side-effecting) applies and the
    # risk tier alone gates the call once escalate_at is on.
    tools, audit = _governed(approve=False, escalate_at=RiskTier.HIGH)
    out = _call(tools["unprofiled"], payload="x")
    assert "governance" in out.lower()
    assert CALLS == []  # an unprofiled side-effecting tool is not waved through
    assert ("unprofiled", "rejected") in _decisions(audit)


def test_audit_chain_is_tamper_evident():
    tools, audit = _governed(approve=True)
    _call(tools["read_file"], path="a.txt")
    _call(tools["delete_all"], confirm=True)
    assert audit.verify()
    # a full trail exists: one governed decision per attempted action
    assert len(_decisions(audit)) == 2


async def test_async_hook_governs_the_async_path():
    tools, audit = _governed(approve=True, asynchronous=True)
    out = await _acall(tools["read_file"], path="config.yaml")
    assert out == "contents of config.yaml"
    assert ("read_file", "allow") in _decisions(audit)


async def test_async_hook_blocks_a_denied_tool():
    tools, audit = _governed(approve=True, asynchronous=True)
    out = await _acall(tools["delete_all"], confirm=True)
    assert "governance" in out.lower()
    assert CALLS == []  # the destructive tool never executed on the async path either
    assert ("delete_all", "deny") in _decisions(audit)


def test_governed_functions_stay_drop_in_compatible():
    # The hook rides on the Function; the tool's own schema is untouched, so Agno and the model
    # see exactly the tool they would have seen without governance.
    tools, _ = _governed(approve=True)
    f = tools["run_shell"]
    assert f.name == "run_shell"
    assert "command" in (f.parameters.get("properties") or {})
