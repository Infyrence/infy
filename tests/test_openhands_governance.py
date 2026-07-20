"""Tests for the OpenHands integration core: infy governance classifies and audits an action the
way an OpenHands SecurityAnalyzer needs. The SDK-free ``assess_action`` is fully covered here; the
thin ``build_analyzer`` wrapper (which needs openhands-sdk) is only checked for its lazy-import
contract, since the SDK does not install in this environment."""

import pytest

from infy.governance import AuditLog, Governance, Policy
from infy.integrations.openhands import assess_action, build_analyzer


def _gov():
    audit = AuditLog()
    gov = Governance(
        policy=Policy(allow=["read", "edit", "cmd", "net"], deny=["delete_all"]),
        audit=audit,
        principal="agent://openhands",
        escalate_at=None,
    )
    return gov, audit


def _decisions(audit):
    return [(e.resource, e.decision) for e in audit.events if e.action == "invoke_tool"]


def test_denylisted_action_is_high_and_denied():
    gov, audit = _gov()
    risk, allowed = assess_action(
        gov, tool_name="delete_all", args={"target": "orders_prod"}, verb="DB", risk_tier="critical"
    )
    assert risk == "HIGH"
    assert allowed is False
    assert ("delete_all", "deny") in _decisions(audit)


def test_unlisted_action_denied_by_default():
    gov, audit = _gov()
    risk, allowed = assess_action(gov, tool_name="exfiltrate", args={}, verb="NETWORK")
    assert risk == "HIGH" and allowed is False
    assert ("exfiltrate", "deny") in _decisions(audit)


def test_allowed_read_is_low_and_permitted():
    gov, audit = _gov()
    risk, allowed = assess_action(
        gov, tool_name="read", args={"path": "config.yaml"}, verb="READ", risk_tier="low"
    )
    assert risk == "LOW" and allowed is True
    assert ("read", "allow") in _decisions(audit)


def test_risk_tiers_map_to_openhands_levels():
    gov, _ = _gov()
    assert assess_action(gov, tool_name="edit", args={}, risk_tier="medium")[0] == "MEDIUM"
    assert assess_action(gov, tool_name="cmd", args={}, risk_tier="high")[0] == "HIGH"
    # CRITICAL folds into OpenHands' top tier, HIGH
    assert assess_action(gov, tool_name="net", args={}, risk_tier="critical")[0] == "HIGH"


def test_every_action_is_audited_tamper_evident():
    gov, audit = _gov()
    assess_action(gov, tool_name="read", args={"path": "a"}, risk_tier="low")
    assess_action(gov, tool_name="delete_all", args={"target": "db"}, risk_tier="critical")
    assess_action(gov, tool_name="edit", args={"path": "b"}, risk_tier="medium")
    assert audit.verify()
    assert len(_decisions(audit)) == 3


def test_build_analyzer_imports_sdk_lazily():
    gov, _ = _gov()
    try:
        import openhands.sdk  # noqa: F401
    except Exception:
        # No SDK here: building must fail cleanly (proving the SDK is imported lazily, not at
        # module import). Importing this integration module did not require the SDK.
        with pytest.raises((ImportError, ModuleNotFoundError)):
            build_analyzer(gov)
        return
    analyzer = build_analyzer(gov, risk={"cmd": {"verb": "EXECUTE", "risk_tier": "high"}})
    assert hasattr(analyzer, "security_risk")
