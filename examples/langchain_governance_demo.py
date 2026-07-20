"""Infyrence governing a real LangChain agent's tools.

Real LangChain tools are wrapped with infy governance and driven through an incident-response
sequence, the way an agent would call them (each through the real ``tool.invoke`` chokepoint).
Reads run, the destructive action is denied by policy and never executes, and the high-risk deploy
pauses for a human before it is allowed. Every decision lands in a tamper-evident audit trail that
verify() confirms was not altered.

Pointing a real LLM at these same governed tools (create_react_agent, AgentExecutor, bind_tools)
behaves identically, because the agent calls tool.invoke just like this demo does, and the tests
cover exactly that chokepoint.

Run:  python examples/langchain_governance_demo.py
Needs: pip install langchain-core
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # run from anywhere

from langchain_core.tools import tool

from infy.governance import AuditLog, CallbackApprover, Governance, Policy
from infy.integrations.langchain import govern

EXECUTED: list[str] = []


@tool
def read_config(path: str) -> str:
    """Read a configuration file."""
    EXECUTED.append(f"read_config({path})")
    return "region=prod, replicas=3"


@tool
def delete_database(name: str) -> str:
    """Delete a database. Destructive and irreversible."""
    EXECUTED.append(f"delete_database({name})")
    return f"DELETED {name}"


@tool
def deploy(target: str) -> str:
    """Deploy a build to an environment."""
    EXECUTED.append(f"deploy({target})")
    return f"deployed {target}"


def approve(request: object) -> bool:
    print(f"  [human] approval requested for '{getattr(request, 'tool', '?')}' -> approving")
    return True


def main() -> None:
    audit = AuditLog()
    gov = Governance(
        policy=Policy(
            allow=["read_config", "deploy"],  # deny-by-default: delete_database is not allowlisted
            deny=["delete_database"],
            require_approval=["deploy"],
        ),
        approver=CallbackApprover(approve),
        audit=audit,
        principal="agent://langchain/sre",
        escalate_at=None,
    )

    tools = {
        t.name: t
        for t in govern(
            [read_config, delete_database, deploy],
            gov,
            risk={
                "read_config": {"verb": "READ", "risk_tier": "low", "side_effect": False},
                "delete_database": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
                "deploy": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
            },
        )
    }

    # What a LangChain agent would decide to call, step by step.
    plan = [
        ("read_config", {"path": "prod.yaml"}),
        ("delete_database", {"name": "orders_prod"}),  # destructive, must be denied
        ("deploy", {"target": "checkout-hotfix"}),  # high-risk, needs a human
    ]

    print("Infyrence governing LangChain tools\n")
    for name, args in plan:
        result = tools[name].invoke(args)
        print(f"  agent -> {name}{tuple(args.values())}: {result}")

    print("\nGovernance decisions (in-process, tamper-evident):")
    for e in audit.events:
        if e.action == "invoke_tool":
            reason = ", ".join(e.reasons)
            print(f"  {e.decision.upper():<9} {e.resource:<16} risk={e.risk_tier:<8} ({reason})")

    print(f"\nActions that actually executed: {EXECUTED}")
    print("The destructive delete_database never ran.")
    print(f"\nAudit chain verify(): {audit.verify()}  ({len(audit.events)} events, hash-chained)")


if __name__ == "__main__":
    main()
