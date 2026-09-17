"""Infyrence governing a real Agno agent's tools.

An Agno agent runs a sequence of actions. infy governance sits at the tool chokepoint as an Agno
``tool_hook``: reads run, the destructive action is denied by policy and never executes, and the
high-risk deploy pauses for a human before it is allowed. Every decision lands in a tamper-evident
audit trail that verify() confirms was not altered.

The tool calls here are scripted and dispatched through ``FunctionCall.execute()``, which is the
same call Agno's model layer makes when a model asks for a tool (``agno/models/base.py``), so the
governed path exercised below is the production one. Scripting the calls rather than a model keeps
the demo deterministic and free of any API key; attach the same hook to a live
``Agent(model=..., tool_hooks=[...])`` and it behaves identically, because the hook rides on the
tool, not on the model.

Run:  python examples/agno_governance_demo.py
Needs: pip install agno
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # run from anywhere

from typing import Any

from agno.tools.function import Function, FunctionCall

from infy.governance import AuditLog, CallbackApprover, Governance, Policy
from infy.integrations.agno import govern_hook

EXECUTED: list[str] = []


def read_config(path: str) -> str:
    """Read a configuration file."""
    EXECUTED.append(f"read_config({path})")
    return "region=prod, replicas=3"


def delete_database(name: str) -> str:
    """Delete a database. Destructive and irreversible."""
    EXECUTED.append(f"delete_database({name})")
    return f"DELETED {name}"


def deploy(target: str) -> str:
    """Deploy a build to an environment."""
    EXECUTED.append(f"deploy({target})")
    return f"deployed {target}"


def approve(request: Any) -> bool:
    print(f"  [human] approval requested for '{request.tool}' -> approving")
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
        principal="agent://agno/sre",
        escalate_at=None,
    )

    # One hook governs every tool the agent executes in-process. On a live agent this is simply
    # Agent(model=..., tools=[...], tool_hooks=[hook]).
    hook = govern_hook(
        gov,
        risk={
            "read_config": {"verb": "READ", "risk_tier": "low", "side_effect": False},
            "delete_database": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
            "deploy": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
        },
    )

    tools = {}
    for fn in (read_config, delete_database, deploy):
        f = Function.from_callable(fn)
        f.tool_hooks = [hook]
        tools[f.name] = f

    script: list[tuple[str, dict[str, Any]]] = [
        ("read_config", {"path": "prod.yaml"}),
        ("delete_database", {"name": "orders_prod"}),  # destructive, must be denied
        ("deploy", {"target": "checkout-hotfix"}),  # high-risk, needs a human
    ]

    print("Infyrence governing an Agno agent\n")
    for name, args in script:
        result = FunctionCall(function=tools[name], arguments=args).execute().result
        print(f"  model called {name}({args}) -> {result}")

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
