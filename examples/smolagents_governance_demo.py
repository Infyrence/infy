"""Infyrence governing a real smolagents agent.

A live smolagents ToolCallingAgent runs a sequence of actions. infy governance sits at the tool
chokepoint: reads run, the destructive action is denied by policy and never executes, and the
high-risk deploy pauses for a human before it is allowed. Every decision lands in a tamper-evident
audit trail that verify() confirms was not altered.

The model here is scripted so the demo is deterministic and needs no API key. Point a real model
(OpenAIServerModel, LiteLLMModel, and so on) at the same governed tools and it behaves identically,
because the governed tools are drop-in replacements.

Run:  python examples/smolagents_governance_demo.py
Needs: pip install smolagents
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # run from anywhere

from typing import Any

from smolagents import ChatMessage, ChatMessageToolCall, Model, ToolCallingAgent, tool
from smolagents.models import ChatMessageToolCallFunction, MessageRole, TokenUsage

from infy.governance import AuditLog, CallbackApprover, Governance, Policy
from infy.integrations.smolagents import govern

EXECUTED: list[str] = []


@tool
def read_config(path: str) -> str:
    """Read a configuration file.

    Args:
        path: the config file path
    """
    EXECUTED.append(f"read_config({path})")
    return "region=prod, replicas=3"


@tool
def delete_database(name: str) -> str:
    """Delete a database. Destructive and irreversible.

    Args:
        name: the database to delete
    """
    EXECUTED.append(f"delete_database({name})")
    return f"DELETED {name}"


@tool
def deploy(target: str) -> str:
    """Deploy a build to an environment.

    Args:
        target: the deploy target
    """
    EXECUTED.append(f"deploy({target})")
    return f"deployed {target}"


class ScriptedModel(Model):
    """A deterministic model: emits a fixed sequence of tool calls, then finishes."""

    def __init__(self, script: list[tuple[str, dict[str, Any]]]) -> None:
        super().__init__()
        self._script = list(script)
        self._i = 0

    def generate(self, messages: Any, **kwargs: Any) -> ChatMessage:
        if self._i < len(self._script):
            step = self._script[self._i]
        else:
            step = ("final_answer", {"answer": "done"})
        self._i += 1
        name, args = step
        call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(name=name, arguments=args),
            id=f"call_{self._i}",
            type="function",
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[call],
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


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
        principal="agent://smolagents/sre",
        escalate_at=None,
    )

    tools = govern(
        [read_config, delete_database, deploy],
        gov,
        risk={
            "read_config": {"verb": "READ", "risk_tier": "low", "side_effect": False},
            "delete_database": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
            "deploy": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
        },
    )

    model = ScriptedModel(
        [
            ("read_config", {"path": "prod.yaml"}),
            ("delete_database", {"name": "orders_prod"}),  # destructive, must be denied
            ("deploy", {"target": "checkout-hotfix"}),  # high-risk, needs a human
        ]
    )

    agent = ToolCallingAgent(tools=tools, model=model, max_steps=6, verbosity_level=0)

    print("Infyrence governing a smolagents ToolCallingAgent\n")
    agent.run("Investigate the incident and remediate it.")

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
