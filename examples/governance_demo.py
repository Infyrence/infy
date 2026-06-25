"""Demo: an in-process governance layer for an infy agent.

The first sellable feature — a governed agent that blocks a forbidden action, pauses a
high-risk action for human approval, and produces a tamper-evident audit trail you can
verify and export. Then it measures the per-tool-call governance overhead to show it is
negligible next to a real LLM round-trip. Runs fully offline (no API key).
"""

from __future__ import annotations

import time
from typing import Any

from infy.agents import create_agent
from infy.governance import ApprovalRequest, AuditLog, CallbackApprover, Governance, Policy
from infy.messages import AIMessage, ToolCall, ToolMessage
from infy.tools import tool


class ScriptedModel:
    """Offline fake model that drives a fixed sequence of tool calls, then stops."""

    model_name = "scripted"

    def __init__(self, script: list[tuple[str, dict[str, Any]]]) -> None:
        self._script = script
        self._i = 0

    def generate(self, messages: Any, *, tools: Any = None, **kw: Any) -> AIMessage:
        if self._i < len(self._script):
            name, args = self._script[self._i]
            self._i += 1
            return AIMessage(
                content="", tool_calls=[ToolCall(name=name, args=args, id=f"c{self._i}")]
            )
        return AIMessage(content="All done.")


@tool(verb="READ")
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"(results for {query!r})"


@tool(risk_tier="high", verb="EXECUTE", side_effect=True)
def deploy(service: str) -> str:
    """Deploy a service to production."""
    return f"deployed {service}"


@tool(risk_tier="critical", verb="DB", side_effect=True)
def delete_database(name: str) -> str:
    """Delete a database. Irreversible."""
    return f"DELETED {name}"


def review(request: ApprovalRequest) -> bool:
    # In production this is a Slack / console / queue prompt; here: approve deploys, reject deletes.
    approved = request.tool != "delete_database"
    verdict = "APPROVED" if approved else "REJECTED"
    print(f"    [approval] {request.tool}({request.args}) risk={request.risk_tier} -> {verdict}")
    return approved


def main() -> None:
    gov = Governance(
        policy=Policy(deny=["delete_database"], require_approval=["deploy"]),
        approver=CallbackApprover(review),
        principal="agent://demo/assistant",
    )
    model = ScriptedModel(
        [
            ("search", {"query": "release checklist"}),
            ("deploy", {"service": "billing"}),
            ("delete_database", {"name": "prod"}),
        ]
    )
    agent = create_agent(
        model, tools=[search, deploy, delete_database], governance=gov, parallel_tools=False
    )

    print("=== running governed agent ===")
    result = agent("ship the release")
    for message in result.messages:
        if isinstance(message, ToolMessage):
            print(f"  tool -> {message.content}")

    print("\n=== tamper-evident audit trail ===")
    for event in gov.audit.events:
        row = f"  #{event.seq:<2} {event.action:13} {event.resource:16} {event.decision:9}"
        print(f"{row} {event.risk_tier}")
    print(f"  chain verified: {gov.audit.verify()}")

    print("\n=== governance overhead (in-process, per tool call) ===")
    bench = Governance(policy=Policy(allow=["search"]), audit=AuditLog())
    reps = 50_000
    args = {"query": "x"}
    start = time.perf_counter()
    for _ in range(reps):
        bench.before_tool(search, args)
    micros = (time.perf_counter() - start) / reps * 1e6
    print(f"  before_tool (risk + policy + audit): {micros:.2f} microseconds/call")
    print("  a real LLM tool round-trip:          ~1,000,000-2,000,000 microseconds")
    print(f"  -> governance is ~{2_000_000 / micros:,.0f}x cheaper than the call it guards")


if __name__ == "__main__":
    main()
