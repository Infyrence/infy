"""infy governance showcase — the 90-second demo.

A production incident. An SRE agent with a 120-tool catalogue and real authority: it can read
logs, restart services, deploy, and drop databases. We give it the authority and then prove,
afterwards, exactly what it did.

Five things happen, in order:

  1. TOOL ROUTING   120 tools in the catalogue, only the turn's schemas sent.
  2. ALLOWED        A read-only diagnostic runs. Governance is not in the way.
  3. DENIED         The agent tries to drop the database. Fail-closed, and still audited.
  4. SUSPENDED      A deploy needs a human. The run persists and waits, TOCTOU-safe.
  5. VERIFIED       The audit chain verifies -- and then we tamper with it, and it doesn't.

The model is deterministic and offline: no API key, no network, no cost, and the same output
every time it runs. That is on purpose -- this demo is about the control plane, and a live
model would only add nondeterminism to the part that isn't the point.

Run:  python examples/governance_showcase.py
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from infy import Objective, ToolRouter, tool
from infy.governance import (
    AuditLog,
    DurableAgent,
    DurableApprover,
    Governance,
    InMemoryApprovalStore,
    Policy,
)
from infy.messages import AIMessage, Message, ToolCall, ToolMessage, UsageMetadata
from infy.models import ToolSchema

BOLD, DIM, RED, GREEN, YELLOW, CYAN, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[36m",
    "\033[0m",
)
PAUSE = 0.9  # readable on a screen recording; set to 0 for CI


def say(text: str = "", pause: float = PAUSE) -> None:
    print(text)
    time.sleep(pause)


def rule(title: str) -> None:
    say()
    say(f"{BOLD}{CYAN}{'─' * 74}{RESET}", 0)
    say(f"{BOLD}{CYAN}  {title}{RESET}", 0)
    say(f"{BOLD}{CYAN}{'─' * 74}{RESET}")


# ---------------------------------------------------------------------------
# The agent's authority: four real tools, profiled by risk
# ---------------------------------------------------------------------------


@tool(verb="READ", risk_tier="low")
def read_logs(service: str) -> str:
    """Read recent error logs for a service."""
    return "ERROR rate 12% on api-gateway; OOMKilled x3 in last 5m"


@tool(verb="READ", risk_tier="low")
def check_health(service: str) -> str:
    """Check a service's health endpoint."""
    return "api-gateway: degraded (2/5 replicas ready)"


@tool(verb="EXECUTE", risk_tier="high", side_effect=True)
def deploy_service(service: str, version: str) -> str:
    """Deploy a service version to the production cluster."""
    return f"{service} rolled out at {version}; 5/5 replicas ready"


@tool(verb="DB", risk_tier="critical", side_effect=True)
def delete_database(name: str) -> str:
    """Permanently drop a database. Irreversible."""
    return f"{name} DROPPED"  # never reached: policy denies this tool


CORE_TOOLS = [read_logs, check_health, deploy_service, delete_database]


def build_catalogue() -> list[Any]:
    """The 4 real tools plus 116 plausible integration tools, for 120 total."""
    domains = [
        "github",
        "jira",
        "slack",
        "s3",
        "postgres",
        "stripe",
        "datadog",
        "sendgrid",
        "twilio",
        "notion",
        "gitlab",
        "pagerduty",
    ]
    ops = [
        "list",
        "get",
        "create",
        "update",
        "search",
        "export",
        "audit",
        "permissions",
        "health",
        "archive",
    ]
    filler = [
        tool(name=f"{d}_{o}", verb="READ", risk_tier="low")(
            _make_stub(f"{o.capitalize()} {d} records.")
        )
        for d in domains
        for o in ops
    ][:116]
    return CORE_TOOLS + filler


def _make_stub(doc: str):
    def stub(query: str = "") -> str:
        return "ok"

    stub.__doc__ = doc
    return stub


# ---------------------------------------------------------------------------
# A deterministic model: a fixed incident-response plan, no network
# ---------------------------------------------------------------------------


class IncidentModel:
    """Replays one scripted incident response, one tool call per turn."""

    model_name = "scripted-sre"

    def __init__(self) -> None:
        self.plan = [
            ("read_logs", {"service": "api-gateway"}),
            ("delete_database", {"name": "prod-orders"}),
            ("deploy_service", {"service": "api-gateway", "version": "v1.2.1"}),
        ]
        self.offered: list[int] = []

    def generate(
        self, messages: list[Message], *, tools: list[ToolSchema] | None = None, **kw: Any
    ) -> AIMessage:
        self.offered.append(len(tools or []))
        step = sum(1 for m in messages if isinstance(m, ToolMessage))
        usage = UsageMetadata(input_tokens=40, output_tokens=12)
        if step >= len(self.plan):
            done = "Incident resolved: api-gateway healthy on v1.2.1."
            return AIMessage(content=done, usage=usage)
        name, args = self.plan[step]
        return AIMessage(
            content="", tool_calls=[ToolCall(name=name, args=args, id=f"c{step}")], usage=usage
        )


def main() -> None:
    catalogue = build_catalogue()

    say()
    say(f"{BOLD}  infy — give an agent real authority, then prove what it did{RESET}")
    say(f"{DIM}  Production incident. Deterministic model, no network, no API key.{RESET}")

    # -- 1. Tool routing ----------------------------------------------------
    rule("1 · TOOL ROUTING — context is not free")
    router = ToolRouter(tools=catalogue, top_k=5, always=["read_logs"])
    selected = router.select("api-gateway is erroring, check the logs and redeploy it")
    say(f"  catalogue        {BOLD}{len(catalogue)}{RESET} tools")
    say(f"  schemas sent     {BOLD}{len(selected)}{RESET} → {', '.join(t.name for t in selected)}")
    n, rest = len(selected), len(catalogue) - len(selected)
    say(f"  {DIM}top_k is 5; only {n} matched, so only {n} were sent — the router{RESET}", 0)
    say(f"  {DIM}will not pad context with schemas this turn has no reason to want.{RESET}", 0)
    say(f"  {DIM}The other {rest} stay visible as a one-line summary pool.{RESET}", 0)
    say(f"  {DIM}Sending all 120 every turn costs ~24k tokens. This is 8.6x lighter.{RESET}")

    # -- Governance ---------------------------------------------------------
    rule("2 · POLICY — deny-by-default, fail-closed")
    audit = AuditLog()
    store = InMemoryApprovalStore()
    gov = Governance(
        policy=Policy(
            allow=[t.name for t in catalogue],  # everything in the catalogue is in scope...
            deny=["delete_database"],  # ...except this, never, at any risk tier
            require_approval=["deploy_service"],  # ...and this needs a human
        ),
        approver=DurableApprover(store),
        audit=audit,
        principal="agent://acme/sre-oncall",
    )
    never = f"{DIM}(never, regardless of approval){RESET}"
    human = f"{DIM}(allowed only with a human yes){RESET}"
    say(f"  principal        {BOLD}agent://acme/sre-oncall{RESET}")
    say(f"  {RED}deny{RESET}             delete_database        {never}")
    say(f"  {YELLOW}require_approval{RESET} deploy_service         {human}")
    say(f"  {GREEN}allow{RESET}            read_logs, check_health, +116 read-only integrations")

    objective = Objective(
        goal="Restore api-gateway to healthy.",
        constraints=["Never destroy data.", "Production changes need a human."],
    )

    model = IncidentModel()
    agent = DurableAgent(model, catalogue, governance=gov, store=store)

    # -- 3. Run -------------------------------------------------------------
    rule("3 · THE RUN — allowed, denied, suspended")
    say(f"  {DIM}objective: {objective.goal}{RESET}")
    say()
    result = agent.run("api-gateway is erroring. Diagnose and fix it.", run_id="inc-4821")

    for m in result.messages:
        if isinstance(m, ToolMessage):
            blocked = m.status == "error"
            mark = f"{RED}✗ BLOCKED{RESET}" if blocked else f"{GREEN}✓ ran{RESET}"
            say(f"  {mark}  {m.content[:66]}")

    if result.status == "suspended":
        pending = result.pending_approvals[0]
        say()
        say(f"  {YELLOW}⏸ SUSPENDED{RESET}  run persisted, waiting on a human")
        say(f"     tool        {pending.request.tool}")
        say(f"     fingerprint {DIM}{pending.fingerprint[:32]}…{RESET}")
        say(f"     {DIM}The approval is bound to this exact action. Change one argument{RESET}", 0)
        say(f"     {DIM}and the fingerprint no longer matches — TOCTOU-safe.{RESET}")

        # -- 4. Human decides, hours later ---------------------------------
        rule("4 · HUMAN APPROVES — the run resumes where it stopped")
        say(f"  {DIM}$ infyctl approve inc-4821 --yes      (minutes or hours later){RESET}")
        result = agent.resume("inc-4821", {pending.fingerprint: True})
        for m in result.messages[-3:]:
            if isinstance(m, ToolMessage) and m.status != "error":
                say(f"  {GREEN}✓ ran{RESET}  {m.content[:66]}")

    say()
    say(f"  {BOLD}final:{RESET} {result.response.text}")

    # -- 5. The receipt -----------------------------------------------------
    rule("5 · THE RECEIPT — tamper-evident, and we prove it")
    say(f"  {len(audit)} events recorded, SHA-256 hash-chained. The tool decisions:")
    say()
    decisions = [e for e in audit.events if e.action in ("invoke_tool", "tool_result")]
    for e in decisions:
        colour = RED if e.decision == "deny" else GREEN if e.decision == "allow" else DIM
        say(
            f"  {DIM}{e.seq:>2}{RESET}  {colour}{e.decision:<8}{RESET}{e.action:<13}"
            f"{e.resource:<17}{DIM}{e.risk_tier:<9}{e.hash[:12]}{RESET}",
            0.3,
        )

    say()
    say(f"  audit.verify()  →  {GREEN}{BOLD}{audit.verify()}{RESET}")
    say()
    say(f"  {DIM}Now someone edits the log to hide the denied database drop:{RESET}")

    # Rewrite one event in place — exactly what a cover-up looks like.
    victim = next(i for i, e in enumerate(audit.events) if e.decision == "deny")
    audit._events[victim] = dataclasses.replace(audit._events[victim], decision="allow")
    say(f"  {DIM}  event #{audit.events[victim].seq}: deny → allow{RESET}")
    say()
    say(f"  audit.verify()  →  {RED}{BOLD}{audit.verify()}{RESET}")
    say()
    say(f"  {BOLD}The record cannot be edited without the chain saying so.{RESET}")
    say(f"  {DIM}That is the difference between a log and evidence.{RESET}")
    say()
    say(f"{DIM}  pip install infy  ·  github.com/Infyrence/infy{RESET}")
    say()


if __name__ == "__main__":
    main()
