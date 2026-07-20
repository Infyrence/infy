"""Durable, out-of-band human approval — pause a governed run for a real human decision, then
resume it minutes or hours later, instead of blocking a thread on a synchronous approver.

The agent loop suspends at the approval chokepoint (a ``DurableApprover`` raises
``ApprovalRequired``), persists its messages to an ``ApprovalStore`` keyed by ``run_id``, and
returns a suspended ``AgentResult`` carrying what a human must decide. A later
``resume(run_id, decisions)`` replays the pending tool batch with each verdict bound to the exact
``(tool, args)`` fingerprint (TOCTOU-safe) and continues. Fail-closed throughout: an action that
is never approved never runs.

Lives in the governance package (not the core agent loop) so ``infy``'s core stays free of any
governance import; it reuses the core helpers rather than duplicating the loop's guarantees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infy.agents import AgentResult, _run_tool_decided, _to_messages
from infy.governance.approval import (
    ApprovalRequired,
    ApprovalStore,
    PendingApproval,
    run_scope,
)
from infy.messages import AIMessage

if TYPE_CHECKING:
    from infy.governance.governance import Governance
    from infy.governance.types import ToolDecision
    from infy.messages import Message
    from infy.models import ChatModel, ToolSchema
    from infy.tools import Tool


class DurableAgent:
    """A governed agent whose approvals suspend and resume durably.

    Usage::

        store = InMemoryApprovalStore()
        gov = Governance(policy=Policy(require_approval=["pay"]), approver=DurableApprover(store))
        agent = DurableAgent(model, tools, governance=gov, store=store)

        result = agent.run("pay the invoice", run_id="run-1")
        if result.status == "suspended":
            fp = result.pending_approvals[0].fingerprint     # what a human is deciding
            result = agent.resume("run-1", {fp: True})       # a human said yes, hours later

    The default ``create_agent`` is untouched; this is the opt-in durable path.
    """

    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool] | None = None,
        *,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        governance: Governance,
        store: ApprovalStore,
    ) -> None:
        self._model = model
        self._tool_map: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._tool_schemas: list[ToolSchema] = [t.to_schema() for t in (tools or [])]
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._governance = governance
        self._store = store

    def run(self, input: str | list[Message], *, run_id: str) -> AgentResult:
        """Start a run. Returns a completed AgentResult, or a suspended one awaiting approval."""
        messages = _to_messages(input, self._system_prompt)
        return self._drive(messages, run_id)

    def resume(self, run_id: str, decisions: dict[str, bool]) -> AgentResult:
        """Continue a suspended run after recording human verdicts (``fingerprint -> approved``).

        Raises ``KeyError`` if there is no suspended run for ``run_id`` (e.g. it already completed).
        """
        for fp, approved in decisions.items():
            self._store.put_decision(run_id, fp, approved)
        messages = self._store.load_state(run_id)
        if messages is None:
            raise KeyError(f"no suspended run to resume for run_id {run_id!r}")
        return self._drive(messages, run_id)

    def _drive(self, messages: list[Message], run_id: str) -> AgentResult:
        with run_scope(run_id):
            result = self._loop(messages, run_id)
        if result.status == "completed":
            self._store.clear(run_id)
        return result

    def _loop(self, messages: list[Message], run_id: str) -> AgentResult:
        total = 0
        response: AIMessage | None = None
        for iteration in range(self._max_iterations):
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and last.tool_calls:
                # Resume entry: the pending response is already the last message with its tool
                # calls not yet executed (we suspended before running any). Execute them now.
                response = last
            else:
                self._governance.before_model(messages)
                response = self._model.generate(
                    messages, tools=self._tool_schemas if self._tool_schemas else None
                )
                self._governance.after_model(response)
                messages.append(response)

            if not response.tool_calls:
                return AgentResult(
                    messages=messages,
                    response=response,
                    iterations=iteration + 1,
                    tool_calls_made=total,
                    status="completed",
                    run_id=run_id,
                )

            total += len(response.tool_calls)
            try:
                self._execute(response, messages)
            except ApprovalRequired as ar:
                # Persist the run (its messages end at the pending response, no tools run yet) and
                # hand back what a human must decide. Resume continues from exactly here.
                self._store.save_state(run_id, messages)
                return AgentResult(
                    messages=messages,
                    response=response,
                    iterations=iteration + 1,
                    tool_calls_made=total,
                    status="suspended",
                    pending_approvals=[PendingApproval(ar.request, ar.fingerprint)],
                    run_id=run_id,
                )

        assert response is not None  # max_iterations >= 1, so the loop assigned it
        return AgentResult(
            messages=messages,
            response=response,
            iterations=self._max_iterations,
            tool_calls_made=total,
            status="completed",
            run_id=run_id,
        )

    def _execute(self, response: AIMessage, messages: list[Message]) -> None:
        # Decide EVERY tool call before executing ANY, so a pending approval suspends the whole
        # batch before a single side effect runs, and no sibling executes ahead of a denial.
        decisions: dict[str, ToolDecision] = {}
        for tc in response.tool_calls:
            tool = self._tool_map.get(tc.name)
            if tool is None:
                continue  # unknown tool handled at execution time
            decisions[tc.id] = self._governance.before_tool(
                tool, tc.args
            )  # may raise ApprovalRequired
        for tc in response.tool_calls:
            messages.append(
                _run_tool_decided(tc, self._tool_map, self._governance, decisions.get(tc.id))
            )
