"""Agent loop — simple while-loop, sync + async.

LangChain equivalent: 2,007 lines (factory.py) + LangGraph dependency + middleware.
infy: ~150 lines, no graph, no LangGraph.

An optional ``governance`` object (infy.governance.Governance) plugs in at the model and tool
chokepoints. When omitted the loop is unchanged; when present, every tool call is policy-checked
in-process (deny / require-approval) and every step is written to a tamper-evident audit log.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from infy.messages import AIMessage, Message, ToolMessage
from infy.models import ChatModel, ToolSchema
from infy.tools import Tool

if TYPE_CHECKING:
    from infy.governance import Governance
    from infy.governance.approval import PendingApproval
    from infy.governance.types import ToolDecision


@dataclass
class AgentResult:
    """The output of an agent run."""

    messages: list[Message]
    response: AIMessage
    iterations: int = 0
    tool_calls_made: int = 0
    # Durable approval: "completed" or "suspended". A suspended run is waiting on the human
    # decisions in ``pending_approvals`` and continues via DurableAgent.resume(run_id, ...).
    status: str = "completed"
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    run_id: str | None = None


def create_agent(
    model: ChatModel,
    tools: list[Tool] | None = None,
    *,
    system_prompt: str | None = None,
    max_iterations: int = 10,
    parallel_tools: bool = True,
    governance: Governance | None = None,
) -> Callable[..., AgentResult]:
    """Create an agent. Returns a sync callable. Use create_async_agent for async."""
    tool_map: dict[str, Tool] = {t.name: t for t in (tools or [])}
    tool_schemas: list[ToolSchema] = [t.to_schema() for t in (tools or [])]

    def agent(input: str | list[Message]) -> AgentResult:
        messages = _to_messages(input, system_prompt)
        total_tool_calls = 0

        for iteration in range(max_iterations):
            if governance is not None:
                governance.before_model(messages)
            response = model.generate(messages, tools=tool_schemas if tool_schemas else None)
            if governance is not None:
                governance.after_model(response)
            messages.append(response)

            if not response.tool_calls:
                return AgentResult(
                    messages=messages,
                    response=response,
                    iterations=iteration + 1,
                    tool_calls_made=total_tool_calls,
                )

            total_tool_calls += len(response.tool_calls)

            if parallel_tools:
                _execute_tools_parallel(response, messages, tool_map, governance)
            else:
                _execute_tools_sequential(response, messages, tool_map, governance)

        return AgentResult(
            messages=messages,
            response=response,
            iterations=max_iterations,
            tool_calls_made=total_tool_calls,
        )

    return agent


def create_async_agent(
    model: ChatModel,
    tools: list[Tool] | None = None,
    *,
    system_prompt: str | None = None,
    max_iterations: int = 10,
    parallel_tools: bool = True,
    governance: Governance | None = None,
) -> Callable[..., Any]:
    """Create an async agent. Returns a coroutine that yields AgentResult."""
    tool_map: dict[str, Tool] = {t.name: t for t in (tools or [])}
    tool_schemas: list[ToolSchema] = [t.to_schema() for t in (tools or [])]

    async def agent(input: str | list[Message]) -> AgentResult:
        messages = _to_messages(input, system_prompt)
        total_tool_calls = 0

        for iteration in range(max_iterations):
            if governance is not None:
                governance.before_model(messages)
            response = await model.agenerate(messages, tools=tool_schemas if tool_schemas else None)
            if governance is not None:
                governance.after_model(response)
            messages.append(response)

            if not response.tool_calls:
                return AgentResult(
                    messages=messages,
                    response=response,
                    iterations=iteration + 1,
                    tool_calls_made=total_tool_calls,
                )

            total_tool_calls += len(response.tool_calls)

            await _aexecute_tools(
                response, messages, tool_map, parallel=parallel_tools, governance=governance
            )

        return AgentResult(
            messages=messages,
            response=response,
            iterations=max_iterations,
            tool_calls_made=total_tool_calls,
        )

    return agent


def _to_messages(input: str | list[Message], system_prompt: str | None) -> list[Message]:
    from infy.messages import HumanMessage, SystemMessage

    messages: list[Message] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    if isinstance(input, str):
        messages.append(HumanMessage(content=input))
    elif isinstance(input, list):
        messages.extend(input)
    return messages


def _run_tool(
    tc: Any, tool_map: dict[str, Tool], governance: Governance | None = None
) -> ToolMessage:
    """Execute a single tool call, always returning a ToolMessage.

    Every tool call must produce exactly one result message — including unknown tools,
    policy denials, and failures — otherwise the next model call sees an orphaned tool call
    and most providers reject the request.
    """
    tool = tool_map.get(tc.name)
    if tool is None:
        if governance is not None:
            with contextlib.suppress(Exception):
                governance.on_unknown_tool(tc.name)
        return ToolMessage(content=f"Unknown tool: {tc.name}", tool_call_id=tc.id, status="error")
    if governance is not None:
        try:
            decision = governance.before_tool(tool, tc.args)
        except Exception as e:  # the seam fails closed even if governance itself throws
            return ToolMessage(
                content=f"Blocked: governance error [{e}]", tool_call_id=tc.id, status="error"
            )
        if not decision.allowed:
            return ToolMessage(content=decision.message, tool_call_id=tc.id, status="error")
    try:
        result = ToolMessage(content=str(tool.invoke(tc.args)), tool_call_id=tc.id)
    except Exception as e:
        result = ToolMessage(content=f"Error: {e}", tool_call_id=tc.id, status="error")
    if governance is not None:
        with contextlib.suppress(Exception):  # post-call audit is best-effort
            result = governance.after_tool(tool, result)
    return result


def _run_tool_decided(
    tc: Any,
    tool_map: dict[str, Tool],
    governance: Governance | None,
    decision: ToolDecision | None,
) -> ToolMessage:
    """Execute a tool call whose governance decision was ALREADY made (durable pre-authorization),
    so an entire batch is decided before any side effect runs. Same guarantees as ``_run_tool``:
    exactly one ToolMessage out, and fail-closed when the decision is missing or not allowed."""
    tool = tool_map.get(tc.name)
    if tool is None:
        if governance is not None:
            with contextlib.suppress(Exception):
                governance.on_unknown_tool(tc.name)
        return ToolMessage(content=f"Unknown tool: {tc.name}", tool_call_id=tc.id, status="error")
    if decision is None or not decision.allowed:
        message = decision.message if decision is not None else "Blocked: no decision recorded"
        return ToolMessage(content=message, tool_call_id=tc.id, status="error")
    try:
        result = ToolMessage(content=str(tool.invoke(tc.args)), tool_call_id=tc.id)
    except Exception as e:
        result = ToolMessage(content=f"Error: {e}", tool_call_id=tc.id, status="error")
    if governance is not None:
        with contextlib.suppress(Exception):  # post-call audit is best-effort
            result = governance.after_tool(tool, result)
    return result


def _execute_tools_parallel(
    response: AIMessage,
    messages: list[Message],
    tool_map: dict[str, Tool],
    governance: Governance | None = None,
) -> None:
    calls = list(response.tool_calls)
    results: list[ToolMessage | None] = [None] * len(calls)
    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as ex:
        future_to_index = {
            ex.submit(_run_tool, tc, tool_map, governance): i for i, tc in enumerate(calls)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:  # one bad call must not orphan its siblings
                results[i] = ToolMessage(
                    content=f"Error: {e}", tool_call_id=calls[i].id, status="error"
                )
    # Preserve request order so each result lines up with its tool call.
    messages.extend(r for r in results if r is not None)


def _execute_tools_sequential(
    response: AIMessage,
    messages: list[Message],
    tool_map: dict[str, Tool],
    governance: Governance | None = None,
) -> None:
    for tc in response.tool_calls:
        messages.append(_run_tool(tc, tool_map, governance))


async def _aexecute_tools(
    response: AIMessage,
    messages: list[Message],
    tool_map: dict[str, Tool],
    *,
    parallel: bool = True,
    governance: Governance | None = None,
) -> None:
    """Execute tool calls, returning one ToolMessage per call in request order."""

    async def _run_one(tc: Any) -> ToolMessage:
        tool = tool_map.get(tc.name)
        if tool is None:
            if governance is not None:
                with contextlib.suppress(Exception):
                    governance.on_unknown_tool(tc.name)
            return ToolMessage(
                content=f"Unknown tool: {tc.name}", tool_call_id=tc.id, status="error"
            )
        if governance is not None:
            try:
                decision = governance.before_tool(tool, tc.args)
            except Exception as e:  # the seam fails closed even if governance itself throws
                return ToolMessage(
                    content=f"Blocked: governance error [{e}]", tool_call_id=tc.id, status="error"
                )
            if not decision.allowed:
                return ToolMessage(content=decision.message, tool_call_id=tc.id, status="error")
        try:
            result = ToolMessage(content=str(await tool.ainvoke(tc.args)), tool_call_id=tc.id)
        except Exception as e:
            result = ToolMessage(content=f"Error: {e}", tool_call_id=tc.id, status="error")
        if governance is not None:
            with contextlib.suppress(Exception):  # post-call audit is best-effort
                result = governance.after_tool(tool, result)
        return result

    calls = list(response.tool_calls)
    gathered: list[Any]
    if parallel:
        gathered = list(
            await asyncio.gather(*[_run_one(tc) for tc in calls], return_exceptions=True)
        )
    else:
        gathered = []
        for tc in calls:
            try:
                gathered.append(await _run_one(tc))
            except Exception as exc:  # defense in depth; _run_one already fails closed
                gathered.append(exc)
    for tc, item in zip(calls, gathered, strict=True):
        if isinstance(item, BaseException):
            messages.append(
                ToolMessage(content=f"Error: {item}", tool_call_id=tc.id, status="error")
            )
        else:
            messages.append(item)
