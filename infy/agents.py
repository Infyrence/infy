"""Agent loop — simple while-loop, sync + async.

LangChain equivalent: 2,007 lines (factory.py) + LangGraph dependency + middleware.
infy: ~150 lines, no graph, no LangGraph.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from infy.messages import AIMessage, Message, ToolMessage
from infy.models import ChatModel, ToolSchema
from infy.tools import Tool


@dataclass
class AgentResult:
    """The output of an agent run."""

    messages: list[Message]
    response: AIMessage
    iterations: int = 0
    tool_calls_made: int = 0


def create_agent(
    model: ChatModel,
    tools: list[Tool] | None = None,
    *,
    system_prompt: str | None = None,
    max_iterations: int = 10,
    parallel_tools: bool = True,
) -> Callable[..., AgentResult]:
    """Create an agent. Returns a sync callable. Use create_async_agent for async."""
    tool_map: dict[str, Tool] = {t.name: t for t in (tools or [])}
    tool_schemas: list[ToolSchema] = [t.to_schema() for t in (tools or [])]

    def agent(input: str | list[Message]) -> AgentResult:
        messages = _to_messages(input, system_prompt)
        total_tool_calls = 0

        for iteration in range(max_iterations):
            response = model.generate(messages, tools=tool_schemas if tool_schemas else None)
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
                _execute_tools_parallel(response, messages, tool_map)
            else:
                _execute_tools_sequential(response, messages, tool_map)

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
) -> Callable[..., Any]:
    """Create an async agent. Returns a coroutine that yields AgentResult."""
    tool_map: dict[str, Tool] = {t.name: t for t in (tools or [])}
    tool_schemas: list[ToolSchema] = [t.to_schema() for t in (tools or [])]

    async def agent(input: str | list[Message]) -> AgentResult:
        messages = _to_messages(input, system_prompt)
        total_tool_calls = 0

        for iteration in range(max_iterations):
            response = await model.agenerate(messages, tools=tool_schemas if tool_schemas else None)
            messages.append(response)

            if not response.tool_calls:
                return AgentResult(
                    messages=messages,
                    response=response,
                    iterations=iteration + 1,
                    tool_calls_made=total_tool_calls,
                )

            total_tool_calls += len(response.tool_calls)

            await _aexecute_tools(response, messages, tool_map, parallel=parallel_tools)

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


def _run_tool(tc: Any, tool_map: dict[str, Tool]) -> ToolMessage:
    """Execute a single tool call, always returning a ToolMessage.

    Every tool call must produce exactly one result message — including unknown
    tools and failures — otherwise the next model call sees an orphaned tool
    call and most providers reject the request.
    """
    if tc.name not in tool_map:
        return ToolMessage(content=f"Unknown tool: {tc.name}", tool_call_id=tc.id, status="error")
    try:
        return ToolMessage(content=str(tool_map[tc.name].invoke(tc.args)), tool_call_id=tc.id)
    except Exception as e:
        return ToolMessage(content=f"Error: {e}", tool_call_id=tc.id, status="error")


def _execute_tools_parallel(
    response: AIMessage, messages: list[Message], tool_map: dict[str, Tool]
) -> None:
    calls = list(response.tool_calls)
    results: list[ToolMessage | None] = [None] * len(calls)
    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as ex:
        future_to_index = {ex.submit(_run_tool, tc, tool_map): i for i, tc in enumerate(calls)}
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    # Preserve request order so each result lines up with its tool call.
    messages.extend(r for r in results if r is not None)


def _execute_tools_sequential(
    response: AIMessage, messages: list[Message], tool_map: dict[str, Tool]
) -> None:
    for tc in response.tool_calls:
        messages.append(_run_tool(tc, tool_map))


async def _aexecute_tools(
    response: AIMessage,
    messages: list[Message],
    tool_map: dict[str, Tool],
    *,
    parallel: bool = True,
) -> None:
    """Execute tool calls, returning one ToolMessage per call in request order."""

    async def _run_one(tc: Any) -> ToolMessage:
        if tc.name not in tool_map:
            return ToolMessage(
                content=f"Unknown tool: {tc.name}", tool_call_id=tc.id, status="error"
            )
        try:
            result = await tool_map[tc.name].ainvoke(tc.args)
            return ToolMessage(content=str(result), tool_call_id=tc.id)
        except Exception as e:
            return ToolMessage(content=f"Error: {e}", tool_call_id=tc.id, status="error")

    if parallel:
        results = await asyncio.gather(*[_run_one(tc) for tc in response.tool_calls])
    else:
        results = [await _run_one(tc) for tc in response.tool_calls]
    messages.extend(results)
