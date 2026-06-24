"""Prebuilt graph agent — create_react_agent with state persistence.

LangGraph: 1,015 lines (chat_agent_executor) + 2,030 lines (tool_node) = 3,045 lines.
infy: ~120 lines. Reuses infy's existing agents, tools, and messages.
"""

from __future__ import annotations

from typing import Any

from infy.graph.checkpoint import BaseCheckpointSaver
from infy.graph.state import CompiledGraph, StateGraph
from infy.graph.types import END, START
from infy.messages import AIMessage, SystemMessage, ToolMessage
from infy.models import ChatModel
from infy.tools import Tool


def create_graph_agent(
    model: ChatModel,
    tools: list[Tool] | None = None,
    *,
    system_prompt: str | None = None,
    state_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    max_iterations: int = 10,
) -> CompiledGraph:
    """Create a stateful graph-based agent with checkpoint support.

    This is the graph version of infy's create_agent(). It returns a CompiledGraph
    that supports:
    - State persistence across invocations (checkpointer)
    - Human-in-the-loop (interrupt_before/after=["tools"])
    - State inspection (get_state)
    - State modification (update_state)

    Usage:
        from infy.graph import create_graph_agent, InMemorySaver
        from infy.providers.openai import OpenAIChat

        agent = create_graph_agent(
            OpenAIChat("gpt-4o"),
            tools=[search, calculator],
            system_prompt="You are helpful.",
            checkpointer=InMemorySaver(),
        )

        # First invocation
        result = agent.invoke(
            {"messages": [HumanMessage(content="Hi")]},
            config={"configurable": {"thread_id": "1"}}
        )

        # Resume (continues from checkpoint)
        result2 = agent.invoke(
            {"messages": [HumanMessage(content="What was my first message?")]},
            config={"configurable": {"thread_id": "1"}}
        )

        # Inspect state
        state = agent.get_state(config)
    """
    tools = tools or []
    tool_map = {t.name: t for t in tools}
    tool_schemas = [t.to_schema() for t in tools]

    # --- Node functions ---

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Call the model with current messages."""
        messages = list(state.get("messages", []))

        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = model.generate(messages, tools=tool_schemas if tool_schemas else None)
        return {"messages": messages + [response]}

    def tools_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute tool calls from the last AI message."""
        messages = list(state.get("messages", []))
        if not messages:
            return {}

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}

        tool_messages = []
        for tc in last_msg.tool_calls:
            # Handle both ToolCall objects and dict tool calls
            if isinstance(tc, dict):
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                tc_id = tc.get("id", "")
            else:
                tc_name = tc.name
                tc_args = tc.args
                tc_id = tc.id

            if tc_name in tool_map:
                try:
                    result = tool_map[tc_name].invoke(tc_args)
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(content=f"Error: {e}", tool_call_id=tc_id, status="error")
                    )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tc_name}",
                        tool_call_id=tc_id,
                        status="error",
                    )
                )

        return {"messages": messages + tool_messages}

    def should_continue(state: dict[str, Any]) -> str:
        """Route based on whether the last message has tool calls."""
        messages = state.get("messages", [])
        if not messages:
            return END

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"

        return END

    # --- Build graph ---

    state_cls = state_schema or dict

    graph = StateGraph(state_cls)
    graph.add_node("agent", agent_node)

    if tools:
        graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")

    if tools:
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
