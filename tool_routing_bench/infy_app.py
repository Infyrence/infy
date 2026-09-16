"""infy arm — the SAME create_agent loop, with tool routing off or on.

``build_static()`` is today's behaviour: every one of the 120 schemas is serialised into every
model call, on every iteration of the loop. ``build_routed(k)`` attaches a ``ToolRouter``, which
sends a one-line summary of all 120 tools plus the full schema of the top k. The delta between
them is the context cost of static schema injection.

The model is a deterministic stub that records, per turn, exactly what it was offered — so the
token numbers are measured off the real serialised payload, not estimated from the catalogue.
"""

from __future__ import annotations

import json
from typing import Any

import common

from infy.agents import create_agent
from infy.messages import AIMessage, Message, SystemMessage, ToolCall, ToolMessage, UsageMetadata
from infy.models import ToolSchema
from infy.tokens import count_tokens
from infy.tool_router import ToolRouter

CATALOGUE = common.build_catalogue()


def _schema_tokens(tools: list[ToolSchema] | None) -> int:
    """Tokens of the tool payload exactly as it goes on the wire."""
    if not tools:
        return 0
    return count_tokens(json.dumps([t.to_dict() for t in tools]))


class FakeModel:
    """Deterministic, offline: call the target tool once, then write the final answer.

    Records the tool-schema token count it was offered on each turn.
    """

    model_name = "fake"

    def __init__(self) -> None:
        self.schema_tokens: list[int] = []
        self.offered: list[list[str]] = []

    def generate(
        self, messages: list[Message], *, tools: list[ToolSchema] | None = None, **kwargs: Any
    ) -> AIMessage:
        self.schema_tokens.append(_schema_tokens(tools))
        self.offered.append([t.name for t in (tools or [])])
        usage = UsageMetadata(input_tokens=20, output_tokens=8)
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content=common.FINAL, usage=usage)
        call = ToolCall(name=common.TARGET, args=common.TARGET_ARGS, id="c0")
        return AIMessage(content="", tool_calls=[call], usage=usage)


def build_static() -> Any:
    # Tools run sequentially so the only variable between arms is routing — a per-run
    # ThreadPoolExecutor would add ms of churn that swamps the us-scale routing cost.
    return create_agent(FakeModel(), CATALOGUE, parallel_tools=False)


def build_routed(top_k: int = 5) -> Any:
    router = ToolRouter(tools=CATALOGUE, top_k=top_k)
    return create_agent(FakeModel(), CATALOGUE, tool_router=router, parallel_tools=False)


def build_routed_3() -> Any:
    return build_routed(3)


# The harness calls build() for cold-start/RSS; default to the routed arm.
def build() -> Any:
    return build_routed()


def run(agent: Any) -> dict[str, Any]:
    result = agent(common.QUERY)
    model = agent.__closure__ and _model_of(agent)
    pool_tokens = sum(
        count_tokens(m.content)
        for m in result.messages
        if isinstance(m, SystemMessage) and m.content.startswith("Available tools.")
    )
    return {
        "final": result.response.text,
        "tool_calls": result.tool_calls_made,
        "turns": result.iterations,
        "schema_tokens": list(model.schema_tokens) if model else [],
        "offered": list(model.offered) if model else [],
        # The summary pool is a message, so it is re-sent on every turn: charge it per turn.
        "pool_tokens": pool_tokens * result.iterations,
        "target_offered": bool(model and common.TARGET in model.offered[0]),
    }


def _model_of(agent: Any) -> FakeModel | None:
    """Recover the FakeModel captured in the agent closure, to read its recorded counts."""
    for cell in agent.__closure__ or ():
        if isinstance(cell.cell_contents, FakeModel):
            return cell.cell_contents
    return None
