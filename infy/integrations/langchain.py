"""Govern a LangChain agent with infy, without changing the agent.

LangChain invokes a tool through ``BaseTool.invoke``. This adapter wraps each tool so that
invocation first asks infy governance whether the action is allowed. A denied or unapproved action
never runs; the model receives the block reason as the tool output. Every decision, allowed or
denied, is written to the tamper-evident audit chain. The wrapped tools are drop-in replacements.

Usage::

    from langchain_core.tools import tool
    from infy.governance import Governance, Policy, CallbackApprover
    from infy.integrations.langchain import govern

    gov = Governance(
        policy=Policy(allow=["read_file", "run_shell"], deny=["delete_all"],
                      require_approval=["run_shell"]),
        approver=CallbackApprover(lambda req: ask_a_human(req)),
    )
    tools = govern([read_file, run_shell, delete_all], gov, risk={
        "run_shell": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
        "delete_all": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
        "read_file": {"verb": "READ", "risk_tier": "low", "side_effect": False},
    })
    # pass `tools` to create_react_agent / AgentExecutor / bind_tools, governed and unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from infy.governance import Governance

# Applied to any tool not named in `risk`: conservatively high-risk and side-effecting, so
# governance fails safe on an unprofiled tool rather than waving it through.
_DEFAULT_RISK: dict[str, Any] = {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True}


@dataclass
class _ToolMeta:
    """The tool metadata infy governance reads: name for the policy, verb/risk_tier/side_effect
    for the risk engine."""

    name: str
    verb: str | None = None
    risk_tier: str | None = None
    side_effect: bool = False


def _govern_one(tool: Any, governance: Governance, profile: dict[str, Any]) -> Any:
    meta = _ToolMeta(
        name=tool.name,
        verb=profile.get("verb"),
        risk_tier=profile.get("risk_tier"),
        side_effect=profile.get("side_effect", True),
    )

    def governed(**kwargs: Any) -> Any:
        decision = governance.before_tool(cast(Any, meta), dict(kwargs))
        if not decision.allowed:
            # The agent's model sees this as the tool result and must adapt, it does not execute.
            return f"[Infyrence governance] {decision.message}"
        result = tool.invoke(dict(kwargs))
        governance.after_tool(cast(Any, meta), result)
        return result

    return StructuredTool.from_function(
        func=governed,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        infer_schema=tool.args_schema is None,
    )


def govern(
    tools: list[Any],
    governance: Governance,
    *,
    risk: dict[str, dict[str, Any]] | None = None,
    default: dict[str, Any] | None = None,
) -> list[Any]:
    """Wrap each LangChain tool so its execution passes through infy governance.

    ``risk`` maps a tool name to ``{"verb": ..., "risk_tier": ..., "side_effect": ...}``; tools not
    listed use ``default`` (conservatively high-risk). Returns drop-in replacement tools.
    """
    profiles = risk or {}
    base = default or _DEFAULT_RISK
    return [_govern_one(t, governance, profiles.get(t.name, base)) for t in tools]
