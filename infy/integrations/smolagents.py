"""Govern a smolagents agent with infy, without changing the agent.

smolagents invokes a tool through ``Tool.__call__`` -> ``Tool.forward``. This adapter wraps each
tool so that ``forward`` first asks infy governance whether the action is allowed. A denied or
unapproved action never runs; the model receives the block reason as the tool output. Every
decision, allowed or denied, is written to the tamper-evident audit chain.

Usage::

    from smolagents import ToolCallingAgent, tool
    from infy.governance import Governance, Policy, CallbackApprover
    from infy.integrations.smolagents import govern

    gov = Governance(
        policy=Policy(allow=["read_file", "run_shell"], deny=["delete_all"],
                      require_approval=["run_shell"]),
        approver=CallbackApprover(lambda req: ask_a_human(req)),
    )
    tools = govern([read_file, run_shell, delete_all], gov, risk={
        "run_shell": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
        "delete_all": {"verb": "EXECUTE", "risk_tier": "critical", "side_effect": True},
        "read_file": {"verb": "READ", "risk_tier": "low", "side_effect": False},
    })
    agent = ToolCallingAgent(tools=tools, model=model)  # governed, unchanged otherwise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from smolagents import Tool

if TYPE_CHECKING:
    from infy.governance import Governance

# Applied to any tool not named in `risk`. Conservative on purpose: an unprofiled tool is treated
# as a high-risk, side-effecting action, so governance fails safe rather than waving it through.
_DEFAULT_RISK: dict[str, Any] = {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True}


@dataclass
class _ToolMeta:
    """The tool metadata infy governance reads: the name for the policy, and verb/risk_tier/
    side_effect for the risk engine. Structurally compatible with what ``before_tool`` needs."""

    name: str
    verb: str | None = None
    risk_tier: str | None = None
    side_effect: bool = False


class GovernedTool(Tool):  # type: ignore[misc]  # smolagents.Tool is an untyped base
    """A smolagents Tool whose execution is gated by infy governance."""

    def __init__(
        self,
        inner: Any,
        governance: Governance,
        *,
        verb: str | None = None,
        risk_tier: str | None = None,
        side_effect: bool = True,
    ) -> None:
        self._inner = inner
        self._gov = governance
        self._meta = _ToolMeta(
            name=inner.name, verb=verb, risk_tier=risk_tier, side_effect=side_effect
        )
        # Mirror the wrapped tool's interface so smolagents treats us identically.
        self.name = inner.name
        self.description = inner.description
        self.inputs = inner.inputs
        self.output_type = inner.output_type
        super().__init__()

    def validate_arguments(self, *args: Any, **kwargs: Any) -> None:
        # smolagents validates that forward's parameter names match `inputs`. We intentionally
        # proxy through *args/**kwargs, and our name/description/inputs/output_type were copied
        # from an already-validated inner tool, so we skip only this re-validation.
        return None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        decision = self._gov.before_tool(cast(Any, self._meta), dict(kwargs))
        if not decision.allowed:
            # The agent's model sees this as the tool result and must adapt, it does not execute.
            return f"[Infyrence governance] {decision.message}"
        result = self._inner.forward(*args, **kwargs)
        self._gov.after_tool(cast(Any, self._meta), result)
        return result


def govern(
    tools: list[Any],
    governance: Governance,
    *,
    risk: dict[str, dict[str, Any]] | None = None,
    default: dict[str, Any] | None = None,
) -> list[Any]:
    """Wrap each smolagents tool so its execution passes through infy governance.

    ``risk`` maps a tool name to ``{"verb": ..., "risk_tier": ..., "side_effect": ...}``. Tools not
    listed use ``default`` (conservatively high-risk and side-effecting). Returns new tools that are
    drop-in replacements for the originals.
    """
    risk = risk or {}
    base = default or _DEFAULT_RISK
    governed: list[Any] = []
    for tool in tools:
        meta = risk.get(tool.name, base)
        governed.append(
            GovernedTool(
                tool,
                governance,
                verb=meta.get("verb"),
                risk_tier=meta.get("risk_tier"),
                side_effect=meta.get("side_effect", True),
            )
        )
    return governed
