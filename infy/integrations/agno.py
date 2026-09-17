"""Govern an Agno agent with infy, without changing the agent.

Agno runs every tool through ``FunctionCall.execute``, which nests the entrypoint inside the chain
of callables on ``Function.tool_hooks``. A hook that returns without calling the next link
short-circuits the chain and the entrypoint never runs, which is exactly the shape of an
authorization gate. This adapter builds that hook: a denied or unapproved action never executes,
the model receives the block reason as the tool output and must adapt, and every decision, allowed
or denied, is written to the tamper-evident audit chain.

Usage::

    from agno.agent import Agent
    from infy.governance import Governance, Policy, CallbackApprover
    from infy.integrations.agno import govern_hook

    gov = Governance(
        policy=Policy(allow=["read_file", "run_shell"], deny=["delete_all"],
                      require_approval=["run_shell"]),
        approver=CallbackApprover(ask_a_human),
    )
    agent = Agent(
        model=model,
        tools=[read_file, run_shell, delete_all],
        tool_hooks=[govern_hook(gov, risk={
            "run_shell": {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True},
            "delete_all": {"verb": "DB", "risk_tier": "critical", "side_effect": True},
            "read_file": {"verb": "READ", "risk_tier": "low", "side_effect": False},
        })],
    )

Use ``agovern_hook`` with ``arun`` and ``aprint_response``. Agno hands the async chain an awaitable
next link, and logs and skips a coroutine hook on the sync path, so the two are not interchangeable.

**Scope.** An agent-level hook reaches every tool Agno executes in this process: plain callables,
``Function`` objects, and every function of a ``Toolkit``. It does not reach a builtin tool passed
as a dict, because that one is executed by the model provider and never enters this process, so
there is no local chokepoint to gate. Governance is in-process by design; what the provider runs on
its own side is outside it.

**No per-tool ``govern()``.** The smolagents and LangChain adapters expose one; this one does not,
on purpose. Agno assigns ``agent.tool_hooks`` onto each function while it builds the tool list,
overwriting hooks already set on that function, so a per-tool wrapper here could be switched off by
an unrelated agent setting. A control plane with a silent-disable path is worse than no control
plane, so per-tool profiles are expressed through ``risk=`` on the agent-level hook instead.

**Durable approval.** Inline approvers (``CallbackApprover``, ``DenyAll``) work fully. A
``DurableApprover`` still fails closed, the tool does not run, but it cannot suspend the run:
``FunctionCall.execute`` catches every exception except its own cancellation, so the
``ApprovalRequired`` that infy raises to suspend is converted into a failed tool call. Use
``DurableAgent`` on an infy agent when you need suspend-and-resume.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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


def _meta_for(name: str, profiles: dict[str, dict[str, Any]], base: dict[str, Any]) -> _ToolMeta:
    """Build the metadata for one tool from its risk profile, falling back to ``base``.

    Agno identifies a tool by name at the hook boundary, which is the same key the policy uses.
    """
    profile = profiles.get(name, base)
    return _ToolMeta(
        name=name,
        verb=profile.get("verb"),
        risk_tier=profile.get("risk_tier"),
        side_effect=profile.get("side_effect", True),
    )


def govern_hook(
    governance: Governance,
    *,
    risk: dict[str, dict[str, Any]] | None = None,
    default: dict[str, Any] | None = None,
) -> Callable[[str, Callable[..., Any], dict[str, Any]], Any]:
    """Build a sync Agno tool hook that passes every tool call through infy governance.

    ``risk`` maps a tool name to ``{"verb": ..., "risk_tier": ..., "side_effect": ...}``. Tools not
    listed use ``default`` (conservatively high-risk and side-effecting). Pass the result to
    ``Agent(tool_hooks=[...])``; use :func:`agovern_hook` for the async agent methods.
    """
    profiles = risk or {}
    base = default or _DEFAULT_RISK

    def hook(name: str, func: Callable[..., Any], args: dict[str, Any]) -> Any:
        # Parameter names matter: Agno inspects the hook's signature and binds `name`, `func` and
        # `args` by name, passing only what the hook actually declares.
        meta = _meta_for(name, profiles, base)
        decision = governance.before_tool(cast(Any, meta), dict(args))
        if not decision.allowed:
            # Returning without calling `func` short-circuits the chain: the entrypoint never runs.
            # The agent's model sees this as the tool result and must adapt.
            return f"[Infyrence governance] {decision.message}"
        result = func(**args)
        governance.after_tool(cast(Any, meta), result)
        return result

    return hook


def agovern_hook(
    governance: Governance,
    *,
    risk: dict[str, dict[str, Any]] | None = None,
    default: dict[str, Any] | None = None,
) -> Callable[[str, Callable[..., Awaitable[Any]], dict[str, Any]], Awaitable[Any]]:
    """Async counterpart of :func:`govern_hook`, for ``Agent.arun`` and ``aprint_response``.

    Agno's async chain always hands the hook an awaitable next link, so this awaits it. The policy,
    risk, and audit decisions themselves are in-process and synchronous (microseconds), so only the
    tool call is awaited.
    """
    profiles = risk or {}
    base = default or _DEFAULT_RISK

    async def hook(name: str, func: Callable[..., Awaitable[Any]], args: dict[str, Any]) -> Any:
        meta = _meta_for(name, profiles, base)
        decision = governance.before_tool(cast(Any, meta), dict(args))
        if not decision.allowed:
            return f"[Infyrence governance] {decision.message}"
        result = await func(**args)
        governance.after_tool(cast(Any, meta), result)
        return result

    return hook
