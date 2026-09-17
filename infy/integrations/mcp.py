"""Govern an MCP client session with infy, without changing the client.

Every tool a Model Context Protocol client invokes goes through one method, ``ClientSession
.call_tool``. :class:`GovernedSession` wraps a live session and gates exactly that: a denied or
unapproved call never reaches the server, the model receives a protocol-correct error result and
adapts, and every decision lands in the tamper-evident audit chain.

Usage::

    from mcp import ClientSession
    from infy.governance import Governance, Policy, CallbackApprover
    from infy.integrations.mcp import GovernedSession

    gov = Governance(
        policy=Policy(allow=["files/read_file"], deny=["payments/transfer"]),
        approver=CallbackApprover(ask_a_human),
    )

    async with ClientSession(read, write) as raw:
        await raw.initialize()
        session = GovernedSession(raw, gov, server="files")
        await session.list_tools()                       # definitions recorded
        await session.call_tool("read_file", {"path": "a.txt"})

Why a wrapper and not a subclass: ``ClientSession.__init__`` takes sixteen parameters and the class
has no stable base, so subclassing would couple infy to that surface and break on SDK churn. This
delegates everything it does not gate through ``__getattr__``, so methods added by later SDK
versions keep working untouched and only the two that matter are intercepted.

**Server-qualified names.** When ``server`` is given, policy and audit see ``"<server>/<tool>"``.
Two MCP servers may expose the same tool name, and an unqualified policy written for a trusted
server would silently also permit an untrusted server's namesake. Qualify in any multi-server
deployment; policies are written against the qualified name.

**Annotations are hints, so they only escalate.** MCP tools may carry ``ToolAnnotations``
(``read_only_hint``, ``destructive_hint``, ...). The SDK is explicit that these are unverified and
that "clients should never make tool use decisions based on ToolAnnotations received from untrusted
servers". So by default a hint may only raise a tool's risk above the conservative baseline, never
lower it: ``destructive_hint`` escalates to CRITICAL, and ``read_only_hint`` from an untrusted
server is ignored. A server can therefore make you more careful but never less. Pass
``trust_annotations=True`` for a server the operator has actually vetted, and ``read_only_hint``
is honoured as LOW. An explicit ``risk=`` entry is operator-supplied and always wins over both.

**Definition drift is recorded.** Tool poisoning is a rug pull: a server advertises a benign
description, gets approved, then changes it. ``list_tools`` digests each tool's name, description,
input schema and annotations, keeps a manifest digest of the set, and on a later listing records
exactly which tools changed. Because the audit chain is hash-linked, a change is not merely
detectable but provable after the fact. This is evidence, not prevention: pair it with policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from mcp import types

if TYPE_CHECKING:
    from infy.governance import Governance

# Applied to any tool not named in `risk` and not covered by a trusted annotation. Conservative on
# purpose: an unprofiled remote tool is treated as high-risk and side-effecting, so an MCP server
# that says nothing about a tool gets no benefit of the doubt.
_DEFAULT_RISK: dict[str, Any] = {"verb": "EXECUTE", "risk_tier": "high", "side_effect": True}


@dataclass
class _ToolMeta:
    """The tool metadata infy governance reads: the name for the policy, and verb/risk_tier/
    side_effect for the risk engine. Structurally compatible with what ``before_tool`` needs."""

    name: str
    verb: str | None = None
    risk_tier: str | None = None
    side_effect: bool = False


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _tool_digest(tool: types.Tool) -> str:
    """Content digest of everything about a tool that could change its meaning to the model.

    Description and input schema are included because both steer the model, and annotations
    because a flip there changes how this adapter scores the tool.
    """
    annotations = tool.annotations.model_dump(exclude_none=True) if tool.annotations else None
    return _digest(
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "annotations": annotations,
        }
    )


class GovernedSession:
    """An MCP ``ClientSession`` whose tool calls pass through infy governance.

    Everything not gated is delegated to the wrapped session unchanged.
    """

    def __init__(
        self,
        session: Any,
        governance: Governance,
        *,
        server: str | None = None,
        risk: dict[str, dict[str, Any]] | None = None,
        default: dict[str, Any] | None = None,
        trust_annotations: bool = False,
    ) -> None:
        self._session = session
        self._gov = governance
        self._server = server
        self._risk = risk or {}
        self._default = default or _DEFAULT_RISK
        self._trust_annotations = trust_annotations
        self._annotations: dict[str, types.ToolAnnotations] = {}
        self._definitions: dict[str, str] = {}
        self._manifest: str | None = None

    # -- delegation --------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so the gated methods below always win. The guard
        # keeps a lookup before __init__ has run (unpickling, copy) from recursing forever.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_session"], name)

    # -- naming and risk ---------------------------------------------------

    def _qualified(self, tool: str) -> str:
        return f"{self._server}/{tool}" if self._server else tool

    def _meta_for(self, tool: str) -> _ToolMeta:
        """Resolve a tool's risk profile: operator profile, else annotations, else the default.

        An explicit ``risk=`` entry is operator-supplied and authoritative. Otherwise the baseline
        is the conservative default, and a server's annotations may only move it up unless the
        operator has declared this server trusted.
        """
        name = self._qualified(tool)
        explicit = self._risk.get(name) or self._risk.get(tool)
        if explicit is not None:
            profile = dict(explicit)
        else:
            profile = dict(self._default)
            hints = self._annotations.get(tool)
            if hints is not None:
                if hints.destructive_hint:
                    profile["risk_tier"] = "critical"  # escalation is always honoured
                elif self._trust_annotations and hints.read_only_hint:
                    profile = {"verb": "READ", "risk_tier": "low", "side_effect": False}
        return _ToolMeta(
            name=name,
            verb=profile.get("verb"),
            risk_tier=profile.get("risk_tier"),
            side_effect=profile.get("side_effect", True),
        )

    def _blocked(self, message: str) -> types.CallToolResult:
        """A denial the model can read: a protocol-correct error result, not a bare string."""
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"[Infyrence governance] {message}")],
            is_error=True,
        )

    # -- gated surface -----------------------------------------------------

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Authorize, then invoke. A denied call never reaches the server.

        A continuation of an input-required call (one carrying ``request_state``) is a fresh
        invocation of the tool and is authorized again rather than riding on the first decision.
        """
        meta = self._meta_for(name)
        supplied = dict(arguments or {})
        if kwargs.get("request_state") is not None:
            # Audit continuations distinctly: the same tool, a second admission, new inputs.
            supplied = {**supplied, "_continuation": True}
        if name not in self._definitions:
            # Calling a tool that was never listed is legal but worth seeing in the trail.
            self._record("call_unlisted_tool", meta.name, "noted")
        decision = self._gov.before_tool(cast(Any, meta), supplied)
        if not decision.allowed:
            return self._blocked(decision.message)
        result = await self._session.call_tool(name, arguments, *args, **kwargs)
        self._gov.after_tool(cast(Any, meta), result)
        return result

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        """List tools, harvest annotations for risk, and record any definition drift."""
        result = await self._session.list_tools(*args, **kwargs)
        tools = list(getattr(result, "tools", []) or [])

        digests = {t.name: _tool_digest(t) for t in tools}
        for tool in tools:
            if tool.annotations is not None:
                self._annotations[tool.name] = tool.annotations

        manifest = _digest(sorted(digests.items()))
        if self._manifest is None:
            self._record(
                "list_tools",
                self._server or "mcp",
                "recorded",
                reasons=(f"tools={len(digests)}", f"manifest={manifest[:16]}"),
            )
        elif manifest != self._manifest:
            for name, digest in digests.items():
                previous = self._definitions.get(name)
                if previous is None:
                    self._record("tool_appeared", self._qualified(name), "changed")
                elif previous != digest:
                    # The description, schema, or annotations the model trusts have been rewritten.
                    self._record(
                        "tool_definition_changed",
                        self._qualified(name),
                        "changed",
                        reasons=(f"was={previous[:16]}", f"now={digest[:16]}"),
                    )
            for name in self._definitions.keys() - digests.keys():
                self._record("tool_withdrawn", self._qualified(name), "changed")

        self._definitions = digests
        self._manifest = manifest
        return result

    # -- internals ---------------------------------------------------------

    def _record(
        self, action: str, resource: str, decision: str, reasons: tuple[str, ...] = ()
    ) -> None:
        self._gov.audit.record(
            actor=self._gov.principal,
            action=action,
            resource=resource,
            decision=decision,
            reasons=reasons,
        )
