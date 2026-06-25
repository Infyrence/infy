"""Policy engine — deny-by-default, forbid-overrides-permit, fail-closed.

These are Cedar's semantics, implemented in pure Python so the governance layer works on a
zero-dependency install. The PolicyEngine protocol is the seam: an embedded Cedar engine
(Rust, in infy_core) can be dropped in behind it later without changing any call site.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from infy.governance.types import AuthRequest, Decision, Effect, Obligation


@dataclass(frozen=True)
class Rule:
    """A single permit/forbid rule. ``tools=None`` matches any tool; ``condition=None``
    matches unconditionally."""

    effect: Effect
    tools: tuple[str, ...] | None = None
    condition: Callable[[AuthRequest], bool] | None = None
    obligations: tuple[Obligation, ...] = ()
    name: str = ""

    def matches(self, request: AuthRequest) -> bool:
        if self.tools is not None and request.tool not in self.tools:
            return False
        return self.condition is None or self.condition(request)


@runtime_checkable
class PolicyEngine(Protocol):
    def evaluate(self, request: AuthRequest) -> Decision: ...


@dataclass
class PythonPolicyEngine:
    """Deny-by-default, forbid-overrides-permit, fail-closed evaluator."""

    rules: list[Rule] = field(default_factory=list)

    def evaluate(self, request: AuthRequest) -> Decision:
        try:
            forbids = [r for r in self.rules if r.effect is Effect.FORBID and r.matches(request)]
            if forbids:
                return Decision(
                    Effect.FORBID, reasons=tuple(f"forbid:{r.name or '?'}" for r in forbids)
                )
            permits = [r for r in self.rules if r.effect is Effect.PERMIT and r.matches(request)]
            if permits:
                obligations = tuple(
                    sorted({o for r in permits for o in r.obligations}, key=lambda o: o.value)
                )
                return Decision(
                    Effect.PERMIT,
                    obligations=obligations,
                    reasons=tuple(f"permit:{r.name or '?'}" for r in permits),
                )
            return Decision(Effect.FORBID, reasons=("default-deny",))
        except Exception as exc:  # any evaluation error fails CLOSED, never open
            return Decision(Effect.FORBID, reasons=(f"fail-closed:{type(exc).__name__}",))


@dataclass
class Policy:
    """Ergonomic policy surface that compiles to rules. Satisfies PolicyEngine.

    - ``deny``: tools always forbidden (forbid overrides everything).
    - ``allow``: if set, ONLY these tools are permitted (otherwise permit-all-minus-deny).
    - ``require_approval``: permitted but gated on human approval (by tool name).
    - ``approve_when``: gate on a predicate over the request (e.g. risk tier).
    """

    allow: Sequence[str] | None = None
    deny: Sequence[str] | None = None
    require_approval: Sequence[str] | None = None
    approve_when: Callable[[AuthRequest], bool] | None = None
    _engine: PythonPolicyEngine | None = field(default=None, init=False, repr=False)

    def _build(self) -> PythonPolicyEngine:
        rules: list[Rule] = []
        allow_set = set(self.allow) if self.allow is not None else None
        if self.deny:
            rules.append(Rule(Effect.FORBID, tools=tuple(self.deny), name="denylist"))
        if self.require_approval:
            # In allowlist mode, approval may only gate tools that are ALSO allowed — it must
            # never widen the permit set (default-deny still applies to non-allowlisted tools).
            names = (
                tuple(self.require_approval)
                if allow_set is None
                else tuple(t for t in self.require_approval if t in allow_set)
            )
            if names:
                rules.append(
                    Rule(
                        Effect.PERMIT,
                        tools=names,
                        obligations=(Obligation.REQUIRE_APPROVAL,),
                        name="approval-list",
                    )
                )
        if self.approve_when is not None:
            base = self.approve_when
            if allow_set is None:
                cond: Callable[[AuthRequest], bool] = base
            else:
                allowed = allow_set

                def _gated(request: AuthRequest) -> bool:
                    return request.tool in allowed and base(request)

                cond = _gated
            rules.append(
                Rule(
                    Effect.PERMIT,
                    condition=cond,
                    obligations=(Obligation.REQUIRE_APPROVAL,),
                    name="approval-when",
                )
            )
        if self.allow is not None:
            rules.append(Rule(Effect.PERMIT, tools=tuple(self.allow), name="allowlist"))
        else:
            rules.append(Rule(Effect.PERMIT, name="permit-all"))
        return PythonPolicyEngine(rules)

    def evaluate(self, request: AuthRequest) -> Decision:
        if self._engine is None:
            self._engine = self._build()
        return self._engine.evaluate(request)
