# infy governance

An optional, **in-process** control plane for the infy agent loop: deny-by-default policy
enforcement, risk tiering, human approval gates, and a tamper-evident audit trail — evaluated
at the existing tool/model chokepoints as a plain function call (microseconds), never an
out-of-band service on the hot path.

It is the answer to the question every team deploying action-taking agents asks: *"how do I
make this safe to give real authority — and prove what it did?"*

```python
from infy import create_agent
from infy.governance import Governance, Policy, CallbackApprover

gov = Governance(
    policy=Policy(
        deny=["delete_database"],          # never, regardless of approval
        require_approval=["deploy"],       # allowed only with a human yes
    ),
    approver=CallbackApprover(my_slack_prompt),   # how a human decides
    principal="agent://acme/assistant",
)

agent = create_agent(model, tools=[search, deploy, delete_database], governance=gov)
agent("ship the release")

for event in gov.audit.events:    # the receipt
    ...
assert gov.audit.verify()         # tamper-evident
```

When `governance` is omitted the loop is byte-for-byte unchanged; nothing in `infy` core
imports this module.

## Why in-process

A policy-decision **microservice** with a network hop before every tool call would erase
infy's reason to exist (cold start, RAM density, latency). So enforcement runs **in-process**;
only the heavy parts (audit persistence, approval queues) touch I/O, and only off the hot path.
The `PolicyEngine` protocol is the seam — the pure-Python evaluator here can be swapped for an
embedded Cedar engine (Rust, in `infy_core`) later without changing a single call site.

**Measured overhead.** The isolated policy + risk + audit decision is ~20–35 µs; end-to-end in a
real agent loop it adds **~50 µs per tool call** (the `governance_bench` integration benchmark) —
about **0.003%** of the ~1–2 s LLM call it guards. Governance is cheap enough to leave on for every
agent. Run `python examples/governance_demo.py`; see [BENCHMARKS.md](../../BENCHMARKS.md).

## Primitives

| Component | Role |
| --- | --- |
| `PolicyEngine` / `PythonPolicyEngine` | deny-by-default, forbid-overrides-permit, **fail-closed** evaluation (Cedar semantics) |
| `Policy` | ergonomic surface (`allow` / `deny` / `require_approval` / `approve_when`) that compiles to rules |
| `RiskEngine` | tier per call from the tool's static profile; an **unprofiled side-effecting tool is HIGH** |
| `Approver` (`CallbackApprover`, `AutoApprove`, `DenyAll`) | the human-in-the-loop decision; default `DenyAll` (fail-closed) |
| `AuditLog` | append-only, SHA-256 **hash-chained**, `verify()`-able, optional JSONL mirror |
| `Governance` | the facade exposing the four hooks the agent loop calls |

Tools carry optional governance metadata:

```python
@tool(risk_tier="critical", verb="DB", side_effect=True)
def delete_database(name: str) -> str: ...
```

## Hook points

`before_model` / `after_model` (audit) and `before_tool` / `after_tool` (the critical
security boundary). `before_tool` runs: risk tier → policy evaluate → if `require_approval`,
call the approver → final allow/deny → write one audit event. A denied call returns a
`ToolMessage(status="error")` the loop already handles, so the run stays recoverable.

## Security posture

The enforcement path is **fail-closed end to end**: any error in risk scoring, policy
evaluation, or the approver yields a *deny* (and is still audited), at both the `Governance`
facade and the agent-loop seam. Risk **gates** the decision — actions at or above `escalate_at`
(default HIGH), including unprofiled side-effecting tools, require approval rather than
auto-running. The audit log is **thread-safe** and supports an optional **HMAC key** (held
outside the agent) so the chain is not purely self-anchored. Hardened against an adversarial
security review (fail-open paths, audit integrity, approval bypass) with regression tests.

**Honest gaps (deferred, not silently missing):**

- The default SHA-256 chain is tamper-*evident*, not tamper-*proof* against a process that can
  regenerate it. Use `AuditLog(key=...)`, and for production anchor the head hash to an external
  notary / WORM sink.
- No segregation-of-duties / approver-identity capture yet — a maker can approve their own
  action. Four-eyes is on the roadmap.
- The JSONL mirror is an append-only export; it is not yet read back and reconciled on startup.
- `before_model` / `after_model` are audit-only (no semantic firewall); the deferred P1 work is
  taint/provenance and egress DLP, not signature scanning.

## Status & roadmap

This is the **MVP**: in-process enforcement, the approval gate, and tamper-evident audit — the
first sellable slice ("safe to give your agent real authority, with a receipt"). Deliberately
**not** built yet (defer until a design partner pulls): embedded Cedar in `infy_core`, the
YAML→Cedar DSL, taint/provenance injection defense, egress DLP, plan-level authorization,
tool-list filtering, durable approvals on the graph path, capability attenuation across
sub-agents, OpenFGA, and compliance packs. The `PolicyEngine`/`Approver` protocols and the
hook seam are the stable interfaces those features attach to.
