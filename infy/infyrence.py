"""Optional Infyrence integration for infy.

Wire a governed infy agent to your Infyrence account with one API key:

    from infy import create_agent
    from infy.governance import Policy
    from infy.infyrence import connect

    model, gov = connect(
        agent="sre-responder",
        policy=Policy(deny=["delete_database"], require_approval=["deploy"]),
        model="gpt-4o",
    )
    agent = create_agent(model, tools, governance=gov)
    agent("investigate the checkout outage")

`connect` routes the model through the Infyrence gateway (200+ models, one key) and
attaches an audit sink that streams every governed decision to your dashboard. The
governance streaming uses only the standard library, so it adds no dependencies; live
model calls reuse infy's OpenAI-compatible provider (pip install infy[openai]).

Configuration is read from arguments first, then the environment:
  INFYRENCE_API_KEY    your Infyrence API key (the same key the gateway uses)
  INFYRENCE_BASE_URL   gateway base url (default https://api.infyrence.com/v1)
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import threading
import urllib.request
from typing import Any

from infy.governance import (
    Approver,
    AuditEvent,
    AuditLog,
    Governance,
    PolicyEngine,
    RiskTier,
)

DEFAULT_BASE_URL = "https://api.infyrence.com/v1"
_EVENTS_PATH = "/governance/events"


def _preview(args: dict[str, Any] | None) -> str:
    """A short, human-readable target from a tool call's arguments."""
    if not args:
        return ""
    single = len(args) == 1
    parts: list[str] = []
    for key, value in args.items():
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(text if single else f"{key}={text}")
    return ", ".join(parts)[:160]


class InfyrenceAudit(AuditLog):
    """An infy AuditLog that streams each governed tool decision to Infyrence.

    It overrides record(): super().record() runs first, so the tamper-evident hash chain
    is computed exactly as it would be offline, then the decision is shipped to the
    Infyrence governance API on a background thread. Network failures are swallowed; the
    sink never blocks or breaks the agent.
    """

    def __init__(
        self,
        *,
        api_key: str,
        agent: str,
        model: str = "",
        base_url: str | None = None,
        key: bytes | None = None,
        path: str | None = None,
        batch_size: int = 20,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(path=path, key=key)
        self._api_key = api_key
        self._agent = agent
        self._model = model
        self._endpoint = (base_url or DEFAULT_BASE_URL).rstrip("/") + _EVENTS_PATH
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="infyrence-audit", daemon=True)
        self._worker.start()
        atexit.register(self.flush)

    def record(
        self,
        *,
        actor: str,
        action: str,
        resource: str,
        decision: str,
        risk_tier: str = "",
        reasons: tuple[str, ...] = (),
        args: dict[str, Any] | None = None,
    ) -> AuditEvent:
        # Record locally FIRST so the hash chain is identical to an offline run.
        event = super().record(
            actor=actor,
            action=action,
            resource=resource,
            decision=decision,
            risk_tier=risk_tier,
            reasons=reasons,
            args=args,
        )
        # Stream tool decisions only; model bookkeeping events stay in the local chain.
        if action == "invoke_tool":
            self._queue.put(
                {
                    "action": event.resource,
                    "target": _preview(args),
                    "risk": event.risk_tier,
                    "decision": event.decision,
                    "reason": ", ".join(event.reasons),
                    "principal": event.actor,
                    "seq": event.seq,
                    "hash": event.hash,
                    "prev_hash": event.prev_hash,
                }
            )
        return event

    def flush(self) -> None:
        """Block until queued events have been sent. Runs automatically at process exit."""
        with contextlib.suppress(Exception):
            self._queue.join()

    # ── internals ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            try:
                while len(batch) < self._batch_size:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            try:
                self._send(batch)
            except Exception:
                pass  # best-effort: a sink must never break the agent
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _send(self, batch: list[dict[str, Any]]) -> None:
        payload = {"agent": self._agent, "model": self._model, "events": batch}
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            response.read()


def infyrence_model(
    model: str = "gpt-4o",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """An OpenAI-compatible chat model pointed at the Infyrence gateway.

    Needs the OpenAI client: pip install infy[openai].
    """
    key = api_key or os.environ.get("INFYRENCE_API_KEY")
    if not key:
        raise RuntimeError("No Infyrence API key. Set INFYRENCE_API_KEY or pass api_key=...")
    base = base_url or os.environ.get("INFYRENCE_BASE_URL") or DEFAULT_BASE_URL
    from infy.providers.openai import OpenAIChat

    return OpenAIChat(model, api_key=key, base_url=base)


def connect(
    agent: str,
    *,
    policy: PolicyEngine,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    approver: Approver | None = None,
    principal: str | None = None,
    escalate_at: RiskTier | None = None,
    audit_key: bytes | None = None,
) -> tuple[Any, Governance]:
    """Wire an agent to Infyrence with one key.

    Returns (model, governance): a gateway-backed model and a Governance whose audit log
    streams to your Infyrence dashboard. Pass both to create_agent.
    """
    key = api_key or os.environ.get("INFYRENCE_API_KEY")
    if not key:
        raise RuntimeError("No Infyrence API key. Set INFYRENCE_API_KEY or pass api_key=...")
    base = base_url or os.environ.get("INFYRENCE_BASE_URL") or DEFAULT_BASE_URL

    audit = InfyrenceAudit(api_key=key, agent=agent, model=model, base_url=base, key=audit_key)
    governance_kwargs: dict[str, Any] = {
        "policy": policy,
        "audit": audit,
        "principal": principal or f"agent://infyrence/{agent}",
        "escalate_at": escalate_at,
    }
    if approver is not None:
        governance_kwargs["approver"] = approver
    governance = Governance(**governance_kwargs)

    model_obj = infyrence_model(model, api_key=key, base_url=base)
    return model_obj, governance


__all__ = ["InfyrenceAudit", "connect", "infyrence_model", "DEFAULT_BASE_URL"]
