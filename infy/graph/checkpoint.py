"""Checkpoint system — save, restore, and time-travel through graph state.

LangGraph: 860 lines (base) + 704 lines (memory) + 253 lines (serde) = 1,817 lines.
infy: ~150 lines.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@dataclass
class Checkpoint:
    """A snapshot of graph state at a point in time.

    ``next_tasks`` records the execution frontier (the ``[node, arg]`` pairs that
    would run next). Persisting it is what lets an interrupted run resume from
    where it paused instead of restarting from START.
    """

    v: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=_now_iso)
    channel_values: dict[str, Any] = field(default_factory=dict)
    channel_versions: dict[str, int | float] = field(default_factory=dict)
    versions_seen: dict[str, dict[str, int | float]] = field(default_factory=dict)
    next_tasks: list[list[Any]] = field(default_factory=list)

    def copy(self) -> Checkpoint:
        return Checkpoint(
            v=self.v,
            id=self.id,
            ts=self.ts,
            channel_values=dict(self.channel_values),
            channel_versions=dict(self.channel_versions),
            versions_seen={k: dict(v) for k, v in self.versions_seen.items()},
            next_tasks=[list(t) for t in self.next_tasks],
        )


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""

    source: str = "input"  # input, loop, update
    step: int = 0
    run_id: str = ""


class CheckpointTuple(NamedTuple):
    """A checkpoint with its config and metadata."""

    config: dict[str, Any]
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    pending_writes: list[tuple[str, str, Any]] | None = None


# ---------------------------------------------------------------------------
# Base saver protocol
# ---------------------------------------------------------------------------


class BaseCheckpointSaver:
    """Protocol for checkpoint persistence.

    Implement this to store checkpoints in any backend.
    """

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Fetch a full checkpoint tuple."""
        raise NotImplementedError

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, int | float],
    ) -> dict[str, Any]:
        """Store a checkpoint. Returns updated config."""
        raise NotImplementedError

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes."""
        pass

    def list(
        self,
        config: dict[str, Any],
        limit: int = 10,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        return iter([])

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        pass

    def get_next_version(self, current: int | float | None) -> int:
        """Generate next version number."""
        if current is None:
            return 1
        if isinstance(current, int):
            return current + 1
        return int(current) + 1


# ---------------------------------------------------------------------------
# InMemorySaver — dict-based checkpoint storage
# ---------------------------------------------------------------------------


_StoredCheckpoint = tuple[dict[str, Any], Checkpoint, CheckpointMetadata]


class InMemorySaver(BaseCheckpointSaver):
    """In-memory checkpoint storage. For development and testing.

    Usage:
        from infy.graph import InMemorySaver, StateGraph, START, END

        saver = InMemorySaver()
        graph = StateGraph(State).compile(checkpointer=saver)

        result = graph.invoke(input, config={"configurable": {"thread_id": "1"}})
        state = graph.get_state(config)
    """

    def __init__(self) -> None:
        self.storage: dict[str, dict[str, list[_StoredCheckpoint]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.writes: dict[tuple[str, str, str], list[tuple[str, str, Any]]] = defaultdict(list)
        self.version_counter: int = 0

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        ns = config.get("configurable", {}).get("checkpoint_ns", "")

        checkpoints = self.storage[thread_id].get(ns, [])
        if not checkpoints:
            return None

        latest_config, checkpoint, metadata = checkpoints[-1]
        pending = self.writes.get((thread_id, ns, checkpoint.id), None)

        return CheckpointTuple(
            config=latest_config,
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending,
        )

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, int | float],
    ) -> dict[str, Any]:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        ns = config.get("configurable", {}).get("checkpoint_ns", "")

        self.storage[thread_id][ns].append(
            (
                config,
                checkpoint.copy(),
                metadata,
            )
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint.id,
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id", "")

        key = (thread_id, ns, checkpoint_id)
        for channel, value, _type in writes:
            self.writes[key].append((task_id, channel, value))

    def list(
        self,
        config: dict[str, Any],
        limit: int = 10,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config.get("configurable", {}).get("thread_id", "")
        ns = config.get("configurable", {}).get("checkpoint_ns", "")

        checkpoints = self.storage[thread_id].get(ns, [])
        for cfg, cp, meta in reversed(checkpoints[-limit:]):
            yield CheckpointTuple(config=cfg, checkpoint=cp, metadata=meta)

    def delete_thread(self, thread_id: str) -> None:
        self.storage.pop(thread_id, None)
        keys_to_delete = [k for k in self.writes if k[0] == thread_id]
        for k in keys_to_delete:
            del self.writes[k]

    def get_next_version(self, current: int | float | None) -> int:
        if current is None:
            return 1
        if isinstance(current, int):
            return current + 1
        return int(current) + 1
