"""Core types for the graph system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

START: str = "__start__"
END: str = "__end__"


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


@dataclass
class Send:
    """Dispatch a dynamic task to a named node with specific input.

    Usage in conditional edges:
        return [Send("process", item) for item in items]
    """

    node: str
    arg: Any = None


@dataclass
class Command:
    """Navigate or update state during graph execution.

    Can be returned from any node to:
    - goto: jump to a specific node
    - update: modify state values
    - resume: provide a value after interrupt
    """

    goto: str | None = None
    update: dict[str, Any] | None = None
    resume: Any = None
    PARENT: str = "__parent__"


@dataclass
class Interrupt:
    """Pauses graph execution for human input.

    Raised inside a node, caught by the execution engine.
    """

    value: Any = None
    id: str = ""


@dataclass(frozen=True)
class StateSnapshot:
    """Read-only view of graph state at a checkpoint."""

    values: dict[str, Any]
    next: list[str]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_config: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"StateSnapshot(values={self.values}, next={self.next})"


@dataclass
class NodeConfig:
    """Configuration for a graph node."""

    func: Any  # The node function
    retry: int = 0
    cache: bool = False
    timeout: float | None = None
