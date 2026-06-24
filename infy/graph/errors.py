"""Graph-specific errors."""

from __future__ import annotations

from typing import Any


class EmptyChannelError(Exception):
    """Raised when reading from an empty channel."""


class GraphInterrupt(Exception):  # noqa: N818 - a pause/resume signal, not an error
    """Internal interrupt for human-in-the-loop.

    Wraps one or more Interrupt values.
    """

    def __init__(self, interrupts: list[Any] | None = None):
        self.interrupts = interrupts or []
        super().__init__(f"Graph interrupted with {len(self.interrupts)} interrupt(s)")


class NodeInterrupt(GraphInterrupt):  # noqa: N818 - a pause/resume signal, not an error
    """Raised when a node calls interrupt()."""

    def __init__(self, value: Any = None):
        super().__init__(interrupts=[value])


class GraphRecursionError(RecursionError):
    """Raised when graph exceeds maximum recursion limit."""

    def __init__(self, limit: int, current: int):
        super().__init__(
            f"Graph recursion limit of {limit} reached after {current} steps. "
            "Increase the limit or add a termination condition."
        )
        self.limit = limit
        self.current = current


class InvalidUpdateError(Exception):
    """Raised when an invalid state update is attempted."""


class NodeError(Exception):
    """Wraps an error from a graph node."""

    def __init__(self, node: str, error: BaseException):
        self.node = node
        self.error = error
        super().__init__(f"Error in node '{node}': {error}")
