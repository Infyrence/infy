"""Channels — typed communication pipes between graph nodes.

3 channel types (LangGraph has 10, we keep the 3 that matter):
- LastValue: single value per key (plain state fields)
- BinOp: reducer accumulation (Annotated[type, reducer])
- Topic: list accumulation (message history)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar, cast

from infy.graph.errors import EmptyChannelError, InvalidUpdateError

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Base channel
# ---------------------------------------------------------------------------


class BaseChannel(ABC, Generic[T]):
    """Abstract base for all channels."""

    __slots__ = ("key",)

    def __init__(self, key: str = "") -> None:
        self.key = key

    @abstractmethod
    def get(self) -> T:
        """Read current value. Raises EmptyChannelError if empty."""

    @abstractmethod
    def update(self, values: Sequence[Any]) -> bool:
        """Apply batch of updates. Returns True if channel changed."""

    @abstractmethod
    def checkpoint(self) -> Any:
        """Serialize channel state for checkpointing."""

    @abstractmethod
    def from_checkpoint(self, data: Any) -> None:
        """Restore channel from checkpoint data."""

    @abstractmethod
    def copy(self) -> BaseChannel[T]:
        """Create a deep copy of this channel."""

    def is_available(self) -> bool:
        try:
            self.get()
            return True
        except EmptyChannelError:
            return False


# ---------------------------------------------------------------------------
# LastValue — single value per key
# ---------------------------------------------------------------------------


class LastValue(BaseChannel[T]):
    """Stores the last value written. Only one write per step allowed.

    Used for plain state fields:
        class State(TypedDict):
            name: str  # → LastValue(str)
    """

    __slots__ = ("value", "has_value")
    value: Any
    has_value: bool

    def __init__(self, key: str = "") -> None:
        super().__init__(key)
        self.value = None
        self.has_value = False

    def get(self) -> T:
        if not self.has_value:
            raise EmptyChannelError(f"Channel '{self.key}' is empty")
        return cast("T", self.value)

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        if len(values) > 1:
            raise InvalidUpdateError(
                f"Channel '{self.key}' can receive only one value per step. "
                "Use an annotated channel type for multiple values."
            )
        self.value = values[-1]
        self.has_value = True
        return True

    def checkpoint(self) -> Any:
        if not self.has_value:
            return None
        return self.value

    def from_checkpoint(self, data: Any) -> None:
        if data is not None:
            self.value = data
            self.has_value = True

    def copy(self) -> LastValue[T]:
        ch = LastValue[T](self.key)
        ch.value = self.value
        ch.has_value = self.has_value
        return ch

    def __repr__(self) -> str:
        if self.has_value:
            return f"LastValue({self.key}={self.value!r})"
        return f"LastValue({self.key}=EMPTY)"


# ---------------------------------------------------------------------------
# BinOp — reducer accumulation
# ---------------------------------------------------------------------------


class BinOp(BaseChannel[T]):
    """Accumulates values using a binary operator (reducer).

    Used for annotated state fields:
        class State(TypedDict):
            count: Annotated[int, operator.add]  # → BinOp(int, add)
            messages: Annotated[list, add_messages]  # → BinOp(list, add_messages)
    """

    __slots__ = ("value", "operator", "has_value")
    value: Any
    operator: Callable[..., Any]
    has_value: bool

    def __init__(self, operator: Callable[..., Any], key: str = "") -> None:
        super().__init__(key)
        self.operator = operator
        self.value = None
        self.has_value = False

    def get(self) -> T:
        if not self.has_value:
            raise EmptyChannelError(f"Channel '{self.key}' is empty")
        return cast("T", self.value)

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        for value in values:
            if not self.has_value:
                self.value = value
                self.has_value = True
            else:
                self.value = self.operator(self.value, value)
        return True

    def checkpoint(self) -> Any:
        if not self.has_value:
            return None
        return self.value

    def from_checkpoint(self, data: Any) -> None:
        if data is not None:
            self.value = data
            self.has_value = True

    def copy(self) -> BinOp[T]:
        ch = BinOp[T](self.operator, self.key)
        ch.value = self.value
        ch.has_value = self.has_value
        return ch

    def __repr__(self) -> str:
        op_name = getattr(self.operator, "__name__", "?")
        if self.has_value:
            return f"BinOp({self.key}, {op_name}, value={self.value!r})"
        return f"BinOp({self.key}, {op_name}, EMPTY)"


# ---------------------------------------------------------------------------
# Topic — list accumulation
# ---------------------------------------------------------------------------


class Topic(BaseChannel[list[T]]):
    """Accumulates values into a list. Per-step by default, or across steps.

    Used for message history:
        class State(TypedDict):
            messages: Annotated[list, add_messages]  # or Topic()
    """

    __slots__ = ("values", "accumulate")
    values: list[Any]
    accumulate: bool

    def __init__(self, accumulate: bool = False, key: str = "") -> None:
        super().__init__(key)
        self.accumulate = accumulate
        self.values = []

    def get(self) -> list[T]:
        if not self.values:
            raise EmptyChannelError(f"Channel '{self.key}' is empty")
        return list(self.values)

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        if not self.accumulate:
            self.values = []
        for value in values:
            if isinstance(value, list):
                self.values.extend(value)
            else:
                self.values.append(value)
        return True

    def checkpoint(self) -> list[Any]:
        return list(self.values)

    def from_checkpoint(self, data: Any) -> None:
        if data is not None and isinstance(data, list):
            self.values = list(data)

    def copy(self) -> Topic[T]:
        ch = Topic[T](self.accumulate, self.key)
        ch.values = list(self.values)
        return ch

    def __repr__(self) -> str:
        return f"Topic({self.key}, {len(self.values)} items, accumulate={self.accumulate})"
