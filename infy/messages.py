"""Message types — the language of LLM communication.

4 message types. No class hierarchy depth. No 7-level inheritance.
LangChain equivalent: 9,356 lines across 20 files.
infy: ~120 lines in one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Usage metadata — always present, never optional
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageMetadata:
    """Token usage from an API call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)


# ---------------------------------------------------------------------------
# Tool call types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model."""

    name: str
    args: dict[str, Any]
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "id": self.id}


@dataclass(frozen=True)
class ToolCallChunk:
    """A streaming chunk of a tool call."""

    name: str | None = None
    args: str = ""
    id: str | None = None
    index: int = 0


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass(frozen=True)
class ImageBlock:
    type: str = "image"
    url: str = ""
    mime_type: str = ""


@dataclass(frozen=True)
class ToolCallBlock:
    type: str = "tool_call"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass(frozen=True)
class ToolResultBlock:
    type: str = "tool_result"
    content: str = ""
    tool_call_id: str = ""


ContentBlock = TextBlock | ImageBlock | ToolCallBlock | ToolResultBlock


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class HumanMessage:
    """User input."""

    content: str | list[ContentBlock]
    name: str | None = None
    id: str | None = None

    @property
    def type(self) -> str:
        return "human"

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        parts = []
        for block in self.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)


@dataclass
class AIMessage:
    """Model output."""

    content: str | list[ContentBlock]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: UsageMetadata | None = None
    name: str | None = None
    id: str | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return "ai"

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        parts = []
        for block in self.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class SystemMessage:
    """System instruction."""

    content: str | list[ContentBlock]
    name: str | None = None

    @property
    def type(self) -> str:
        return "system"

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(b.text for b in self.content if hasattr(b, "text"))


@dataclass
class ToolMessage:
    """Tool execution result."""

    content: str
    tool_call_id: str
    name: str | None = None
    artifact: Any = None
    status: str = "success"

    @property
    def type(self) -> str:
        return "tool"


# Union type
Message = HumanMessage | AIMessage | SystemMessage | ToolMessage


# ---------------------------------------------------------------------------
# Streaming chunks
# ---------------------------------------------------------------------------


@dataclass
class AIMessageChunk:
    """A streaming chunk of an AI message. Additive via __add__."""

    content: str | list[ContentBlock] = ""
    tool_call_chunks: list[ToolCallChunk] = field(default_factory=list)
    usage: UsageMetadata | None = None

    def __add__(self, other: AIMessageChunk) -> AIMessageChunk:
        # Merge content (narrow to matching types so concatenation is well-typed).
        # Kept as separate branches on purpose — combining them defeats the
        # type narrowing that makes the concatenations valid.
        content: str | list[ContentBlock]
        if isinstance(self.content, str) and isinstance(other.content, str):  # noqa: SIM114
            content = self.content + other.content
        elif isinstance(self.content, list) and isinstance(other.content, list):
            content = self.content + other.content
        else:
            content = other.content if other.content else self.content

        # Merge tool call chunks by index
        merged_chunks: list[ToolCallChunk] = list(self.tool_call_chunks)
        for new_chunk in other.tool_call_chunks:
            found = False
            for i, existing in enumerate(merged_chunks):
                if existing.index == new_chunk.index:
                    merged_chunks[i] = ToolCallChunk(
                        name=existing.name or new_chunk.name,
                        args=existing.args + new_chunk.args,
                        id=existing.id or new_chunk.id,
                        index=existing.index,
                    )
                    found = True
                    break
            if not found:
                merged_chunks.append(new_chunk)

        # Merge usage (additive)
        usage = self.usage
        if other.usage:
            if usage:
                usage = UsageMetadata(
                    input_tokens=usage.input_tokens + other.usage.input_tokens,
                    output_tokens=usage.output_tokens + other.usage.output_tokens,
                )
            else:
                usage = other.usage

        return AIMessageChunk(
            content=content,
            tool_call_chunks=merged_chunks,
            usage=usage,
        )

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(b.text for b in self.content if hasattr(b, "text"))

    def materialize(self) -> AIMessage:
        """Convert accumulated chunk to a full AIMessage."""
        tool_calls: list[ToolCall] = []
        for tc in self.tool_call_chunks:
            if tc.name:
                import json

                args_str = tc.args
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(name=tc.name, args=args, id=tc.id or ""))
        return AIMessage(
            content=self.content,
            tool_calls=tool_calls,
            usage=self.usage,
        )
