"""Conversation memory — buffer, summary, token-limited.

LangChain equivalent: ~5,000 lines across memory/ module with 10+ classes.
infy: ~200 lines, 3 strategies, one protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from infy.messages import HumanMessage, Message, SystemMessage

# ---------------------------------------------------------------------------
# Memory protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Memory(Protocol):
    """A memory that manages conversation history."""

    def load_memory_variables(self) -> list[Message]:
        """Load the current memory as a list of messages."""
        ...

    def save_context(self, input: Message, output: Message) -> None:
        """Save an input/output pair to memory."""
        ...

    def clear(self) -> None:
        """Clear all memory."""
        ...


# ---------------------------------------------------------------------------
# BufferMemory — keep everything
# ---------------------------------------------------------------------------


@dataclass
class BufferMemory:
    """Keeps all messages in a buffer. No eviction.

    Usage:
        mem = BufferMemory()
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
    """

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None

    def load_memory_variables(self) -> list[Message]:
        result: list[Message] = []
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
        result.extend(self.messages)
        return result

    def save_context(self, input: Message, output: Message) -> None:
        self.messages.append(input)
        self.messages.append(output)

    def clear(self) -> None:
        self.messages.clear()

    @property
    def message_count(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# TokenLimitedMemory — keep messages up to a token budget
# ---------------------------------------------------------------------------


@dataclass
class TokenLimitedMemory:
    """Keeps messages within a token budget by trimming the oldest.

    Uses infy.tokens.count_tokens for fast Rust-accelerated counting.

    Usage:
        mem = TokenLimitedMemory(max_tokens=4000)
        # Add many messages...
        messages = mem.load_memory_variables()  # only fits within budget
    """

    max_tokens: int = 4000
    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None

    def load_memory_variables(self) -> list[Message]:
        from infy.tokens import count_tokens

        result: list[Message] = []
        system_tokens = 0
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
            system_tokens = count_tokens(self.system_prompt)

        budget = self.max_tokens - system_tokens
        if budget <= 0:
            return result

        # Walk backwards from most recent, keep as many as fit
        kept: list[Message] = []
        used_tokens = 0
        for msg in reversed(self.messages):
            msg_tokens = count_tokens(_message_to_text(msg))
            if used_tokens + msg_tokens > budget:
                break
            kept.append(msg)
            used_tokens += msg_tokens

        kept.reverse()
        result.extend(kept)
        return result

    def save_context(self, input: Message, output: Message) -> None:
        self.messages.append(input)
        self.messages.append(output)

    def clear(self) -> None:
        self.messages.clear()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def estimated_tokens(self) -> int:
        from infy.tokens import count_tokens

        return sum(count_tokens(_message_to_text(m)) for m in self.messages)


# ---------------------------------------------------------------------------
# SummaryMemory — summarize old messages using the LLM
# ---------------------------------------------------------------------------


@dataclass
class SummaryMemory:
    """Keeps a running summary of old messages + recent messages in full.

    When the recent buffer exceeds `buffer_size` messages, the oldest
    messages are summarized and the summary replaces them.

    Usage:
        from infy.providers.openai import OpenAIChat

        mem = SummaryMemory(
            model=OpenAIChat("gpt-4o"),
            buffer_size=6,
        )
        # Add messages...
        messages = mem.load_memory_variables()
    """

    model: Any = None  # ChatModel — Any to avoid import cycle
    buffer_size: int = 6
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    system_prompt: str | None = None

    def load_memory_variables(self) -> list[Message]:
        result: list[Message] = []
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
        if self.summary:
            result.append(SystemMessage(content=f"Previous conversation summary:\n{self.summary}"))
        result.extend(self._recent_messages())
        return result

    def save_context(self, input: Message, output: Message) -> None:
        self.messages.append(input)
        self.messages.append(output)

        if len(self.messages) > self.buffer_size:
            self._summarize()

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def _recent_messages(self) -> list[Message]:
        """Return the most recent messages that fit in the buffer."""
        return self.messages[-self.buffer_size :]

    def _summarize(self) -> None:
        """Summarize old messages and keep only recent ones."""
        if self.model is None:
            # No model — just trim to buffer size
            self.messages = self._recent_messages()
            return

        # Split: messages to summarize vs messages to keep
        messages_to_summarize = self.messages[: -self.buffer_size]
        messages_to_keep = self._recent_messages()

        # Build summary prompt
        conversation = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {_message_to_text(m)}"
            for m in messages_to_summarize
        )

        summary_prompt = [
            SystemMessage(
                content="You are a helpful assistant that summarizes conversations concisely."
            ),
            HumanMessage(
                content=f"Summarize this conversation in 2-3 sentences:\n\n{conversation}"
            ),
        ]

        if self.summary:
            summary_prompt.insert(
                1,
                SystemMessage(content=f"Previous summary: {self.summary}"),
            )

        try:
            if hasattr(self.model, "generate"):
                response = self.model.generate(summary_prompt)
                self.summary = response.text
            elif callable(self.model):
                result = self.model(summary_prompt)
                self.summary = result.text if hasattr(result, "text") else str(result)
        except Exception:
            # If summarization fails, just keep recent messages
            pass

        self.messages = messages_to_keep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message_to_text(msg: Message) -> str:
    """Extract text content from any message type."""
    if isinstance(msg, str):
        return msg
    if hasattr(msg, "text"):
        return msg.text
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "".join(parts)
    return str(msg)
