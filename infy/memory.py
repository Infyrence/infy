"""Conversation memory — buffer, summary, token-limited, fact sheet.

LangChain equivalent: ~5,000 lines across memory/ module with 10+ classes.
infy: ~350 lines, 4 strategies, one protocol.

The four differ in *what they throw away*, which is the only interesting question once a
conversation outgrows the window:

- ``BufferMemory`` — nothing. Correct until it isn't affordable.
- ``TokenLimitedMemory`` — the oldest turns. Cheap, and silently drops the constraint someone
  stated in turn two.
- ``SummaryMemory`` — detail, into prose. Keeps the gist of a *conversation*.
- ``FactSheetMemory`` — prose, keeping the facts. Built for tool-using agents, where the bulk
  of the context is observations rather than dialogue and a paragraph of narrative summary is
  a poor trade for the four numbers it was derived from.

For facts that need to survive *across* conversations, and to answer "what did we believe
when we acted?", see ``infy.temporal.BiTemporalMemory``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from infy.messages import HumanMessage, Message, SystemMessage, ToolMessage

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
# FactSheetMemory — compact observations into facts, keep recent turns verbatim
# ---------------------------------------------------------------------------


@dataclass
class FactSheetMemory:
    """Recent turns verbatim, older tool output compacted into a list of established facts.

    A tool-using agent's context is mostly *observations*: JSON payloads, page text, command
    output. Trimming the oldest of those (``TokenLimitedMemory``) throws away the finding, and
    narrating them (``SummaryMemory``) spends a paragraph to carry a number. This keeps the
    last ``buffer_size`` messages exactly as they were — recent detail is what the next turn
    reasons over — and folds everything older into a flat fact sheet.

    Two ways to build the sheet:

    - **With a model**: it is asked to extract standalone facts, one per line, and told to
      discard narration. Extraction is the job LLMs are good at.
    - **Without one** (``model=None``): tool results are recorded verbatim as
      ``tool_name: result``. Lossier, but free, deterministic, and offline — which makes it
      the sane default in tests and a working fallback when extraction fails.

    Facts accumulate and are de-duplicated, so a run that re-reads the same file does not
    grow its sheet. ``max_facts`` bounds it; the oldest go first, on the assumption that a
    long run's later findings are built on its earlier ones.

    Usage::

        mem = FactSheetMemory(model=model, buffer_size=6, max_facts=40)
        mem.save_context(HumanMessage(content="..."), AIMessage(content="..."))
        messages = mem.load_memory_variables()   # [system prompt?, fact sheet, recent turns]
    """

    model: Any = None  # ChatModel — Any to avoid an import cycle
    buffer_size: int = 8
    max_facts: int = 50
    messages: list[Message] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    system_prompt: str | None = None

    def load_memory_variables(self) -> list[Message]:
        result: list[Message] = []
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
        if self.facts:
            result.append(SystemMessage(content=self.render_facts()))
        result.extend(self.messages[-self.buffer_size :])
        return result

    def save_context(self, input: Message, output: Message) -> None:
        self.messages.append(input)
        self.messages.append(output)
        if len(self.messages) > self.buffer_size:
            self._compact()

    def clear(self) -> None:
        self.messages.clear()
        self.facts.clear()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def render_facts(self) -> str:
        """The fact sheet as it is injected into the prompt."""
        lines = "\n".join(f"  - {f}" for f in self.facts)
        return f"Established so far (do not re-derive or re-fetch these):\n{lines}"

    def add_fact(self, fact: str) -> None:
        """Record a fact directly — useful for things the loop knows but never said aloud."""
        fact = " ".join(fact.split())
        if fact and fact not in self.facts:
            self.facts.append(fact)
            if len(self.facts) > self.max_facts:
                del self.facts[: len(self.facts) - self.max_facts]

    # --- internals --------------------------------------------------------

    def _compact(self) -> None:
        older = self.messages[: -self.buffer_size]
        self.messages = self.messages[-self.buffer_size :]
        if not older:
            return
        extracted = self._extract_with_model(older) if self.model is not None else []
        for fact in extracted or _observations(older):
            self.add_fact(fact)

    def _extract_with_model(self, older: list[Message]) -> list[str]:
        transcript = "\n".join(f"{_role_of(m)}: {_message_to_text(m)}" for m in older)
        prompt = [
            SystemMessage(
                content=(
                    "Extract the durable facts from this transcript: findings, values, "
                    "decisions, and constraints that later turns must not contradict. One "
                    "fact per line, no bullets, no numbering, no preamble. Each line must "
                    "stand alone without the transcript. Omit narration of what was tried."
                )
            ),
            HumanMessage(content=transcript[:12000]),
        ]
        try:
            response = self.model.generate(prompt)
            text = response.text if hasattr(response, "text") else str(response)
        except Exception:
            # Extraction is best-effort: a failed call must not lose the observations, so
            # the caller falls through to the deterministic path.
            return []
        return [line.strip(" -*\t") for line in text.splitlines() if line.strip()]


def _role_of(msg: Message) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, ToolMessage):
        return "tool"
    if isinstance(msg, SystemMessage):
        return "system"
    return "assistant"


def _observations(messages: list[Message]) -> list[str]:
    """Model-free fallback: keep tool results, which are the load-bearing part."""
    facts: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            text = " ".join(_message_to_text(msg).split())
            if text:
                facts.append(text[:300])
    return facts


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
