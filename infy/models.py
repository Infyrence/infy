"""Chat model protocol — the interface every LLM provider implements.

LangChain equivalent: 2,711 lines (BaseChatModel) + per-provider implementations.
infy: ~130 lines protocol + ~200 lines per provider.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

from infy.messages import AIMessage, AIMessageChunk, Message

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSchema:
    """JSON Schema for a tool, as sent to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Structured output wrapper
# ---------------------------------------------------------------------------


@dataclass
class StructuredOutputModel:
    """A wrapper around a ChatModel that automatically parses structured output.

    Created by ``model.with_structured_output(schema)``. The model is instructed (via a
    cached system message) to return JSON matching ``schema``, and the reply is parsed:

    - ``schema`` is a **pydantic model class** -> the reply is validated with
      ``model_validate_json`` (pydantic-core's fused parse+validate) and the **validated
      model instance** is returned. Requires pydantic (``pip install infy[pydantic]``).
    - ``schema`` is a **plain dict** (JSON schema) -> the reply is parsed with the Rust
      ``JsonParser`` and a **dict** is returned (no validation).
    """

    model: Any  # ChatModel
    schema: Any  # dict JSON-schema OR a pydantic BaseModel subclass
    _parser: Any = field(default=None, repr=False)
    _pydantic: Any = field(default=None, repr=False)
    _instruction: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self._parser is None:
            from infy.parsers import JsonParser

            self._parser = JsonParser()
        # Detect a pydantic model once, and cache the schema instruction so it is not
        # re-serialized on every call.
        self._pydantic = _as_pydantic_model(self.schema)
        schema_dict = self._pydantic.model_json_schema() if self._pydantic else self.schema
        self._instruction = _build_instruction(schema_dict)

    @property
    def model_name(self) -> str:
        return cast(str, self.model.model_name)

    def generate(self, messages: list[Message], **kwargs: Any) -> Any:
        """Generate, then validate (pydantic) or parse (dict) the structured output."""
        response = self.model.generate(self._enhance(messages), **kwargs)
        return self._coerce(response.text)

    async def agenerate(self, messages: list[Message], **kwargs: Any) -> Any:
        """Async generate, then validate (pydantic) or parse (dict)."""
        response = await self.model.agenerate(self._enhance(messages), **kwargs)
        return self._coerce(response.text)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[Any]:
        """Stream partial structured outputs (always partial dicts — pydantic validation
        applies on generate/agenerate, not mid-stream)."""
        for chunk in self.model.stream(self._enhance(messages), **kwargs):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if text:
                with contextlib.suppress(Exception):
                    yield self._parser.parse(text, partial=True)

    async def astream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[Any]:
        """Async stream partial structured outputs (partial dicts; see ``stream``)."""
        async for chunk in self.model.astream(self._enhance(messages), **kwargs):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if text:
                with contextlib.suppress(Exception):
                    yield self._parser.parse(text, partial=True)

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> StructuredOutputModel:
        return StructuredOutputModel(
            model=self.model.bind_tools(tools, **kwargs),
            schema=self.schema,
            _parser=self._parser,
        )

    def __or__(self, other: Any) -> Any:
        from infy.core import Sequence, coerce

        return Sequence([coerce(self), coerce(other)])

    def __ror__(self, other: Any) -> Any:
        from infy.core import Sequence, coerce

        return Sequence([coerce(other), coerce(self)])

    def invoke(self, input: Any, ctx: Any = None, **kwargs: Any) -> Any:
        return self.generate(self._as_messages(input), **kwargs)

    async def ainvoke(self, input: Any, ctx: Any = None, **kwargs: Any) -> Any:
        return await self.agenerate(self._as_messages(input), **kwargs)

    # --- internals --------------------------------------------------------

    def _enhance(self, messages: list[Message]) -> list[Message]:
        from infy.messages import SystemMessage

        return [SystemMessage(content=self._instruction), *messages]

    def _coerce(self, text: str) -> Any:
        if self._pydantic is not None:
            try:
                return self._pydantic.model_validate_json(text)
            except Exception as validation_error:
                # The model may have wrapped the JSON in prose/fences: extract, then
                # validate. If extraction also fails, surface the original (clean)
                # validation error rather than the parser's ValueError.
                try:
                    return self._pydantic.model_validate(self._parser.parse(text))
                except Exception:
                    raise validation_error from None
        return self._parser.parse(text)

    def _as_messages(self, input: Any) -> list[Message]:
        from infy.messages import HumanMessage

        if isinstance(input, list):
            return input
        if isinstance(input, str):
            return [HumanMessage(content=input)]
        return [HumanMessage(content=str(input))]


def _as_pydantic_model(schema: Any) -> Any:
    """Return ``schema`` if it is a pydantic ``BaseModel`` subclass, else ``None``.

    Uses a lazy import so the core stays zero-dependency when pydantic is absent.
    """
    if not isinstance(schema, type):
        return None
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    return schema if issubclass(schema, BaseModel) else None


def _build_instruction(schema: dict[str, Any]) -> str:
    """Build the JSON-schema system instruction (computed once, then cached)."""
    import json as _json

    schema_str = _json.dumps(schema, indent=2)
    return (
        "You must respond with valid JSON that conforms to this schema:\n"
        f"```json\n{schema_str}\n```\n"
        "Return ONLY the JSON object. No explanation, no markdown, no code fences."
    )


# ---------------------------------------------------------------------------
# ChatModel protocol — sync + async
# ---------------------------------------------------------------------------


@runtime_checkable
class ChatModel(Protocol):
    """The ENTIRE contract for a chat model. Sync + async."""

    model_name: str

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIMessage: ...

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIMessage: ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]: ...

    # NOTE: a plain `def`, not `async def`: astream is an async *generator* — callers
    # `async for` over its return value, they do not `await` it. Declaring it `async def`
    # would type it as a coroutine and break every `async for chunk in model.astream(...)`.
    def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]: ...

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> BoundChatModel: ...

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputModel:
        """Return a model wrapper that automatically parses structured JSON output."""
        ...


# ---------------------------------------------------------------------------
# BoundChatModel
# ---------------------------------------------------------------------------


@dataclass
class BoundChatModel:
    """A chat model with tools pre-bound. Created by model.bind_tools()."""

    model: ChatModel
    bound_tools: list[ToolSchema]
    bound_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self.model.model_name

    def generate(self, messages: list[Message], **kwargs: Any) -> AIMessage:
        merged = {**self.bound_kwargs, **kwargs}
        return self.model.generate(messages, tools=self.bound_tools, **merged)

    async def agenerate(self, messages: list[Message], **kwargs: Any) -> AIMessage:
        merged = {**self.bound_kwargs, **kwargs}
        return await self.model.agenerate(messages, tools=self.bound_tools, **merged)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[AIMessageChunk]:
        merged = {**self.bound_kwargs, **kwargs}
        return self.model.stream(messages, tools=self.bound_tools, **merged)

    async def astream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[AIMessageChunk]:
        merged = {**self.bound_kwargs, **kwargs}
        async for chunk in self.model.astream(messages, tools=self.bound_tools, **merged):
            yield chunk

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> BoundChatModel:
        return BoundChatModel(
            model=self.model,
            bound_tools=tools,
            bound_kwargs={**self.bound_kwargs, **kwargs},
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputModel:
        """Return a model wrapper that automatically parses structured JSON output."""
        return StructuredOutputModel(model=self, schema=schema)
