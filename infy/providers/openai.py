"""OpenAI provider — sync + async.

LangChain equivalent: 5,131 lines (chat_models/base.py).
infy: ~250 lines with full async support.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from infy.messages import (
    AIMessage,
    AIMessageChunk,
    Message,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    UsageMetadata,
)
from infy.models import BoundChatModel, StructuredOutputModel, ToolSchema


@dataclass
class OpenAIChat:
    """OpenAI chat model. Sync + async.

    Usage:
        model = OpenAIChat("gpt-4o")
        result = model.generate([HumanMessage(content="Hello!")])

        # Async
        result = await model.agenerate([HumanMessage(content="Hello!")])
    """

    model_name: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    _client: Any = field(default=None, repr=False)
    _aclient: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            self._aclient = AsyncOpenAI(**kwargs)
        except ImportError:
            raise ImportError(
                "OpenAI not installed. Install with: pip install infy[openai]"
            ) from None

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        params = self._build_params(messages, tools, tool_choice, temperature, max_tokens, kwargs)
        response = self._client.chat.completions.create(**params)
        return self._parse_response(response)

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        params = self._build_params(messages, tools, tool_choice, temperature, max_tokens, kwargs)
        response = await self._aclient.chat.completions.create(**params)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        params = self._build_params(messages, tools, tool_choice, temperature, max_tokens, kwargs)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        for chunk in self._client.chat.completions.create(**params):
            yield self._parse_stream_chunk(chunk)

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        params = self._build_params(messages, tools, tool_choice, temperature, max_tokens, kwargs)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        async for chunk in await self._aclient.chat.completions.create(**params):
            yield self._parse_stream_chunk(chunk)

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> BoundChatModel:
        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputModel:
        return StructuredOutputModel(model=self, schema=schema)

    def _build_params(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [self._to_msg(m) for m in messages],
        }
        if tools:
            params["tools"] = [t.to_dict() for t in tools]
        if tool_choice:
            params["tool_choice"] = tool_choice
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(extra)
        return params

    def _parse_response(self, response: Any) -> AIMessage:
        choice = response.choices[0]
        tool_calls = [
            ToolCall(
                name=tc.function.name,
                args=json.loads(tc.function.arguments) if tc.function.arguments else {},
                id=tc.id,
            )
            for tc in (choice.message.tool_calls or [])
        ]
        usage = None
        if response.usage:
            usage = UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return AIMessage(
            content=choice.message.content or "", tool_calls=tool_calls, usage=usage, id=response.id
        )

    def _parse_stream_chunk(self, chunk: Any) -> AIMessageChunk:
        delta = chunk.choices[0].delta if chunk.choices else None
        usage = None
        if chunk.usage:
            usage = UsageMetadata(
                input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens
            )
        content = delta.content if delta and delta.content else ""
        tc_chunks = [
            ToolCallChunk(
                name=tc.function.name if tc.function else None,
                args=tc.function.arguments if tc.function else "",
                id=tc.id,
                index=tc.index or 0,
            )
            for tc in (delta.tool_calls if delta and delta.tool_calls else [])
        ]
        return AIMessageChunk(content=content, tool_call_chunks=tc_chunks, usage=usage)

    def _to_msg(self, msg: Message) -> dict[str, Any]:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.text}
        if isinstance(msg, AIMessage):
            m: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in msg.tool_calls
                ]
            return m
        if hasattr(msg, "tool_call_id"):
            return {"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id}
        return {"role": "user", "content": msg.text if hasattr(msg, "text") else str(msg.content)}
