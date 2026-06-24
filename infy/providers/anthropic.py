"""Anthropic provider — sync + async.

LangChain equivalent: 2,369 lines (chat_models.py).
infy: ~280 lines with full async support.
"""

from __future__ import annotations

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
class AnthropicChat:
    """Anthropic Claude chat model. Sync + async."""

    model_name: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    _client: Any = field(default=None, repr=False)
    _aclient: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from anthropic import Anthropic, AsyncAnthropic

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = Anthropic(**kwargs)
            self._aclient = AsyncAnthropic(**kwargs)
        except ImportError:
            raise ImportError(
                "Anthropic not installed. Install with: pip install infy[anthropic]"
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
        system_text, msg_list = self._convert_messages(messages)
        params = self._build_params(
            system_text, msg_list, tools, tool_choice, temperature, max_tokens
        )
        response = self._client.messages.create(**params)
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
        system_text, msg_list = self._convert_messages(messages)
        params = self._build_params(
            system_text, msg_list, tools, tool_choice, temperature, max_tokens
        )
        response = await self._aclient.messages.create(**params)
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
        system_text, msg_list = self._convert_messages(messages)
        params = self._build_params(
            system_text, msg_list, tools, tool_choice, temperature, max_tokens
        )
        with self._client.messages.stream(**params) as stream:
            yield from self._iter_stream(stream)

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
        system_text, msg_list = self._convert_messages(messages)
        params = self._build_params(
            system_text, msg_list, tools, tool_choice, temperature, max_tokens
        )
        async with self._aclient.messages.stream(**params) as stream:
            async for event in stream:
                chunk = self._event_to_chunk(event)
                if chunk is not None:
                    yield chunk
            final = await stream.get_final_message()
            if final is not None:
                yield AIMessageChunk(
                    content="",
                    usage=UsageMetadata(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                    ),
                )

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> BoundChatModel:
        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputModel:
        return StructuredOutputModel(model=self, schema=schema)

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_text = ""
        msg_list: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_text = m.text
            elif hasattr(m, "tool_call_id"):
                msg_list.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            elif isinstance(m, AIMessage):
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                    )
                msg_list.append({"role": "assistant", "content": content})
            else:
                msg_list.append(
                    {"role": "user", "content": m.text if hasattr(m, "text") else str(m.content)}
                )
        return system_text, msg_list

    def _build_params(
        self,
        system_text: str,
        msg_list: list[dict[str, Any]],
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens or 4096,
            "messages": msg_list,
        }
        if system_text:
            params["system"] = system_text
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
            choice = _to_anthropic_tool_choice(tool_choice)
            if choice is not None:
                params["tool_choice"] = choice
        return params

    def _parse_response(self, response: Any) -> AIMessage:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(name=block.name, args=block.input, id=block.id))
        return AIMessage(
            content="".join(text_parts),
            tool_calls=tool_calls,
            usage=UsageMetadata(
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
            ),
        )

    def _iter_stream(self, stream: Any) -> Iterator[AIMessageChunk]:
        for event in stream:
            chunk = self._event_to_chunk(event)
            if chunk is not None:
                yield chunk
        final = stream.get_final_message()
        if final is not None:
            yield AIMessageChunk(
                content="",
                usage=UsageMetadata(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                ),
            )

    def _event_to_chunk(self, event: Any) -> AIMessageChunk | None:
        """Convert one streaming event to a chunk, keyed by content-block index.

        Using ``event.index`` (the position of the block in the message) means
        parallel tool calls land in distinct chunks instead of all collapsing
        onto index 0 and merging into a single mangled call.
        """
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                return AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        ToolCallChunk(name=block.name, id=block.id, index=event.index)
                    ],
                )
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                return AIMessageChunk(content=event.delta.text)
            if event.delta.type == "input_json_delta":
                return AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        ToolCallChunk(args=event.delta.partial_json, index=event.index)
                    ],
                )
        return None


def _to_anthropic_tool_choice(tool_choice: str | None) -> dict[str, Any] | None:
    """Map the cross-provider ``tool_choice`` string to Anthropic's schema."""
    if not tool_choice:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice in ("any", "required"):
        return {"type": "any"}
    # Anything else is treated as a specific tool name.
    return {"type": "tool", "name": tool_choice}
