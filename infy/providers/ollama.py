"""Ollama provider — local models, sync + async.

LangChain equivalent: 1,794 lines (chat_models.py).
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
class OllamaChat:
    """Ollama local chat model. Sync + async."""

    model_name: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    _client: Any = field(default=None, repr=False)
    _aclient: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from ollama import AsyncClient, Client

            self._client = Client(host=self.base_url)
            self._aclient = AsyncClient(host=self.base_url)
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Install with: pip install infy[ollama]"
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
        params = self._build_params(messages, tools, temperature, max_tokens, stream=False)
        return _parse_response(self._client.chat(**params))

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
        params = self._build_params(messages, tools, temperature, max_tokens, stream=False)
        return _parse_response(await self._aclient.chat(**params))

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
        params = self._build_params(messages, tools, temperature, max_tokens, stream=True)
        for chunk in self._client.chat(**params):
            yield _parse_chunk(chunk)

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
        params = self._build_params(messages, tools, temperature, max_tokens, stream=True)
        async for chunk in await self._aclient.chat(**params):
            yield _parse_chunk(chunk)

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> BoundChatModel:
        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputModel:
        return StructuredOutputModel(model=self, schema=schema)

    def _build_params(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float | None,
        max_tokens: int | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_to_msg(m) for m in messages],
            "stream": stream,
        }
        if tools:
            # Ollama expects OpenAI-style tool definitions.
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            params["options"] = options
        return params


def _parse_response(response: Any) -> AIMessage:
    message = response.get("message", {})
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(message.get("tool_calls", []) or []):
        func = tc.get("function", {})
        tool_calls.append(
            ToolCall(
                name=func.get("name", ""),
                args=func.get("arguments", {}) or {},
                id=f"call_{i}_{func.get('name', '')}",
            )
        )
    usage = _parse_usage(response)
    return AIMessage(content=message.get("content", ""), tool_calls=tool_calls, usage=usage)


def _parse_chunk(chunk: Any) -> AIMessageChunk:
    message = chunk.get("message", {})
    tc_chunks: list[ToolCallChunk] = []
    for i, tc in enumerate(message.get("tool_calls", []) or []):
        func = tc.get("function", {})
        tc_chunks.append(
            ToolCallChunk(
                name=func.get("name"),
                args=json.dumps(func.get("arguments", {}) or {}),
                id=f"call_{i}_{func.get('name', '')}",
                index=i,
            )
        )
    return AIMessageChunk(
        content=message.get("content", ""),
        tool_call_chunks=tc_chunks,
        usage=_parse_usage(chunk),
    )


def _parse_usage(response: Any) -> UsageMetadata | None:
    prompt = response.get("prompt_eval_count")
    completion = response.get("eval_count")
    if prompt is None and completion is None:
        return None
    return UsageMetadata(input_tokens=prompt or 0, output_tokens=completion or 0)


def _to_msg(msg: Message) -> dict[str, Any]:
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.text}
    if isinstance(msg, AIMessage):
        m: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            m["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.args}} for tc in msg.tool_calls
            ]
        return m
    if hasattr(msg, "tool_call_id"):
        return {"role": "tool", "content": msg.content}
    return {"role": "user", "content": msg.text if hasattr(msg, "text") else str(msg.content)}
