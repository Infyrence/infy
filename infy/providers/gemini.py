"""Google Gemini provider — sync + async.

LangChain equivalent: 2,000+ lines across partner package.
infy: ~200 lines with full async + tool calling.
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
class GeminiChat:
    """Google Gemini chat model. Sync + async.

    Usage:
        model = GeminiChat("gemini-2.0-flash")
        result = model.generate([HumanMessage(content="Hello!")])

        # Async
        result = await model.agenerate([HumanMessage(content="Hello!")])
    """

    model_name: str = "gemini-2.0-flash"
    api_key: str | None = None
    vertexai: bool = False
    _client: Any = field(default=None, repr=False)
    _aclient: Any = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def _ensure_client(self) -> None:
        if self._initialized:
            return
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key, vertexai=self.vertexai)
            self._aclient = self._client
            self._initialized = True
        except ImportError:
            raise ImportError(
                "Google GenAI not installed. Install with: pip install infy[gemini]"
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
        self._ensure_client()
        contents, system_text = self._convert_messages(messages)
        config = self._build_config(system_text, tools, tool_choice, temperature, max_tokens)

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config if config else None,
        )

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
        self._ensure_client()
        contents, system_text = self._convert_messages(messages)
        config = self._build_config(system_text, tools, tool_choice, temperature, max_tokens)

        response = await self._aclient.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config if config else None,
        )

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
        self._ensure_client()
        contents, system_text = self._convert_messages(messages)
        config = self._build_config(system_text, tools, tool_choice, temperature, max_tokens)

        for chunk in self._client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config if config else None,
        ):
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
        self._ensure_client()
        contents, system_text = self._convert_messages(messages)
        config = self._build_config(system_text, tools, tool_choice, temperature, max_tokens)

        async for chunk in self._aclient.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config if config else None,
        ):
            yield self._parse_stream_chunk(chunk)

    def bind_tools(self, tools: list[ToolSchema], **kwargs: Any) -> BoundChatModel:
        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputModel:
        return StructuredOutputModel(model=self, schema=schema)

    # --- Internal helpers ---

    def _convert_messages(self, messages: list[Message]) -> tuple[list[dict[str, Any]], str]:

        system_text = ""
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Merge (don't overwrite) so an injected schema/system instruction is
                # not clobbered by a caller-supplied system message later in the list.
                system_text = f"{system_text}\n{msg.text}" if system_text else msg.text
            elif isinstance(msg, AIMessage):
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "function_call": {
                                "name": tc.name,
                                "args": tc.args,
                            }
                        }
                    )
                contents.append({"role": "model", "parts": parts})
            elif hasattr(msg, "tool_call_id"):
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.name or "unknown",
                                    "response": {"content": msg.content},
                                }
                            }
                        ],
                    }
                )
            else:
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": msg.text if hasattr(msg, "text") else str(msg.content)}],
                    }
                )

        return contents, system_text

    def _build_config(
        self,
        system_text: str,
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        from google.genai.types import FunctionDeclaration, Tool

        config: dict[str, Any] = {}
        if system_text:
            config["system_instruction"] = system_text
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if tools:
            func_decls = [
                FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    # JSON-schema dict is coerced to a Schema by the SDK at runtime.
                    parameters=t.parameters if t.parameters else None,  # type: ignore[arg-type]
                )
                for t in tools
            ]
            config["tools"] = [Tool(function_declarations=func_decls)]
            mode = _to_gemini_mode(tool_choice)
            if mode is not None:
                config["tool_config"] = {"function_calling_config": {"mode": mode}}

        return config

    def _parse_response(self, response: Any) -> AIMessage:
        # Build text from parts directly: response.text raises when the response
        # is a pure function call with no text part (the tool-calling case).
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for i, part in enumerate(candidate.content.parts):
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        args = dict(fc.args) if fc.args else {}
                        tool_calls.append(
                            ToolCall(name=fc.name, args=args, id=f"call_{i}_{fc.name}")
                        )

        usage = None
        if response.usage_metadata:
            usage = UsageMetadata(
                input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
            )

        return AIMessage(content="".join(text_parts), tool_calls=tool_calls, usage=usage)

    def _parse_stream_chunk(self, chunk: Any) -> AIMessageChunk:
        text_parts: list[str] = []
        tc_chunks: list[ToolCallChunk] = []

        if chunk.candidates:
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for i, part in enumerate(candidate.content.parts):
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        args_str = json.dumps(dict(fc.args)) if fc.args else ""
                        tc_chunks.append(
                            ToolCallChunk(
                                name=fc.name,
                                args=args_str,
                                id=f"call_{i}_{fc.name}",
                                index=i,
                            )
                        )

        usage = None
        if chunk.usage_metadata:
            usage = UsageMetadata(
                input_tokens=getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0,
                output_tokens=getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0,
            )

        return AIMessageChunk(content="".join(text_parts), tool_call_chunks=tc_chunks, usage=usage)


def _to_gemini_mode(tool_choice: str | None) -> str | None:
    """Map the cross-provider ``tool_choice`` string to a Gemini calling mode."""
    if not tool_choice:
        return None
    return {
        "auto": "AUTO",
        "any": "ANY",
        "required": "ANY",
        "none": "NONE",
    }.get(tool_choice, "ANY")
