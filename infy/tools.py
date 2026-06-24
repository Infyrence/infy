"""Tool system — decorator-based, minimal, schema-auto-generating.

LangChain equivalent: 1,711 lines (BaseTool + args_schema + validation).
infy: ~140 lines with async support.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from infy.models import ToolSchema


@dataclass
class Tool:
    """A callable tool with auto-generated schema. Supports sync + async functions."""

    name: str
    description: str
    func: Callable[..., Any]
    args_schema: dict[str, Any] = field(default_factory=dict)
    _is_async: bool = field(default=False, repr=False)

    def invoke(self, input: str | dict[str, Any]) -> Any:
        if isinstance(input, str):
            try:
                input = json.loads(input)
            except (json.JSONDecodeError, TypeError):
                input = {"input": input}
        if not isinstance(input, dict):
            input = {"input": input}
        validated = self._validate(input)
        return self.func(**validated)

    async def ainvoke(self, input: str | dict[str, Any]) -> Any:
        if isinstance(input, str):
            try:
                input = json.loads(input)
            except (json.JSONDecodeError, TypeError):
                input = {"input": input}
        if not isinstance(input, dict):
            input = {"input": input}
        validated = self._validate(input)
        if self._is_async:
            return await self.func(**validated)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.func(**validated))

    def to_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, parameters=self.args_schema)

    def _validate(self, args: dict[str, Any]) -> dict[str, Any]:
        required = self.args_schema.get("required", [])
        properties = self.args_schema.get("properties", {})
        result = {k: v for k, v in args.items() if k in properties}
        for req in required:
            if req not in result:
                if req in properties and "default" in properties[req]:
                    result[req] = properties[req]["default"]
                else:
                    raise ValueError(f"Missing required argument: {req}")
        return result


def tool(
    func: Callable[..., Any] | None = None, *, name: str | None = None
) -> Tool | Callable[..., Any]:
    """Decorator to turn a function into a Tool. Works with sync and async functions."""

    def decorator(f: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or f.__name__,
            description=(f.__doc__ or "").strip(),
            func=f,
            args_schema=_schema_from_function(f),
            _is_async=asyncio.iscoroutinefunction(f),
        )

    if func is not None:
        return decorator(func)
    return decorator


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        prop: dict[str, Any] = {}
        if param.annotation is not inspect.Parameter.empty:
            json_type = _python_type_to_json(param.annotation)
            if json_type:
                prop["type"] = json_type
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)
        prop["description"] = ""
        properties[param_name] = prop
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _python_type_to_json(annotation: Any) -> str | None:
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return "string"
