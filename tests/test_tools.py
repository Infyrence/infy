"""Tests for infy tools."""

import json

import pytest

from infy.tools import Tool, ToolSchema, tool


def test_tool_decorator():
    @tool
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.invoke({"x": 1, "y": 2}) == 3


def test_tool_decorator_with_custom_name():
    @tool(name="my_add")
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    assert add.name == "my_add"


def test_tool_invoke_string():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    result = greet.invoke(json.dumps({"name": "World"}))
    assert result == "Hello, World!"


def test_tool_invoke_plain_string():
    @tool
    def echo(input: str) -> str:
        """Echo input."""
        return input

    result = echo.invoke("hello")
    assert result == "hello"


def test_tool_schema_generation():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return "results"

    schema = search.to_schema()
    assert isinstance(schema, ToolSchema)
    assert schema.name == "search"
    assert schema.description == "Search the web."
    assert schema.parameters["type"] == "object"
    assert "query" in schema.parameters["properties"]


def test_tool_schema_to_dict():
    @tool
    def add(x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    d = add.to_schema().to_dict()
    assert d["type"] == "function"
    assert d["function"]["name"] == "add"
    assert "parameters" in d["function"]


def test_tool_validation_missing_required():
    @tool
    def add(x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    with pytest.raises(ValueError, match="y"):
        add.invoke({"x": 1})  # Missing y


def test_tool_validation_with_default():
    @tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    result = greet.invoke({"name": "World"})
    assert result == "Hello, World!"


def test_tool_schema_types():
    @tool
    def func(a: str, b: int, c: float, d: bool) -> str:
        """Test types."""
        return ""

    schema = func.args_schema
    props = schema["properties"]
    assert props["a"]["type"] == "string"
    assert props["b"]["type"] == "integer"
    assert props["c"]["type"] == "number"
    assert props["d"]["type"] == "boolean"
