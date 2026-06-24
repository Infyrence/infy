"""Tests for infy Gemini provider.

These tests require GOOGLE_API_KEY env var.
Run with: GOOGLE_API_KEY=xxx pytest tests/test_gemini.py -v
"""

import os

import pytest

from infy.messages import AIMessage, HumanMessage, SystemMessage

# Skip all tests if no API key or google-genai not installed
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    ),
    pytest.mark.skipif(
        not pytest.importorskip("google.genai", reason="google-genai not installed"),
        reason="google-genai not installed",
    ),
]


@pytest.fixture
def model():
    from infy.providers.gemini import GeminiChat

    return GeminiChat("gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


def test_generate_simple(model):
    result = model.generate([HumanMessage(content="Say hello in one word.")])
    assert isinstance(result, AIMessage)
    assert len(result.text) > 0
    print(f"Response: {result.text}")


def test_generate_with_system(model):
    result = model.generate(
        [
            SystemMessage(content="You are a pirate. Always respond in pirate speak."),
            HumanMessage(content="Hello!"),
        ]
    )
    assert isinstance(result, AIMessage)
    assert len(result.text) > 0
    print(f"Response: {result.text}")


def test_generate_with_temperature(model):
    result = model.generate(
        [HumanMessage(content="Pick a random number between 1 and 100.")],
        temperature=0.0,
    )
    assert isinstance(result, AIMessage)
    print(f"Response: {result.text}")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_simple(model):
    chunks = list(model.stream([HumanMessage(content="Say hello.")]))
    assert len(chunks) > 0
    full_text = "".join(c.content for c in chunks if c.content)
    assert len(full_text) > 0
    print(f"Streamed: {full_text}")


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agenerate_simple(model):
    result = await model.agenerate([HumanMessage(content="Say hello in one word.")])
    assert isinstance(result, AIMessage)
    assert len(result.text) > 0
    print(f"Async response: {result.text}")


@pytest.mark.asyncio
async def test_astream_simple(model):
    chunks = []
    async for chunk in model.astream([HumanMessage(content="Say hello.")]):
        chunks.append(chunk)
    assert len(chunks) > 0
    full_text = "".join(c.content for c in chunks if c.content)
    assert len(full_text) > 0
    print(f"Async streamed: {full_text}")


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


def test_generate_with_tools(model):
    from infy.models import ToolSchema

    tools = [
        ToolSchema(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        )
    ]

    result = model.generate(
        [HumanMessage(content="What's the weather in Paris?")],
        tools=tools,
    )
    assert isinstance(result, AIMessage)
    # Gemini may or may not call tools — just check it returns valid response
    print(f"Response with tools: {result.text}, tool_calls: {result.tool_calls}")


# ---------------------------------------------------------------------------
# Usage metadata
# ---------------------------------------------------------------------------


def test_usage_metadata(model):
    result = model.generate([HumanMessage(content="Hi")])
    if result.usage:
        assert result.usage.input_tokens >= 0
        assert result.usage.output_tokens >= 0
        print(f"Usage: {result.usage}")
