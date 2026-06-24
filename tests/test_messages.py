"""Tests for infy messages."""

from infy.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    UsageMetadata,
)


def test_human_message():
    msg = HumanMessage(content="Hello")
    assert msg.type == "human"
    assert msg.text == "Hello"


def test_ai_message():
    msg = AIMessage(content="Hi there!")
    assert msg.type == "ai"
    assert msg.text == "Hi there!"
    assert not msg.has_tool_calls


def test_ai_message_with_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[ToolCall(name="search", args={"q": "test"}, id="call_1")],
    )
    assert msg.has_tool_calls
    assert msg.tool_calls[0].name == "search"


def test_system_message():
    msg = SystemMessage(content="You are helpful.")
    assert msg.type == "system"
    assert msg.text == "You are helpful."


def test_tool_message():
    msg = ToolMessage(content="result data", tool_call_id="call_1")
    assert msg.type == "tool"
    assert msg.tool_call_id == "call_1"


def test_usage_metadata_auto_total():
    usage = UsageMetadata(input_tokens=10, output_tokens=20)
    assert usage.total_tokens == 30


def test_usage_metadata_explicit_total():
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=35)
    assert usage.total_tokens == 35


def test_chunk_addition():
    c1 = AIMessageChunk(content="Hello")
    c2 = AIMessageChunk(content=" world")
    merged = c1 + c2
    assert merged.content == "Hello world"


def test_chunk_tool_call_merge():
    c1 = AIMessageChunk(
        tool_call_chunks=[
            ToolCallChunk(name="search", args='{"q":', id="c1", index=0),
        ]
    )
    c2 = AIMessageChunk(
        tool_call_chunks=[
            ToolCallChunk(args=' "test"}', index=0),
        ]
    )
    merged = c1 + c2
    assert len(merged.tool_call_chunks) == 1
    assert merged.tool_call_chunks[0].args == '{"q": "test"}'


def test_chunk_usage_merge():
    c1 = AIMessageChunk(usage=UsageMetadata(input_tokens=10, output_tokens=5))
    c2 = AIMessageChunk(usage=UsageMetadata(input_tokens=0, output_tokens=3))
    merged = c1 + c2
    assert merged.usage.input_tokens == 10
    assert merged.usage.output_tokens == 8
    assert merged.usage.total_tokens == 18


def test_chunk_materialize():
    chunk = AIMessageChunk(
        content="Hello",
        tool_call_chunks=[
            ToolCallChunk(name="search", args='{"q": "test"}', id="call_1", index=0),
        ],
        usage=UsageMetadata(input_tokens=10, output_tokens=5),
    )
    msg = chunk.materialize()
    assert isinstance(msg, AIMessage)
    assert msg.content == "Hello"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"
    assert msg.tool_calls[0].args == {"q": "test"}


def test_tool_call_to_dict():
    tc = ToolCall(name="search", args={"q": "test"}, id="call_1")
    d = tc.to_dict()
    assert d == {"name": "search", "args": {"q": "test"}, "id": "call_1"}
