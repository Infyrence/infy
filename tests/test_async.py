"""Tests for async support across infy."""

import pytest

from infy import AsyncLambda, Lambda, Parallel, SystemMessage, tool
from infy.agents import AgentResult, create_async_agent
from infy.core import Context, Sequence, acoerce
from infy.messages import AIMessage, ToolCall

# ---------------------------------------------------------------------------
# Fake async model for testing
# ---------------------------------------------------------------------------


class FakeAsyncChatModel:
    """A fake chat model with async methods for testing."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.call_count = 0
        self.model_name = "fake-async"

    def generate(self, messages, *, tools=None, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Done.")

    async def agenerate(self, messages, *, tools=None, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Done.")

    def stream(self, messages, *, tools=None, **kwargs):
        yield self.generate(messages, tools=tools)

    async def astream(self, messages, *, tools=None, **kwargs):
        yield await self.agenerate(messages, tools=tools)

    def bind_tools(self, tools, **kwargs):
        from infy.models import BoundChatModel

        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)


# ---------------------------------------------------------------------------
# Async core tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lambda_ainvoke():
    fn = Lambda(lambda x: x + 1)
    result = await fn.ainvoke(5)
    assert result == 6


@pytest.mark.asyncio
async def test_async_lambda_ainvoke():
    async def add_one(x):
        return x + 1

    fn = AsyncLambda(add_one)
    result = await fn.ainvoke(5)
    assert result == 6


@pytest.mark.asyncio
async def test_sequence_ainvoke():
    s = Sequence([Lambda(lambda x: x + 1), Lambda(lambda x: x * 2)])
    result = await s.ainvoke(5)
    assert result == 12


@pytest.mark.asyncio
async def test_parallel_ainvoke():
    p = Parallel({"upper": Lambda(lambda x: x.upper()), "length": Lambda(lambda x: len(x))})
    result = await p.ainvoke("hello")
    assert result == {"upper": "HELLO", "length": 5}


@pytest.mark.asyncio
async def test_pipe_operator_async():
    chain = Lambda(lambda x: x + 1) | Lambda(lambda x: x * 2)
    result = await chain.ainvoke(5)
    assert result == 12


@pytest.mark.asyncio
async def test_acoerce_callable():
    r = acoerce(lambda x: x + 1)
    assert isinstance(r, Lambda)
    result = await r.ainvoke(5)
    assert result == 6


@pytest.mark.asyncio
async def test_acoerce_async_function():
    async def add(x):
        return x + 1

    r = acoerce(add)
    assert isinstance(r, AsyncLambda)
    result = await r.ainvoke(5)
    assert result == 6


@pytest.mark.asyncio
async def test_sequence_astream():
    s = Sequence([Lambda(lambda x: [x, x + 1]), Lambda(lambda x: x * 10)])
    chunks = []
    async for chunk in s.astream(5):
        chunks.append(chunk)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_parallel_astream():
    p = Parallel({"double": Lambda(lambda x: x * 2), "triple": Lambda(lambda x: x * 3)})
    chunks = []
    async for chunk in p.astream(5):
        chunks.append(chunk)
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Async tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_tool():
    @tool
    async def async_add(x: int, y: int) -> int:
        """Async add."""
        return x + y

    result = await async_add.ainvoke({"x": 1, "y": 2})
    assert result == 3


@pytest.mark.asyncio
async def test_sync_tool_ainvoke():
    @tool
    def sync_add(x: int, y: int) -> int:
        """Sync add."""
        return x + y

    result = await sync_add.ainvoke({"x": 1, "y": 2})
    assert result == 3


# ---------------------------------------------------------------------------
# Async agent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_agent_simple():
    model = FakeAsyncChatModel([AIMessage(content="Hello async!")])
    agent = create_async_agent(model)
    result = await agent("Hi")
    assert isinstance(result, AgentResult)
    assert result.response.text == "Hello async!"


@pytest.mark.asyncio
async def test_async_agent_with_tool_call():
    @tool
    def search(query: str) -> str:
        """Search."""
        return f"Results for {query}"

    model = FakeAsyncChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="search", args={"query": "test"}, id="call_1")],
            ),
            AIMessage(content="Found results!"),
        ]
    )
    agent = create_async_agent(model, tools=[search])
    result = await agent("Search for test")
    assert result.response.text == "Found results!"
    assert result.tool_calls_made == 1


@pytest.mark.asyncio
async def test_async_agent_with_system_prompt():
    model = FakeAsyncChatModel([AIMessage(content="Async response")])
    agent = create_async_agent(model, system_prompt="You are helpful.")
    result = await agent("Hello")
    assert result.response.text == "Async response"
    assert isinstance(result.messages[0], SystemMessage)


@pytest.mark.asyncio
async def test_async_agent_max_iterations():
    model = FakeAsyncChatModel(
        [AIMessage(content="", tool_calls=[ToolCall(name="unknown", args={}, id="c1")])] * 20
    )
    agent = create_async_agent(model, max_iterations=3)
    result = await agent("Do it")
    assert result.iterations == 3


# ---------------------------------------------------------------------------
# Async tool as Runnable in chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_in_async_chain():
    @tool
    def double(x: str) -> str:
        """Double the input."""
        return x * 2

    chain = Lambda(lambda x: {"x": x}) | double | Lambda(lambda x: f"Result: {x}")
    result = await chain.ainvoke("hello")
    assert result == "Result: hellohello"


# ---------------------------------------------------------------------------
# Context in async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_passed_through_async():
    ctx = Context(tags=["test"], metadata={"key": "value"})

    def check_ctx(x):
        return x

    fn = Lambda(check_ctx)
    result = await fn.ainvoke("hello", ctx)
    assert result == "hello"
