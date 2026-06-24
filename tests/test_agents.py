"""Tests for infy agents."""

from infy import AIMessage, HumanMessage, SystemMessage, ToolMessage, tool
from infy.agents import AgentResult, create_agent
from infy.messages import ToolCall

# ---------------------------------------------------------------------------
# Fake model for testing (no API calls)
# ---------------------------------------------------------------------------


class FakeChatModel:
    """A fake chat model that returns pre-programmed responses."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.call_count = 0
        self.model_name = "fake"

    def generate(self, messages, *, tools=None, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Done.")

    def stream(self, messages, *, tools=None, **kwargs):
        yield self.generate(messages, tools=tools, **kwargs)

    def bind_tools(self, tools, **kwargs):
        from infy.models import BoundChatModel

        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_simple_no_tools():
    model = FakeChatModel([AIMessage(content="Hello!")])
    agent = create_agent(model)
    result = agent("Hi")

    assert isinstance(result, AgentResult)
    assert result.response.text == "Hello!"
    assert result.iterations == 1
    assert result.tool_calls_made == 0


def test_agent_with_tool_call():
    @tool
    def search(query: str) -> str:
        """Search."""
        return f"Results for {query}"

    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="search", args={"query": "test"}, id="call_1"),
                ],
            ),
            AIMessage(content="Found results!"),
        ]
    )

    agent = create_agent(model, tools=[search])
    result = agent("Search for test")

    assert result.response.text == "Found results!"
    assert result.tool_calls_made == 1
    # Messages: HumanMessage + AIMessage(tool_call) + ToolMessage + AIMessage(response)
    assert len(result.messages) == 4


def test_agent_with_system_prompt():
    model = FakeChatModel([AIMessage(content="Helpful response")])
    agent = create_agent(model, system_prompt="You are helpful.")
    result = agent("Hello")

    assert result.response.text == "Helpful response"
    # First message should be SystemMessage
    assert isinstance(result.messages[0], SystemMessage)


def test_agent_multiple_tool_calls():
    @tool
    def calc(expr: str) -> str:
        """Calculate."""
        return "42"

    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="calc", args={"expr": "1+1"}, id="call_1"),
                    ToolCall(name="calc", args={"expr": "2+2"}, id="call_2"),
                ],
            ),
            AIMessage(content="Results: 42, 42"),
        ]
    )

    agent = create_agent(model, tools=[calc], parallel_tools=False)
    result = agent("Calculate")

    assert result.tool_calls_made == 2


def test_agent_max_iterations():
    # Model always returns tool calls -> should stop at max_iterations
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="unknown_tool", args={}, id="call_1"),
                ],
            )
        ]
        * 20
    )  # Plenty of responses

    agent = create_agent(model, max_iterations=3)
    result = agent("Do something")

    assert result.iterations == 3


def test_agent_with_message_list_input():
    model = FakeChatModel([AIMessage(content="Response")])
    agent = create_agent(model)

    messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hello"),
    ]
    result = agent(messages)
    assert result.response.text == "Response"


def test_agent_tool_error_handling():
    @tool
    def failing_tool(input: str) -> str:
        """This tool always fails."""
        raise RuntimeError("Boom!")

    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="failing_tool", args={"input": "test"}, id="call_1"),
                ],
            ),
            AIMessage(content="The tool failed."),
        ]
    )

    agent = create_agent(model, tools=[failing_tool])
    result = agent("Use the tool")

    # Should have an error ToolMessage
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert any("Error" in m.content for m in tool_msgs)
    assert result.response.text == "The tool failed."
