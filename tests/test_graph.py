"""Tests for infy graph system — state graph, channels, checkpoint, prebuilt agent."""

import pytest

from infy.graph import (
    END,
    START,
    InMemorySaver,
    StateGraph,
)
from infy.graph.channels import BinOp, LastValue, Topic
from infy.graph.checkpoint import Checkpoint, CheckpointMetadata
from infy.graph.errors import (
    EmptyChannelError,
    GraphInterrupt,
    GraphRecursionError,
    InvalidUpdateError,
    NodeError,
)
from infy.graph.types import Command, Send
from infy.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Channel tests
# ---------------------------------------------------------------------------


class TestLastValue:
    def test_empty(self):
        ch = LastValue(key="x")
        assert not ch.is_available()
        with pytest.raises(EmptyChannelError):
            ch.get()

    def test_set_and_get(self):
        ch = LastValue(key="x")
        ch.update([42])
        assert ch.get() == 42
        assert ch.is_available()

    def test_overwrite(self):
        ch = LastValue(key="x")
        ch.update([1])
        ch.update([2])
        assert ch.get() == 2

    def test_multiple_per_step_raises(self):
        ch = LastValue(key="x")
        with pytest.raises(InvalidUpdateError):
            ch.update([1, 2])

    def test_checkpoint_roundtrip(self):
        ch = LastValue(key="x")
        ch.update([42])
        data = ch.checkpoint()
        ch2 = LastValue(key="x")
        ch2.from_checkpoint(data)
        assert ch2.get() == 42

    def test_copy(self):
        ch = LastValue(key="x")
        ch.update([42])
        ch2 = ch.copy()
        assert ch2.get() == 42
        ch2.update([99])
        assert ch.get() == 42  # Original unchanged


class TestBinOp:
    def test_empty(self):
        ch = BinOp(operator=lambda a, b: a + b, key="count")
        assert not ch.is_available()

    def test_accumulate(self):
        ch = BinOp(operator=lambda a, b: a + b, key="count")
        ch.update([5])
        ch.update([3])
        assert ch.get() == 8

    def test_list_append(self):
        ch = BinOp(operator=lambda a, b: a + b, key="items")
        ch.update([[1, 2]])
        ch.update([[3, 4]])
        assert ch.get() == [1, 2, 3, 4]

    def test_checkpoint_roundtrip(self):
        ch = BinOp(operator=lambda a, b: a + b, key="x")
        ch.update([10])
        data = ch.checkpoint()
        ch2 = BinOp(operator=lambda a, b: a + b, key="x")
        ch2.from_checkpoint(data)
        assert ch2.get() == 10


class TestTopic:
    def test_per_step_clear(self):
        ch = Topic(key="msgs", accumulate=False)
        ch.update(["hello"])
        assert ch.get() == ["hello"]
        ch.update(["world"])  # Should clear previous
        assert ch.get() == ["world"]

    def test_accumulate(self):
        ch = Topic(key="msgs", accumulate=True)
        ch.update(["hello"])
        ch.update(["world"])
        assert ch.get() == ["hello", "world"]

    def test_flatten_lists(self):
        ch = Topic(key="msgs", accumulate=True)
        ch.update([["a", "b"], "c"])
        assert ch.get() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_create(self):
        cp = Checkpoint(channel_values={"x": 1, "y": 2})
        assert cp.v == 1
        assert cp.channel_values == {"x": 1, "y": 2}

    def test_copy(self):
        cp = Checkpoint(channel_values={"x": 1})
        cp2 = cp.copy()
        cp2.channel_values["x"] = 99
        assert cp.channel_values["x"] == 1

    def test_in_memory_saver(self):
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "t1"}}
        cp = Checkpoint(channel_values={"x": 1})
        meta = CheckpointMetadata(source="input", step=0)

        saver.put(config, cp, meta, {"x": 1})
        loaded = saver.get_tuple(config)
        assert loaded is not None
        assert loaded.checkpoint.channel_values["x"] == 1

    def test_in_memory_list(self):
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "t1"}}

        for i in range(5):
            cp = Checkpoint(channel_values={"step": i})
            meta = CheckpointMetadata(source="loop", step=i)
            saver.put(config, cp, meta, {"step": i})

        checkpoints = list(saver.list(config, limit=3))
        assert len(checkpoints) == 3
        # list returns newest first (reversed)
        assert checkpoints[0].checkpoint.channel_values["step"] == 4
        assert checkpoints[2].checkpoint.channel_values["step"] == 2

    def test_delete_thread(self):
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "t1"}}
        saver.put(config, Checkpoint(), CheckpointMetadata(), {})
        saver.delete_thread("t1")
        assert saver.get_tuple(config) is None


# ---------------------------------------------------------------------------
# StateGraph tests
# ---------------------------------------------------------------------------


class TestStateGraph:
    def test_linear_chain(self):
        """A → B → C"""

        def node_a(state):
            return {"value": "a"}

        def node_b(state):
            return {"value": state.get("value", "") + "b"}

        def node_c(state):
            return {"value": state.get("value", "") + "c"}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        app = graph.compile()
        result = app.invoke({})
        assert result["value"] == "abc"

    def test_conditional_routing(self):
        """START → decide → (path_a | path_b) → END"""

        def decide(state):
            return "path_a" if state.get("fast") else "path_b"

        def path_a(state):
            return {"result": "fast path"}

        def path_b(state):
            return {"result": "slow path"}

        graph = StateGraph()
        graph.add_node("decide", decide)
        graph.add_node("path_a", path_a)
        graph.add_node("path_b", path_b)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", decide, {"path_a": "path_a", "path_b": "path_b"})
        graph.add_edge("path_a", END)
        graph.add_edge("path_b", END)

        app = graph.compile()

        result = app.invoke({"fast": True})
        assert result["result"] == "fast path"

        result = app.invoke({"fast": False})
        assert result["result"] == "slow path"

    def test_parallel_fanout(self):
        """START → fanout → Send×3 → process → reducer-merged results → END."""
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            results: Annotated[list, operator.add]

        def fanout(state):
            return {}

        def route(state):
            return [Send("process", i) for i in range(3)]

        def process(state):
            return {"results": [state["input"] * 2]}

        graph = StateGraph(State)
        graph.add_node("fanout", fanout)
        graph.add_node("process", process)
        graph.add_edge(START, "fanout")
        graph.add_conditional_edges("fanout", route)
        graph.add_edge("process", END)

        result = graph.compile().invoke({})
        # All three Send tasks ran and their writes merged through the reducer.
        assert sorted(result["results"]) == [0, 2, 4]

    def test_state_accumulation(self):
        """Test that state accumulates across nodes."""

        def add_a(state):
            return {"log": ["step_a"]}

        def add_b(state):
            existing = state.get("log", [])
            return {"log": existing + ["step_b"]}

        graph = StateGraph()
        graph.add_node("a", add_a)
        graph.add_node("b", add_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        app = graph.compile()
        result = app.invoke({})
        assert "step_a" in result.get("log", [])
        assert "step_b" in result.get("log", [])

    def test_empty_graph_raises(self):
        graph = StateGraph()
        with pytest.raises(ValueError, match="no nodes"):
            graph.compile()

    def test_command_from_node(self):
        """Test that a node can return a Command to navigate."""

        def router(state):
            return Command(goto="target", update={"routed": True})

        def target(state):
            return {"done": True}

        graph = StateGraph()
        graph.add_node("router", router)
        graph.add_node("target", target)
        graph.add_edge(START, "router")
        graph.add_edge("target", END)

        app = graph.compile()
        result = app.invoke({})
        assert result.get("done") is True
        assert result.get("routed") is True

    def test_node_error_wrapping(self):
        def bad_node(state):
            raise ValueError("boom")

        graph = StateGraph()
        graph.add_node("bad", bad_node)
        graph.add_edge(START, "bad")
        graph.add_edge("bad", END)

        app = graph.compile()
        with pytest.raises(NodeError):
            app.invoke({})

    def test_interrupt(self):
        """Test interrupt before a node."""

        def before_tools(state):
            return {"ready": True}

        def tools(state):
            return {"tool_result": "done"}

        graph = StateGraph()
        graph.add_node("before_tools", before_tools)
        graph.add_node("tools", tools)
        graph.add_edge(START, "before_tools")
        graph.add_edge("before_tools", "tools")
        graph.add_edge("tools", END)

        app = graph.compile(interrupt_before=["tools"])

        with pytest.raises(GraphInterrupt):
            app.invoke({})

    def test_checkpoint_persistence(self):
        """Test that checkpoints save and restore state."""
        saver = InMemorySaver()

        def node_a(state):
            return {"count": state.get("count", 0) + 1}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test-thread"}}

        result = app.invoke({"count": 0}, config)
        assert result["count"] == 1

        # Load state from checkpoint
        snapshot = app.get_state(config)
        assert snapshot.values.get("count") == 1

    def test_update_state(self):
        """Test manual state update."""
        saver = InMemorySaver()

        def node_a(state):
            return {"x": state.get("x", 0) + 1}

        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test-thread"}}

        # Manual state update
        new_config = app.update_state(config, {"x": 100})
        snapshot = app.get_state(new_config)
        assert snapshot.values.get("x") == 100


# ---------------------------------------------------------------------------
# Superstep engine — regression tests for the rebuilt executor
# ---------------------------------------------------------------------------


class TestSuperstepEngine:
    def test_reducer_accumulates_through_graph(self):
        """Annotated[list, add] must accumulate across nodes, not overwrite."""
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            items: Annotated[list, operator.add]

        def n1(state):
            return {"items": [1]}

        def n2(state):
            return {"items": [2]}

        graph = StateGraph(State)
        graph.add_node("n1", n1)
        graph.add_node("n2", n2)
        graph.add_edge(START, "n1")
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", END)

        assert graph.compile().invoke({})["items"] == [1, 2]

    def test_scalar_reducer_through_graph(self):
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            total: Annotated[int, operator.add]

        graph = StateGraph(State)
        graph.add_node("a", lambda s: {"total": 5})
        graph.add_node("b", lambda s: {"total": 3})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        assert graph.compile().invoke({"total": 0})["total"] == 8

    def test_multi_edge_fanout_runs_all_targets(self):
        """Two edges out of one node must run BOTH successors."""
        order = []

        graph = StateGraph()
        graph.add_node("a", lambda s: order.append("a"))
        graph.add_node("b", lambda s: order.append("b"))
        graph.add_node("c", lambda s: order.append("c"))
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", END)
        graph.add_edge("c", END)

        graph.compile().invoke({})
        assert set(order) == {"a", "b", "c"}

    def test_diamond_join_runs_sink_once(self):
        """A→B, A→C, B→D, C→D must run D exactly once."""
        counts = {"d": 0}

        graph = StateGraph()
        graph.add_node("a", lambda s: None)
        graph.add_node("b", lambda s: None)
        graph.add_node("c", lambda s: None)

        def d(state):
            counts["d"] += 1

        graph.add_node("d", d)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")
        graph.add_edge("d", END)

        graph.compile().invoke({})
        assert counts["d"] == 1

    def test_recursion_limit_raises(self):
        """A graph that never terminates must raise GraphRecursionError."""

        graph = StateGraph()
        graph.add_node("loop", lambda s: {"n": s.get("n", 0) + 1})
        graph.add_edge(START, "loop")
        graph.add_conditional_edges("loop", lambda s: "loop")

        with pytest.raises(GraphRecursionError):
            graph.compile().invoke({})

    def test_recursion_limit_configurable(self):
        graph = StateGraph()
        graph.add_node("loop", lambda s: {"n": s.get("n", 0) + 1})
        graph.add_edge(START, "loop")
        graph.add_conditional_edges("loop", lambda s: "loop")

        app = graph.compile()
        with pytest.raises(GraphRecursionError):
            app.invoke({}, {"recursion_limit": 5})

    def test_interrupt_then_resume_continues(self):
        """Resume must continue from the saved frontier, not restart at START."""
        saver = InMemorySaver()
        calls = {"before": 0, "tools": 0}

        def before(state):
            calls["before"] += 1
            return {"ready": True}

        def tools(state):
            calls["tools"] += 1
            return {"tool_result": "done"}

        graph = StateGraph()
        graph.add_node("before", before)
        graph.add_node("tools", tools)
        graph.add_edge(START, "before")
        graph.add_edge("before", "tools")
        graph.add_edge("tools", END)

        app = graph.compile(checkpointer=saver, interrupt_before=["tools"])
        config = {"configurable": {"thread_id": "resume-1"}}

        with pytest.raises(GraphInterrupt):
            app.invoke({}, config)

        # The frontier was persisted: "tools" is what runs next.
        assert app.get_state(config).next == ["tools"]

        result = app.invoke(None, config)
        assert result["tool_result"] == "done"
        assert result["ready"] is True
        # No amnesia: "before" did not re-run from START on resume.
        assert calls["before"] == 1
        assert calls["tools"] == 1


# ---------------------------------------------------------------------------
# Prebuilt agent tests (with mock model)
# ---------------------------------------------------------------------------


class MockGraphModel:
    def __init__(self, responses=None):
        self.responses = list(responses or [AIMessage(content="I can help with that.")])
        self.call_count = 0

    def generate(self, messages, *, tools=None, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Done.")


def test_graph_agent_simple():
    from infy.graph.prebuilt import create_graph_agent

    model = MockGraphModel([AIMessage(content="Hello from graph agent!")])
    agent = create_graph_agent(model)
    result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_graph_agent_with_tools():
    from infy.graph.prebuilt import create_graph_agent
    from infy.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: sunny"

    model = MockGraphModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}],
            ),
            AIMessage(content="It's sunny in Paris!"),
        ]
    )

    agent = create_graph_agent(model, tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")]})

    # Should have: user, ai(tool_call), tool, ai(final)
    assert len(result["messages"]) >= 3


def test_graph_agent_with_system_prompt():
    from infy.graph.prebuilt import create_graph_agent

    model = MockGraphModel([AIMessage(content="Arrr!")])
    agent = create_graph_agent(model, system_prompt="You are a pirate.")
    result = agent.invoke({"messages": [HumanMessage(content="Hi")]})

    msgs = result["messages"]
    # System message should be first
    assert isinstance(msgs[0], SystemMessage)
    assert "pirate" in msgs[0].content


def test_graph_agent_with_checkpoint():
    from infy.graph.prebuilt import create_graph_agent

    saver = InMemorySaver()
    model = MockGraphModel(
        [
            AIMessage(content="First response"),
            AIMessage(content="Second response"),
        ]
    )

    agent = create_graph_agent(model, checkpointer=saver)
    config = {"configurable": {"thread_id": "test"}}

    result1 = agent.invoke({"messages": [HumanMessage(content="First")]}, config)
    assert len(result1["messages"]) >= 2

    # Check state was saved
    state = agent.get_state(config)
    assert state is not None


# ---------------------------------------------------------------------------
# Integration: full graph lifecycle
# ---------------------------------------------------------------------------


class TestFullGraphLifecycle:
    def test_complete_lifecycle(self):
        """Test: build → compile → invoke → get_state → update_state → invoke again."""
        saver = InMemorySaver()

        def agent_node(state):
            count = state.get("turns", 0) + 1
            return {"turns": count, "last_action": "agent"}

        def check_node(state):
            if state.get("turns", 0) >= 3:
                return {"done": True}
            return {"last_action": "check"}

        def route(state):
            if state.get("done"):
                return END
            return "check"

        graph = StateGraph()
        graph.add_node("agent", agent_node)
        graph.add_node("check", check_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", route, {END: END, "check": "check"})
        graph.add_edge("check", "agent")

        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "lifecycle-test"}}

        result = app.invoke({"turns": 0}, config)
        assert result.get("done") is True
        assert result.get("turns", 0) >= 3

        # State should be persisted
        snapshot = app.get_state(config)
        assert snapshot.values.get("done") is True


# ---------------------------------------------------------------------------
# Async graph execution (ainvoke / astream)
# ---------------------------------------------------------------------------


class TestGraphAsync:
    @staticmethod
    def _diamond(node_a, node_b, node_c):
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            count: Annotated[int, operator.add]
            log: Annotated[list, operator.add]

        graph = StateGraph(State)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")  # fan-out
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")  # diamond join
        graph.add_edge("c", END)
        return graph.compile()

    async def test_ainvoke_matches_invoke_for_sync_nodes(self):
        """Sync nodes through both paths must produce identical results."""
        app = self._diamond(
            lambda s: {"count": 1, "log": ["a"]},
            lambda s: {"count": 10, "log": ["b"]},
            lambda s: {"count": 100, "log": ["c"]},
        )
        sync = app.invoke({"count": 0, "log": []})
        asyncr = await app.ainvoke({"count": 0, "log": []})
        assert sync == asyncr == {"count": 111, "log": ["a", "b", "c"]}

    async def test_ainvoke_awaits_async_nodes(self):
        """Async node callables are awaited; fan-out + diamond + reducers stay correct."""

        async def a(s):
            return {"count": 1, "log": ["a"]}

        async def b(s):
            return {"count": 10, "log": ["b"]}

        async def c(s):
            return {"count": 100, "log": ["c"]}

        app = self._diamond(a, b, c)
        result = await app.ainvoke({"count": 0, "log": []})
        assert result == {"count": 111, "log": ["a", "b", "c"]}

    async def test_ainvoke_mixes_sync_and_async_nodes(self):
        async def a(s):
            return {"count": 1, "log": ["a"]}

        def b(s):  # sync node in an async graph
            return {"count": 10, "log": ["b"]}

        async def c(s):
            return {"count": 100, "log": ["c"]}

        app = self._diamond(a, b, c)
        result = await app.ainvoke({"count": 0, "log": []})
        assert result == {"count": 111, "log": ["a", "b", "c"]}

    async def test_astream_yields_each_superstep(self):
        async def a(s):
            return {"count": 1, "log": ["a"]}

        async def b(s):
            return {"count": 10, "log": ["b"]}

        async def c(s):
            return {"count": 100, "log": ["c"]}

        app = self._diamond(a, b, c)
        states = [s async for s in app.astream({"count": 0, "log": []})]
        assert states == [
            {"count": 11, "log": ["a", "b"]},
            {"count": 111, "log": ["a", "b", "c"]},
        ]

    async def test_ainvoke_wraps_node_errors(self):
        async def boom(s):
            raise ValueError("kaboom")

        graph = StateGraph()
        graph.add_node("x", boom)
        graph.add_edge(START, "x")
        graph.add_edge("x", END)
        app = graph.compile()
        with pytest.raises(NodeError):
            await app.ainvoke({})

    async def test_ainvoke_enforces_recursion_limit(self):
        def loop(state):
            return {"n": state.get("n", 0) + 1}

        graph = StateGraph()
        graph.add_node("loop", loop)
        graph.add_edge(START, "loop")
        graph.add_edge("loop", "loop")  # never terminates
        app = graph.compile(recursion_limit=5)
        with pytest.raises(GraphRecursionError):
            await app.ainvoke({})

    async def test_ainvoke_conditional_routing(self):
        def decide(state):
            return "fast" if state.get("fast") else "slow"

        graph = StateGraph()
        graph.add_node("decide", lambda s: {})
        graph.add_node("fast", lambda s: {"result": "fast path"})
        graph.add_node("slow", lambda s: {"result": "slow path"})
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", decide, {"fast": "fast", "slow": "slow"})
        graph.add_edge("fast", END)
        graph.add_edge("slow", END)
        app = graph.compile()
        assert (await app.ainvoke({"fast": True}))["result"] == "fast path"
        assert (await app.ainvoke({"fast": False}))["result"] == "slow path"

    async def test_ainvoke_interrupt_before_and_resume(self):
        """interrupt_before pauses, checkpoint persists the frontier, resume continues."""
        graph = StateGraph()
        graph.add_node("n1", lambda s: {"x": s.get("x", 0) + 1})
        graph.add_node("n2", lambda s: {"x": s.get("x", 0) + 10})
        graph.add_edge(START, "n1")
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", END)
        app = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["n2"])
        config = {"configurable": {"thread_id": "async-interrupt"}}

        with pytest.raises(GraphInterrupt):
            await app.ainvoke({"x": 0}, config)
        snap = app.get_state(config)
        assert snap.values == {"x": 1}
        assert snap.next == ["n2"]

        assert await app.ainvoke(None, config) == {"x": 11}

    async def test_ainvoke_send_dynamic_fanout(self):
        """Async Send fan-out runs one worker per Send, distinct, results reduced in order."""
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            results: Annotated[list, operator.add]

        async def worker(s):
            return {"results": [s["item"] * 10]}

        graph = StateGraph(State)
        graph.add_node("fanout", lambda s: {})
        graph.add_node("worker", worker)
        graph.add_edge(START, "fanout")
        graph.add_conditional_edges(
            "fanout", lambda s: [Send("worker", {"item": i}) for i in (1, 2, 3)]
        )
        graph.add_edge("worker", END)
        app = graph.compile()
        assert (await app.ainvoke({"results": []})) == {"results": [10, 20, 30]}

    async def test_ainvoke_interrupt_after_then_resume(self):
        """interrupt_after commits the node's writes, persists next_frontier, resumes once."""
        graph = StateGraph()
        graph.add_node("a", lambda s: {"log": "a"})
        graph.add_node("b", lambda s: {"log": s.get("log", "") + "b"})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile(checkpointer=InMemorySaver(), interrupt_after=["a"])
        config = {"configurable": {"thread_id": "async-after"}}

        with pytest.raises(GraphInterrupt):
            await app.ainvoke({}, config)
        snap = app.get_state(config)
        assert snap.values["log"] == "a"  # a's writes committed before pausing
        assert snap.next == ["b"]
        assert (await app.ainvoke(None, config))["log"] == "ab"  # b runs once, a not re-run

    async def test_ainvoke_in_node_interrupt_checkpoints_once(self):
        """A node raising NodeInterrupt pauses + persists exactly one checkpoint, even when
        a sibling in the same frontier also interrupts."""
        from infy.graph.errors import NodeInterrupt

        class CountingSaver(InMemorySaver):
            def __init__(self):
                super().__init__()
                self.puts = 0

            def put(self, *a, **k):
                self.puts += 1
                return super().put(*a, **k)

        def boom(s):
            raise NodeInterrupt("approve?")

        graph = StateGraph()
        graph.add_node("x", boom)
        graph.add_node("y", boom)
        graph.add_edge(START, "x")
        graph.add_edge(START, "y")
        graph.add_edge("x", END)
        graph.add_edge("y", END)
        saver = CountingSaver()
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "node-int"}}

        with pytest.raises(GraphInterrupt):
            await app.ainvoke({}, config)
        assert saver.puts == 1  # one checkpoint despite two interrupting siblings
        assert app.get_state(config).next  # resumable frontier persisted

    async def test_ainvoke_conditional_returns_list(self):
        """A conditional that returns a LIST of node names fans out to all of them."""
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            seen: Annotated[list, operator.add]

        graph = StateGraph(State)
        graph.add_node("split", lambda s: {})
        graph.add_node("x", lambda s: {"seen": ["x"]})
        graph.add_node("y", lambda s: {"seen": ["y"]})
        graph.add_edge(START, "split")
        graph.add_conditional_edges("split", lambda s: ["x", "y"])
        graph.add_edge("x", END)
        graph.add_edge("y", END)
        app = graph.compile()
        assert sorted((await app.ainvoke({"seen": []}))["seen"]) == ["x", "y"]

    async def test_astream_runs_frontier_concurrently(self):
        """Two same-superstep async nodes must overlap — a handshake that deadlocks if serial."""
        import asyncio
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            order: Annotated[list, operator.add]

        both_started = asyncio.Event()
        started = {"n": 0}

        async def make(tag):
            started["n"] += 1
            if started["n"] == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)  # serial -> TimeoutError
            return {"order": [tag]}

        graph = StateGraph(State)
        graph.add_node("a", lambda s: make("a"))
        graph.add_node("b", lambda s: make("b"))
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("a", END)
        graph.add_edge("b", END)
        app = graph.compile()
        # Frontier-order write fold despite genuine overlap.
        assert (await app.ainvoke({"order": []}))["order"] == ["a", "b"]

    async def test_ainvoke_persists_state(self):
        graph = StateGraph()
        graph.add_node("inc", lambda s: {"count": s.get("count", 0) + 1})
        graph.add_edge(START, "inc")
        graph.add_edge("inc", END)
        app = graph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "persist"}}
        assert (await app.ainvoke({"count": 0}, config))["count"] == 1
        assert app.get_state(config).values["count"] == 1

    def test_interrupt_after_then_resume_sync(self):
        """Sync companion: interrupt_after branch (previously untested in sync too)."""
        graph = StateGraph()
        graph.add_node("a", lambda s: {"log": "a"})
        graph.add_node("b", lambda s: {"log": s.get("log", "") + "b"})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile(checkpointer=InMemorySaver(), interrupt_after=["a"])
        config = {"configurable": {"thread_id": "sync-after"}}
        with pytest.raises(GraphInterrupt):
            app.invoke({}, config)
        assert app.get_state(config).next == ["b"]
        assert app.invoke(None, config)["log"] == "ab"
