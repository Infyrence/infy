"""Adversarial / edge-case tests.

These exist to catch the "green but broken" failure mode: every test asserts on
an error path, a boundary, or a behavioral invariant rather than just that code
runs. Several deliberately force the pure-Python fallback and compare it against
the Rust core, which is where silent divergence (e.g. the SIMD tail bug) hides.
"""

import math

import pytest

from infy import similarity, tokens
from infy.agents import create_agent
from infy.documents import Document
from infy.messages import AIMessage, AIMessageChunk, TextBlock, ToolCall, ToolCallChunk, ToolMessage
from infy.parsers import JsonParser, StrParser
from infy.tools import tool
from infy.vectorstores import InMemoryVectorStore

# ---------------------------------------------------------------------------
# Rust <-> Python parity (cosine similarity)
# ---------------------------------------------------------------------------


def _ref_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na * nb == 0 else dot / (na * nb)


# Lengths chosen to include non-multiples of 8 (the SIMD tail the old Rust
# kernel double-counted on non-x86 builds).
PARITY_VECTORS = [
    [1.0, 2.0, 3.0],
    [1.0] * 9,
    [float(i) for i in range(17)],
    [float(i) - 8.0 for i in range(33)],
]


class TestCosineParity:
    @pytest.mark.parametrize("vec", PARITY_VECTORS)
    def test_python_fallback_matches_reference(self, vec, monkeypatch):
        other = [v * 0.5 + 1.0 for v in vec]
        monkeypatch.setattr(similarity, "_HAS_RUST", False)
        assert similarity.cosine_similarity(vec, other) == pytest.approx(
            _ref_cosine(vec, other), abs=1e-5
        )

    @pytest.mark.parametrize("vec", PARITY_VECTORS)
    def test_default_path_matches_reference(self, vec):
        other = [v * 0.5 + 1.0 for v in vec]
        assert similarity.cosine_similarity(vec, other) == pytest.approx(
            _ref_cosine(vec, other), abs=1e-4
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            similarity.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_no_div_by_zero(self):
        assert similarity.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Token counting (both code paths stay sane)
# ---------------------------------------------------------------------------


class TestTokenCounting:
    @pytest.mark.parametrize("text", ["", "hello world", "a" * 200, "你好世界", "!?,.;:()"])
    def test_both_paths_positive(self, text, monkeypatch):
        rust = tokens.count_tokens(text)
        monkeypatch.setattr(tokens, "_HAS_RUST", False)
        py = tokens.count_tokens(text)
        assert rust >= 1
        assert py >= 1

    def test_empty_is_one_both_paths(self, monkeypatch):
        assert tokens.count_tokens("") == 1
        monkeypatch.setattr(tokens, "_HAS_RUST", False)
        assert tokens.count_tokens("") == 1


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class TestParsers:
    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            JsonParser().parse("not json {[")

    def test_partial_incomplete_never_raises(self):
        out = JsonParser().parse('{"a": 1, "b":', partial=True)
        assert out is None or isinstance(out, dict)

    def test_markdown_fenced(self):
        assert JsonParser().parse('```json\n{"x": [1, 2, 3]}\n```') == {"x": [1, 2, 3]}

    def test_stream_yields_complete_final(self):
        chunks = ['{"name":', ' "infy",', ' "n": 42}']
        results = list(JsonParser().stream(iter(chunks)))
        assert results[-1] == {"name": "infy", "n": 42}

    def test_str_parser_list_content(self):
        msg = AIMessage(content=[TextBlock(text="hel"), TextBlock(text="lo")])
        assert StrParser().parse(msg) == "hello"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestTools:
    def test_missing_required_raises(self):
        @tool
        def add(x: int, y: int) -> int:
            """Add."""
            return x + y

        with pytest.raises(ValueError):
            add.invoke({"x": 1})

    def test_unexpected_args_dropped(self):
        @tool
        def greet(name: str) -> str:
            """Greet."""
            return f"hi {name}"

        assert greet.invoke({"name": "a", "bogus": 99}) == "hi a"

    def test_malformed_json_string_falls_back_to_input(self):
        @tool
        def echo(input: str) -> str:
            """Echo."""
            return input

        assert echo.invoke("not json") == "not json"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class TestVectorStore:
    def test_mismatched_lengths_raise(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError):
            store.add_documents([Document(page_content="a")], [[1.0], [2.0]])

    def test_empty_search_returns_empty(self):
        assert InMemoryVectorStore().similarity_search([1.0, 2.0], k=5) == []

    def test_topk_exceeds_count(self):
        store = InMemoryVectorStore()
        store.add_documents([Document(page_content="a")], [[1.0, 0.0]])
        assert len(store.similarity_search([1.0, 0.0], k=10)) == 1


# ---------------------------------------------------------------------------
# Streaming chunk merge
# ---------------------------------------------------------------------------


class TestChunkMerge:
    def test_distinct_tool_indices_stay_separate(self):
        c1 = AIMessageChunk(
            tool_call_chunks=[ToolCallChunk(name="a", args='{"x":', id="1", index=0)]
        )
        c2 = AIMessageChunk(
            tool_call_chunks=[ToolCallChunk(name="b", args='{"y":', id="2", index=1)]
        )
        merged = AIMessageChunk() + c1 + c2
        assert len(merged.tool_call_chunks) == 2

    def test_same_index_accumulates_args(self):
        c1 = AIMessageChunk(
            tool_call_chunks=[ToolCallChunk(name="a", args='{"x":', id="1", index=0)]
        )
        c2 = AIMessageChunk(tool_call_chunks=[ToolCallChunk(args=" 1}", index=0)])
        merged = c1 + c2
        assert len(merged.tool_call_chunks) == 1
        assert merged.tool_call_chunks[0].args == '{"x": 1}'

    def test_materialize_bad_json_args_is_empty_dict(self):
        chunk = AIMessageChunk(
            tool_call_chunks=[ToolCallChunk(name="f", args="{not json", id="1", index=0)]
        )
        assert chunk.materialize().tool_calls[0].args == {}


# ---------------------------------------------------------------------------
# Agent tool dispatch (no orphaned tool calls, ever)
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0
        self.model_name = "fake"

    def generate(self, messages, *, tools=None, **kwargs):
        resp = self.responses[self.i] if self.i < len(self.responses) else AIMessage(content="done")
        self.i += 1
        return resp


class TestAgentDispatch:
    def test_parallel_unknown_tool_is_answered(self):
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[ToolCall(name="ghost", args={}, id="c1")]),
                AIMessage(content="ok"),
            ]
        )
        result = create_agent(model, tools=[], parallel_tools=True)("go")
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].status == "error"

    def test_parallel_duplicate_ids_both_execute(self):
        @tool
        def echo(x: str) -> str:
            """Echo."""
            return x

        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="echo", args={"x": "a"}, id=""),
                        ToolCall(name="echo", args={"x": "b"}, id=""),
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        result = create_agent(model, tools=[echo], parallel_tools=True)("go")
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        assert sorted(m.content for m in tool_msgs) == ["a", "b"]

    def test_tool_exception_becomes_error_message(self):
        @tool
        def boom(x: str) -> str:
            """Boom."""
            raise RuntimeError("kaboom")

        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[ToolCall(name="boom", args={"x": "1"}, id="c1")]),
                AIMessage(content="recovered"),
            ]
        )
        result = create_agent(model, tools=[boom])("go")
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        assert any("kaboom" in m.content for m in tool_msgs)
        assert result.response.text == "recovered"

    def test_respects_max_iterations(self):
        looping = [AIMessage(content="", tool_calls=[ToolCall(name="x", args={}, id="c")])] * 50
        result = create_agent(_FakeModel(looping), max_iterations=3)("go")
        assert result.iterations == 3


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_token_limited_never_exceeds_budget(self):
        from infy.memory import TokenLimitedMemory
        from infy.messages import HumanMessage

        mem = TokenLimitedMemory(max_tokens=30)
        for i in range(40):
            mem.save_context(HumanMessage(content=f"q{i} " * 5), AIMessage(content=f"a{i} " * 5))
        msgs = mem.load_memory_variables()
        total = sum(tokens.count_tokens(m.content) for m in msgs)
        assert total <= 30
        assert msgs  # most-recent messages that fit are kept

    def test_summary_memory_failing_model_degrades_gracefully(self):
        from infy.memory import SummaryMemory
        from infy.messages import HumanMessage

        class FailModel:
            def generate(self, messages):
                raise RuntimeError("api down")

        mem = SummaryMemory(model=FailModel(), buffer_size=4)
        for i in range(5):
            mem.save_context(HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}"))
        msgs = mem.load_memory_variables()
        assert len(msgs) == 4  # trimmed to the buffer, no crash


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------


class TestTextSplitter:
    def test_giant_token_is_bounded_and_lossless(self):
        from infy.text_splitter import TextSplitter

        chunks = TextSplitter(chunk_size=20, chunk_overlap=0).split_text("x" * 100)
        assert all(len(c) <= 20 for c in chunks)
        assert "".join(chunks) == "x" * 100

    def test_short_text_single_chunk(self):
        from infy.text_splitter import TextSplitter

        assert TextSplitter(chunk_size=1000).split_text("hi") == ["hi"]
