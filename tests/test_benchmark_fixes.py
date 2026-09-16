"""Tests for Gap fixes: Vertex AI flag + with_structured_output."""

import json

import pytest

from infy.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Mock model for testing
# ---------------------------------------------------------------------------


class MockStructuredModel:
    """Mock that returns JSON text matching a schema."""

    def __init__(self, response_json: dict):
        self.response_text = json.dumps(response_json)
        self.model_name = "mock"
        self.last_messages = None

    def generate(self, messages, *, tools=None, **kwargs):
        self.last_messages = messages
        return AIMessage(content=self.response_text)

    async def agenerate(self, messages, *, tools=None, **kwargs):
        self.last_messages = messages
        return AIMessage(content=self.response_text)

    def stream(self, messages, *, tools=None, **kwargs):
        self.last_messages = messages
        # Yield in chunks to simulate streaming
        yield AIMessageChunk(content=self.response_text)

    async def astream(self, messages, *, tools=None, **kwargs):
        self.last_messages = messages
        yield AIMessageChunk(content=self.response_text)

    def bind_tools(self, tools, **kwargs):
        from infy.models import BoundChatModel

        return BoundChatModel(model=self, bound_tools=tools, bound_kwargs=kwargs)

    def with_structured_output(self, schema, **kwargs):
        from infy.models import StructuredOutputModel

        return StructuredOutputModel(model=self, schema=schema)


# ---------------------------------------------------------------------------
# Gap 1: Vertex AI flag
# ---------------------------------------------------------------------------


class TestVertexAIFlag:
    def test_gemini_chat_has_vertexai_field(self):
        from infy.providers.gemini import GeminiChat

        m = GeminiChat("gemini-2.0-flash")
        assert m.vertexai is False

    def test_gemini_chat_vertexai_true(self):
        from infy.providers.gemini import GeminiChat

        m = GeminiChat("gemini-2.0-flash", vertexai=True)
        assert m.vertexai is True

    def test_gemini_chat_express_mode_flag(self):
        from infy.providers.gemini import GeminiChat

        m = GeminiChat("gemini-2.5-flash", api_key="test-key", vertexai=True)
        assert m.api_key == "test-key"
        assert m.vertexai is True
        assert m.model_name == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Gap 3: with_structured_output
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_basic_structured_output(self):
        model = MockStructuredModel({"name": "Alice", "age": 30})
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        structured = model.with_structured_output(schema)
        result = structured.generate([HumanMessage(content="Tell me about Alice")])

        assert result == {"name": "Alice", "age": 30}

    def test_structured_output_injects_schema_instruction(self):
        model = MockStructuredModel({"key": "value"})
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        structured = model.with_structured_output(schema)
        structured.generate([HumanMessage(content="test")])

        # Check that system message was prepended
        msgs = model.last_messages
        assert len(msgs) >= 2
        assert isinstance(msgs[0], SystemMessage)
        assert "schema" in msgs[0].content.lower()
        assert "json" in msgs[0].content.lower()
        assert "key" in msgs[0].content

    def test_structured_output_complex_schema(self):
        response = {
            "users": [
                {"name": "Alice", "role": "admin"},
                {"name": "Bob", "role": "user"},
            ],
            "total": 2,
        }
        model = MockStructuredModel(response)
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                        },
                    },
                },
                "total": {"type": "integer"},
            },
        }

        structured = model.with_structured_output(schema)
        result = structured.generate([HumanMessage(content="list users")])
        assert result["total"] == 2
        assert len(result["users"]) == 2

    def test_structured_output_preserves_original_messages(self):
        model = MockStructuredModel({"ok": True})
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

        structured = model.with_structured_output(schema)
        msgs = [HumanMessage(content="test")]
        structured.generate(msgs)

        # Original messages list should not be mutated
        assert len(msgs) == 1

    def test_structured_output_model_name(self):
        model = MockStructuredModel({"x": 1})
        structured = model.with_structured_output({"type": "object"})
        assert structured.model_name == "mock"

    def test_structured_output_on_bound_model(self):
        from infy.models import BoundChatModel, ToolSchema

        model = MockStructuredModel({"result": 42})
        bound = BoundChatModel(
            model=model,
            bound_tools=[ToolSchema(name="test", description="test")],
        )

        structured = bound.with_structured_output(
            {
                "type": "object",
                "properties": {"result": {"type": "integer"}},
            }
        )

        result = structured.generate([HumanMessage(content="compute")])
        assert result == {"result": 42}

    def test_structured_output_streaming(self):
        model = MockStructuredModel({"name": "test", "value": 42})
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        }

        structured = model.with_structured_output(schema)
        results = list(structured.stream([HumanMessage(content="test")]))
        # At least one partial result should be yielded
        assert len(results) >= 1

    def test_structured_output_async(self):
        import asyncio

        model = MockStructuredModel({"status": "active", "count": 5})
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
        }

        structured = model.with_structured_output(schema)
        result = asyncio.run(structured.agenerate([HumanMessage(content="get status")]))
        assert result == {"status": "active", "count": 5}


# ---------------------------------------------------------------------------
# Integration: structured output in pipeline
# ---------------------------------------------------------------------------


class TestStructuredOutputPipeline:
    def test_in_chain(self):
        """Test structured output works with pipe operator."""
        from infy import Lambda

        model = MockStructuredModel({"answer": "Paris", "confidence": 0.95})
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

        structured = model.with_structured_output(schema)

        # Use in a chain
        chain = Lambda(lambda q: [HumanMessage(content=q)]) | structured
        result = chain.invoke("What is the capital of France?")

        assert result["answer"] == "Paris"
        assert result["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Regression: the original tests only exercised a mock, so the gap that
# with_structured_output was missing from the REAL providers slipped through.
# These lock the wiring + the two bug fixes in place. All offline (class-level
# checks; only GeminiChat is constructed, and its import has no SDK requirement).
# ---------------------------------------------------------------------------


class TestProviderWiringRegression:
    def test_all_real_providers_define_with_structured_output(self):
        from infy.providers.anthropic import AnthropicChat
        from infy.providers.gemini import GeminiChat
        from infy.providers.ollama import OllamaChat
        from infy.providers.openai import OpenAIChat

        for cls in (GeminiChat, OpenAIChat, AnthropicChat, OllamaChat):
            assert hasattr(cls, "with_structured_output"), cls.__name__

    def test_gemini_with_structured_output_returns_wrapper(self):
        from infy.models import StructuredOutputModel
        from infy.providers.gemini import GeminiChat

        model = GeminiChat("gemini-2.5-flash")
        structured = model.with_structured_output({"type": "object"})
        assert isinstance(structured, StructuredOutputModel)
        assert structured.model is model

    def test_structured_bind_tools_preserves_tools(self):
        """Regression: bind_tools used to drop the tools, returning the bare model."""
        from infy.models import BoundChatModel, StructuredOutputModel, ToolSchema

        structured = MockStructuredModel({"x": 1}).with_structured_output({"type": "object"})
        rebound = structured.bind_tools([ToolSchema(name="t", description="d")])
        assert isinstance(rebound, StructuredOutputModel)
        assert isinstance(rebound.model, BoundChatModel)
        assert rebound.model.bound_tools[0].name == "t"

    def test_gemini_merges_multiple_system_messages(self):
        """Regression: an injected schema/system instruction must not be clobbered by a
        caller-supplied system message later in the list."""
        from infy.providers.gemini import GeminiChat

        msgs = [
            SystemMessage(content="SCHEMA-INSTRUCTION"),
            SystemMessage(content="USER-SYSTEM"),
            HumanMessage(content="hi"),
        ]
        _contents, system_text = GeminiChat("gemini-2.5-flash")._convert_messages(msgs)
        assert "SCHEMA-INSTRUCTION" in system_text
        assert "USER-SYSTEM" in system_text

    def test_anthropic_merges_multiple_system_messages(self):
        """Regression: Anthropic hoists system messages into a single top-level field, and
        used to keep only the last one — silently dropping the caller's system_prompt as soon
        as the agent loop injected a second (a tool summary pool, or an objective anchor)."""
        pytest.importorskip("anthropic")
        from infy.providers.anthropic import AnthropicChat

        msgs = [
            SystemMessage(content="USER-SYSTEM"),
            SystemMessage(content="INJECTED-ANCHOR"),
            HumanMessage(content="hi"),
        ]
        system_text, _msg_list = AnthropicChat("claude-sonnet-5")._convert_messages(msgs)
        assert "USER-SYSTEM" in system_text
        assert "INJECTED-ANCHOR" in system_text

    def test_all_providers_have_async_methods(self):
        """Regression: every provider must expose agenerate + astream (async graphs use them)."""
        from infy.providers.anthropic import AnthropicChat
        from infy.providers.gemini import GeminiChat
        from infy.providers.ollama import OllamaChat
        from infy.providers.openai import OpenAIChat

        for cls in (GeminiChat, OpenAIChat, AnthropicChat, OllamaChat):
            assert hasattr(cls, "agenerate"), cls.__name__
            assert hasattr(cls, "astream"), cls.__name__


# ---------------------------------------------------------------------------
# with_structured_output + pydantic: real validation via pydantic-core
# ---------------------------------------------------------------------------


class _JsonModel:
    """Fake model that echoes a fixed JSON string (sync + async)."""

    model_name = "fake"

    def __init__(self, json_text):
        self.json_text = json_text

    def generate(self, messages, **kw):
        return AIMessage(content=self.json_text)

    async def agenerate(self, messages, **kw):
        return AIMessage(content=self.json_text)


class TestStructuredOutputPydantic:
    def test_pydantic_schema_returns_validated_model(self):
        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        sm = StructuredOutputModel(model=_JsonModel('{"name": "Alice", "age": 30}'), schema=Person)
        out = sm.generate([HumanMessage(content="who?")])
        assert isinstance(out, Person)
        assert out.name == "Alice" and out.age == 30

    def test_pydantic_validation_coerces_types(self):
        """Real validation: pydantic-core coerces the string '42' to int 42."""
        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        sm = StructuredOutputModel(model=_JsonModel('{"name": "Bob", "age": "42"}'), schema=Person)
        out = sm.generate([HumanMessage(content="who?")])
        assert out.age == 42 and isinstance(out.age, int)

    def test_pydantic_handles_fenced_json(self):
        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        sm = StructuredOutputModel(
            model=_JsonModel('```json\n{"name": "Cara", "age": 5}\n```'), schema=Person
        )
        out = sm.generate([HumanMessage(content="who?")])
        assert isinstance(out, Person) and out.name == "Cara"

    def test_pydantic_ainvoke(self):
        import asyncio

        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        sm = StructuredOutputModel(model=_JsonModel('{"name": "Eve", "age": 7}'), schema=Person)
        out = asyncio.run(sm.ainvoke("who?"))
        assert isinstance(out, Person) and out.name == "Eve"

    def test_dict_schema_still_returns_dict(self):
        from infy.models import StructuredOutputModel

        sm = StructuredOutputModel(model=_JsonModel('{"name": "Dee"}'), schema={"type": "object"})
        out = sm.generate([HumanMessage(content="who?")])
        assert isinstance(out, dict) and out == {"name": "Dee"}

    def test_instruction_is_cached(self):
        from infy.models import StructuredOutputModel

        sm = StructuredOutputModel(
            model=_JsonModel('{"a": 1}'),
            schema={"type": "object", "properties": {"a": {"type": "integer"}}},
        )
        assert sm._instruction  # built once in __post_init__
        first = sm._instruction
        sm.generate([HumanMessage(content="x")])
        assert sm._instruction is first  # not rebuilt per call

    def test_pydantic_validation_error_propagates(self):
        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        bad = StructuredOutputModel(
            model=_JsonModel('{"name": "Bob", "age": "abc"}'), schema=Person
        )
        with pytest.raises(pydantic.ValidationError):
            bad.generate([HumanMessage(content="x")])

        missing = StructuredOutputModel(model=_JsonModel('{"name": "Bob"}'), schema=Person)
        with pytest.raises(pydantic.ValidationError):
            missing.generate([HumanMessage(content="x")])

    def test_pydantic_non_json_raises_validation_error(self):
        """Prose/refusal reply surfaces a clean ValidationError, not the parser's ValueError."""
        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        sm = StructuredOutputModel(model=_JsonModel("I cannot help with that."), schema=Person)
        with pytest.raises(pydantic.ValidationError):
            sm.generate([HumanMessage(content="x")])

    def test_dict_schema_without_pydantic(self, monkeypatch):
        """Zero-dependency: with pydantic unavailable, a dict schema still works and a class
        is not mistaken for a pydantic model."""
        import sys

        from infy.models import StructuredOutputModel, _as_pydantic_model

        monkeypatch.setitem(sys.modules, "pydantic", None)  # force ImportError on `import pydantic`

        class NotPydantic:
            pass

        assert _as_pydantic_model(NotPydantic) is None
        sm = StructuredOutputModel(model=_JsonModel('{"name": "Dee"}'), schema={"type": "object"})
        assert sm.generate([HumanMessage(content="x")]) == {"name": "Dee"}

    def test_pydantic_schema_streams_partial_dicts(self):
        """A pydantic schema still streams partial dicts (validation is generate-only)."""
        pydantic = pytest.importorskip("pydantic")
        from infy.models import StructuredOutputModel

        class Person(pydantic.BaseModel):
            name: str
            age: int

        class StreamModel:
            model_name = "fake"

            def stream(self, messages, **kw):
                yield AIMessage(content='{"name": "Al", "age": 9}')

        sm = StructuredOutputModel(model=StreamModel(), schema=Person)
        chunks = list(sm.stream([HumanMessage(content="x")]))
        assert chunks and all(isinstance(c, dict) for c in chunks)

    async def test_structured_output_in_async_pipe(self):
        from infy import Lambda

        model = MockStructuredModel({"answer": "Paris"})
        chain = Lambda(lambda q: [HumanMessage(content=q)]) | model.with_structured_output(
            {"type": "object"}
        )
        assert (await chain.ainvoke("capital?"))["answer"] == "Paris"
