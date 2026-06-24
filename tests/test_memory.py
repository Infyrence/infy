"""Tests for infy memory system."""

from infy.memory import BufferMemory, Memory, SummaryMemory, TokenLimitedMemory
from infy.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_buffer_memory_implements_protocol():
    mem = BufferMemory()
    assert isinstance(mem, Memory)


def test_token_limited_memory_implements_protocol():
    mem = TokenLimitedMemory()
    assert isinstance(mem, Memory)


# ---------------------------------------------------------------------------
# BufferMemory
# ---------------------------------------------------------------------------


class TestBufferMemory:
    def test_empty(self):
        mem = BufferMemory()
        assert mem.load_memory_variables() == []
        assert mem.message_count == 0

    def test_save_and_load(self):
        mem = BufferMemory()
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello!"

    def test_multiple_exchanges(self):
        mem = BufferMemory()
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        mem.save_context(HumanMessage(content="How are you?"), AIMessage(content="Good!"))
        assert mem.message_count == 4

    def test_with_system_prompt(self):
        mem = BufferMemory(system_prompt="You are helpful.")
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are helpful."

    def test_clear(self):
        mem = BufferMemory()
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        mem.clear()
        assert mem.message_count == 0
        assert mem.load_memory_variables() == []

    def test_preserves_order(self):
        mem = BufferMemory()
        for i in range(5):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        messages = mem.load_memory_variables()
        assert messages[0].content == "Q0"
        assert messages[-1].content == "A4"


# ---------------------------------------------------------------------------
# TokenLimitedMemory
# ---------------------------------------------------------------------------


class TestTokenLimitedMemory:
    def test_empty(self):
        mem = TokenLimitedMemory(max_tokens=1000)
        assert mem.load_memory_variables() == []

    def test_fits_within_budget(self):
        mem = TokenLimitedMemory(max_tokens=1000)
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
        assert len(messages) == 2

    def test_trims_oldest(self):
        mem = TokenLimitedMemory(max_tokens=50)
        # Add many messages to exceed budget
        for i in range(20):
            mem.save_context(
                HumanMessage(content=f"Question {i} with some extra text to make it longer"),
                AIMessage(content=f"Answer {i} with some extra text to make it longer"),
            )
        messages = mem.load_memory_variables()
        # Should have trimmed some old messages
        assert len(messages) < 40
        # Most recent messages should be present
        assert messages[-1].content.startswith("Answer 19")

    def test_with_system_prompt_fits(self):
        mem = TokenLimitedMemory(max_tokens=1000, system_prompt="Be helpful.")
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Be helpful."

    def test_system_prompt_eats_budget(self):
        mem = TokenLimitedMemory(
            max_tokens=10, system_prompt="A very long system prompt that uses most of the budget"
        )
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
        # System prompt takes most of the budget, few messages fit
        assert len(messages) <= 3

    def test_clear(self):
        mem = TokenLimitedMemory(max_tokens=1000)
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        mem.clear()
        assert mem.message_count == 0

    def test_estimated_tokens(self):
        mem = TokenLimitedMemory(max_tokens=1000)
        assert mem.estimated_tokens == 0
        mem.save_context(HumanMessage(content="Hello world"), AIMessage(content="Hi there"))
        assert mem.estimated_tokens > 0


# ---------------------------------------------------------------------------
# SummaryMemory
# ---------------------------------------------------------------------------


class TestSummaryMemory:
    def test_empty(self):
        mem = SummaryMemory(buffer_size=4)
        messages = mem.load_memory_variables()
        assert messages == []

    def test_within_buffer_no_summary(self):
        mem = SummaryMemory(buffer_size=6)
        for i in range(3):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        messages = mem.load_memory_variables()
        # All 6 messages should be present (no summarization needed)
        assert len(messages) == 6

    def test_exceeds_buffer_no_model(self):
        mem = SummaryMemory(buffer_size=4)
        for i in range(5):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        messages = mem.load_memory_variables()
        # Without a model, it just trims to buffer_size
        assert len(messages) == 4
        # Should keep the most recent
        assert messages[0].content == "Q3"

    def test_with_system_prompt(self):
        mem = SummaryMemory(buffer_size=4, system_prompt="Be concise.")
        mem.save_context(HumanMessage(content="Hi"), AIMessage(content="Hello!"))
        messages = mem.load_memory_variables()
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Be concise."

    def test_clear_resets_summary(self):
        mem = SummaryMemory(buffer_size=4)
        for i in range(5):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        mem.clear()
        assert mem.message_count == 0
        assert mem.summary == ""

    def test_with_mock_model(self):
        """Test summarization with a mock model."""

        class MockModel:
            def generate(self, messages):
                return AIMessage(content="Summary of conversation.")

        mem = SummaryMemory(model=MockModel(), buffer_size=4)
        for i in range(5):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))

        messages = mem.load_memory_variables()
        # Should have system message with summary + recent messages
        summary_msgs = [
            m for m in messages if isinstance(m, SystemMessage) and "summary" in m.content.lower()
        ]
        assert len(summary_msgs) == 1
        assert "Summary of conversation" in summary_msgs[0].content

    def test_summary_updates(self):
        """Test that summary gets updated as more messages arrive."""

        class MockModel:
            def __init__(self):
                self.call_count = 0

            def generate(self, messages):
                self.call_count += 1
                return AIMessage(content=f"Updated summary #{self.call_count}")

        model = MockModel()
        mem = SummaryMemory(model=model, buffer_size=2)

        # First batch — Q0/A0, Q1/A1, Q2/A2 (6 messages, buffer=2, triggers summarization)
        for i in range(3):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))

        # Summary was called at least once (when messages exceeded buffer)
        assert model.call_count >= 1
        first_summary_call = model.call_count

        # Second batch — triggers more summarization
        for i in range(3, 6):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))

        # More summaries were created
        assert model.call_count > first_summary_call

        messages = mem.load_memory_variables()
        # Should have system message with summary + recent messages
        summary_msgs = [
            m for m in messages if isinstance(m, SystemMessage) and "summary" in m.content.lower()
        ]
        assert len(summary_msgs) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_buffer_memory_single_message(self):
        mem = BufferMemory()
        mem.save_context(HumanMessage(content="Only input"), AIMessage(content="Only output"))
        assert mem.message_count == 2

    def test_token_limited_very_small_budget(self):
        mem = TokenLimitedMemory(max_tokens=1)
        mem.save_context(HumanMessage(content="Hello"), AIMessage(content="Hi"))
        messages = mem.load_memory_variables()
        # Very small budget — might only keep system prompt
        assert len(messages) <= 2

    def test_summary_memory_with_failed_model(self):
        class FailModel:
            def generate(self, messages):
                raise RuntimeError("API error")

        mem = SummaryMemory(model=FailModel(), buffer_size=4)
        for i in range(5):
            mem.save_context(HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        # Should gracefully handle failure — just trim
        messages = mem.load_memory_variables()
        assert len(messages) == 4
