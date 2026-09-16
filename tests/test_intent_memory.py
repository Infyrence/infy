"""Intent anchoring, fact-sheet memory, and bi-temporal facts."""

from __future__ import annotations

import dataclasses

import pytest

from infy.agents import create_agent, create_async_agent
from infy.intent import ANCHOR_PREFIX, Objective, is_anchor
from infy.memory import FactSheetMemory
from infy.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from infy.temporal import FOREVER, BiTemporalMemory, Fact
from infy.tools import tool

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def test_objective_rejects_an_empty_goal():
    with pytest.raises(ValueError):
        Objective(goal="   ")


def test_objective_normalises_lists_to_tuples():
    o = Objective(goal="g", constraints=["a"], success_criteria=["b"])
    assert isinstance(o.constraints, tuple) and isinstance(o.success_criteria, tuple)


def test_objective_is_frozen():
    o = Objective(goal="g")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.goal = "other"  # type: ignore[misc]


def test_render_includes_everything_and_reminder_is_shorter():
    o = Objective(goal="Ship 1.2", constraints=["No prod"], success_criteria=["Smoke passes"])
    rendered, reminder = o.render(), o.reminder()
    assert "Ship 1.2" in rendered and "No prod" in rendered and "Smoke passes" in rendered
    assert "Ship 1.2" in reminder and "No prod" in reminder
    # The criteria are the part deliberately not repeated every turn.
    assert "Smoke passes" not in reminder
    assert len(reminder) < len(rendered) + 200


def test_is_anchor_distinguishes_scaffolding_from_conversation():
    o = Objective(goal="g")
    assert is_anchor(SystemMessage(content=o.render()))
    assert is_anchor(SystemMessage(content=o.reminder()))
    assert not is_anchor(HumanMessage(content="hello"))


# ---------------------------------------------------------------------------
# Objective wired into the agent loop
# ---------------------------------------------------------------------------


def text_of(message) -> str:
    """ToolMessage carries `content`, everything else exposes `text`."""
    return getattr(message, "text", None) or str(getattr(message, "content", ""))


@tool
def look(thing: str) -> str:
    """Look something up."""
    return "found it"


class LoopModel:
    """Calls a tool for `tool_turns` turns, then answers. Records each prompt it saw."""

    model_name = "loop"

    def __init__(self, tool_turns: int = 2):
        self.tool_turns = tool_turns
        self.seen: list[list] = []

    def _next(self, messages):
        self.seen.append(list(messages))
        calls = sum(1 for m in messages if isinstance(m, ToolMessage))
        if calls >= self.tool_turns:
            return AIMessage(content="done")
        return AIMessage(
            content="", tool_calls=[ToolCall(name="look", args={"thing": "x"}, id=f"c{calls}")]
        )

    def generate(self, messages, *, tools=None, **kwargs):
        return self._next(messages)

    async def agenerate(self, messages, *, tools=None, **kwargs):
        return self._next(messages)


def test_objective_leads_the_prompt():
    model = LoopModel(tool_turns=0)
    o = Objective(goal="Ship 1.2")
    create_agent(model, [look], system_prompt="Be terse.", objective=o)("go")
    first = model.seen[0]
    assert first[0].text == o.render(), "objective must precede the system prompt"
    assert first[1].text == "Be terse."


def test_no_reminder_on_the_first_turn():
    model = LoopModel(tool_turns=0)
    create_agent(model, [look], objective=Objective(goal="g"))("go")
    assert sum(1 for m in model.seen[0] if ANCHOR_PREFIX in m.text and "unchanged" in m.text) == 0


def test_reminder_is_last_on_later_turns():
    model = LoopModel(tool_turns=2)
    create_agent(model, [look], objective=Objective(goal="g"), parallel_tools=False)("go")
    assert len(model.seen) >= 2
    for prompt in model.seen[1:]:
        assert "unchanged" in prompt[-1].text, "reminder must be the last thing the model reads"


def test_exactly_one_reminder_survives_no_matter_how_long_the_run():
    model = LoopModel(tool_turns=4)
    result = create_agent(
        model, [look], objective=Objective(goal="g"), parallel_tools=False, max_iterations=8
    )("go")
    for prompt in model.seen:
        assert sum(1 for m in prompt if "unchanged" in text_of(m)) <= 1
    assert sum(1 for m in result.messages if "unchanged" in text_of(m)) <= 1


def test_loop_is_untouched_without_an_objective():
    model = LoopModel(tool_turns=1)
    create_agent(model, [look], system_prompt="Be terse.", parallel_tools=False)("go")
    for prompt in model.seen:
        assert not any(is_anchor(m) for m in prompt)


async def test_async_agent_anchors_identically():
    sync_model, async_model = LoopModel(tool_turns=2), LoopModel(tool_turns=2)
    o = Objective(goal="g", constraints=["c"])
    create_agent(sync_model, [look], objective=o, parallel_tools=False)("go")
    await create_async_agent(async_model, [look], objective=o, parallel_tools=False)("go")
    assert [[text_of(m) for m in p] for p in sync_model.seen] == [
        [text_of(m) for m in p] for p in async_model.seen
    ]


# ---------------------------------------------------------------------------
# FactSheetMemory
# ---------------------------------------------------------------------------


def _fill(mem: FactSheetMemory, n: int) -> None:
    for i in range(n):
        mem.save_context(
            HumanMessage(content=f"q{i}"),
            ToolMessage(content=f"result {i}", tool_call_id=f"c{i}"),
        )


def test_recent_turns_stay_verbatim():
    mem = FactSheetMemory(buffer_size=4)
    _fill(mem, 5)
    assert len(mem.messages) <= 4
    assert mem.messages[-1].content == "result 4"


def test_older_tool_output_becomes_facts_without_a_model():
    mem = FactSheetMemory(buffer_size=2)
    _fill(mem, 4)
    assert mem.facts, "observations must survive compaction"
    assert any("result 0" in f for f in mem.facts)


def test_facts_are_deduped():
    mem = FactSheetMemory(buffer_size=2)
    for _ in range(4):
        mem.save_context(HumanMessage(content="q"), ToolMessage(content="same", tool_call_id="c"))
    assert mem.facts.count("same") == 1


def test_max_facts_bounds_the_sheet_dropping_oldest():
    mem = FactSheetMemory(buffer_size=2, max_facts=3)
    for i in range(10):
        mem.add_fact(f"fact {i}")
    assert len(mem.facts) == 3
    assert mem.facts == ["fact 7", "fact 8", "fact 9"]


def test_load_puts_the_sheet_ahead_of_recent_turns():
    mem = FactSheetMemory(buffer_size=2, system_prompt="sys")
    _fill(mem, 4)
    loaded = mem.load_memory_variables()
    assert loaded[0].text == "sys"
    assert "Established so far" in loaded[1].text
    assert loaded[-1].content == "result 3"


def test_model_extraction_is_used_when_available():
    class Extractor:
        model_name = "x"

        def generate(self, messages, **kwargs):
            return AIMessage(content="- the port is 8787\n- the build is green")

    mem = FactSheetMemory(model=Extractor(), buffer_size=2)
    _fill(mem, 4)
    assert "the port is 8787" in mem.facts


def test_extraction_failure_falls_back_instead_of_losing_observations():
    class Broken:
        model_name = "x"

        def generate(self, messages, **kwargs):
            raise RuntimeError("provider down")

    mem = FactSheetMemory(model=Broken(), buffer_size=2)
    _fill(mem, 4)
    assert mem.facts, "a failed extraction must not silently drop the observations"


def test_clear_resets_both_halves():
    mem = FactSheetMemory(buffer_size=2)
    _fill(mem, 4)
    mem.clear()
    assert mem.facts == [] and mem.messages == []


# ---------------------------------------------------------------------------
# BiTemporalMemory
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, by: float = 10.0) -> float:
        self.t += by
        return self.t


def test_assert_then_read_back():
    mem = BiTemporalMemory(clock=Clock())
    mem.assert_fact("user", "lives_in", "New York")
    assert mem.get("user", "lives_in") == "New York"


def test_reasserting_the_same_value_is_a_noop():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    first = mem.assert_fact("user", "lives_in", "New York")
    clock.tick()
    again = mem.assert_fact("user", "lives_in", "New York")
    assert first is again
    assert len(mem.history("user", "lives_in")) == 1


def test_the_world_changing_keeps_the_old_fact_true_for_its_interval():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "lives_in", "New York")
    t_before_move = clock.tick()
    moved_at = clock.tick()
    mem.assert_fact("user", "lives_in", "London")

    assert mem.get("user", "lives_in") == "London"
    was = mem.as_of(valid_time=t_before_move)
    assert [f.object for f in was] == ["New York"], "history must not be rewritten"
    assert moved_at  # the move time is the boundary, asserted below


def test_intervals_are_half_open_so_no_instant_is_double_counted():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "lives_in", "New York")
    boundary = clock.tick()
    mem.assert_fact("user", "lives_in", "London")
    at_boundary = [f.object for f in mem.as_of(valid_time=boundary)]
    assert at_boundary == ["London"], f"exactly one fact holds at the boundary, got {at_boundary}"


def test_retract_closes_validity_without_erasing_the_past():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "employer", "Acme")
    before = clock.tick()
    clock.tick()
    mem.retract("user", "employer")

    assert mem.get("user", "employer") is None
    assert [f.object for f in mem.as_of(valid_time=before)] == ["Acme"]


def test_retract_on_an_unknown_identity_is_a_noop():
    assert BiTemporalMemory(clock=Clock()).retract("nobody", "nothing") is None


def test_correct_removes_a_mistake_from_every_valid_time():
    """A correction says the fact was never true — unlike a retraction."""
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "employer", "Acme")
    during = clock.tick()
    clock.tick()
    mem.correct("user", "employer", "Globex")

    assert mem.get("user", "employer") == "Globex"
    # The mistaken value must not answer a valid-time query from inside its old interval.
    assert [f.object for f in mem.as_of(valid_time=during)] == ["Globex"]


def test_a_correction_is_still_on_the_record():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "employer", "Acme")
    clock.tick()
    mem.correct("user", "employer", "Globex")
    history = mem.history("user", "employer")
    assert [f.object for f in history] == ["Acme", "Globex"]
    assert not history[0].believed, "the mistake is superseded, not deleted"


def test_transaction_time_reconstructs_what_we_believed_when_we_acted():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "employer", "Acme")
    acted_at = clock.tick()
    clock.tick()
    mem.correct("user", "employer", "Globex")

    believed_then = [f.object for f in mem.as_of(transaction_time=acted_at)]
    assert believed_then == ["Acme"], "the agent acted on Acme; the audit must show that"
    assert [f.object for f in mem.as_of()] == ["Globex"]


def test_current_and_render_only_show_live_facts():
    clock = Clock()
    mem = BiTemporalMemory(clock=clock)
    mem.assert_fact("user", "lives_in", "New York")
    mem.assert_fact("user", "role", "engineer")
    clock.tick()
    mem.retract("user", "role")
    assert [f.predicate for f in mem.current()] == ["lives_in"]
    rendered = mem.render()
    assert "New York" in rendered and "engineer" not in rendered


def test_render_is_empty_when_nothing_is_known():
    assert BiTemporalMemory(clock=Clock()).render() == ""


def test_subjects_do_not_collide():
    mem = BiTemporalMemory(clock=Clock())
    mem.assert_fact("alice", "lives_in", "Paris")
    mem.assert_fact("bob", "lives_in", "Berlin")
    assert mem.get("alice", "lives_in") == "Paris"
    assert [f.subject for f in mem.current(subject="bob")] == ["bob"]


def test_fact_flags():
    live = Fact(subject="s", predicate="p", object=1)
    assert live.current and live.believed
    assert Fact(subject="s", predicate="p", object=1, valid_to=5.0).believed
    assert not Fact(subject="s", predicate="p", object=1, valid_to=5.0).current
    assert not Fact(subject="s", predicate="p", object=1, superseded_at=5.0).believed
    assert live.holds_at(0.0) and live.valid_to == FOREVER
