"""Tool routing: ranking, the summary pool, and the agent-loop wiring."""

from __future__ import annotations

import pytest

from infy.agents import create_agent, create_async_agent
from infy.embeddings import FakeEmbeddings
from infy.messages import AIMessage, ToolCall, ToolMessage
from infy.tool_router import RRF_K, ToolRouter, reciprocal_rank_fusion
from infy.tools import Tool, tool

# ---------------------------------------------------------------------------
# A catalogue big enough that slicing is the point
# ---------------------------------------------------------------------------


@tool
def search_web(query: str) -> str:
    """Search the public web for pages matching a query."""
    return "web results"


@tool
def read_file(path: str) -> str:
    """Read the contents of a file from the local filesystem."""
    return "file contents"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file on the local filesystem."""
    return "written"


@tool
def run_shell(cmd: str) -> str:
    """Execute a shell command on the host machine."""
    return "command output"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email message to a recipient."""
    return "sent"


@tool
def query_database(sql: str) -> str:
    """Run a SQL query against the analytics database."""
    return "rows"


@tool
def deploy_service(name: str) -> str:
    """Deploy a service to the production cluster."""
    return "deployed"


@tool
def list_buckets() -> str:
    """List the object storage buckets in the account."""
    return "buckets"


@tool
def resize_image(path: str, width: int) -> str:
    """Resize an image to a target width in pixels."""
    return "resized"


@tool
def translate_text(text: str, language: str) -> str:
    """Translate text into another natural language."""
    return "translated"


CATALOGUE: list[Tool] = [
    search_web,
    read_file,
    write_file,
    run_shell,
    send_email,
    query_database,
    deploy_service,
    list_buckets,
    resize_image,
    translate_text,
]


def names(tools: list[Tool]) -> list[str]:
    return [t.name for t in tools]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_rrf_rewards_consensus_across_rankers():
    # 2 is mid-ranked in both lists; 0 and 9 are each top of one list but absent from the other.
    fused = reciprocal_rank_fusion([[0, 2, 1], [9, 2, 1]])
    order = [index for index, _ in fused]
    assert order[0] == 2


def test_rrf_scores_follow_the_formula():
    fused = dict(reciprocal_rank_fusion([[5, 7]], k=RRF_K))
    assert fused[5] == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[7] == pytest.approx(1.0 / (RRF_K + 2))


def test_rrf_breaks_ties_on_lower_index_deterministically():
    # Two rankings that are exact mirrors give every item the same total score.
    fused = reciprocal_rank_fusion([[3, 1], [1, 3]])
    assert [index for index, _ in fused] == [1, 3]


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([]) == []


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_lexical_selection_promotes_the_relevant_tool():
    router = ToolRouter(tools=CATALOGUE, top_k=3)
    selected = names(router.select("translate this paragraph into German"))
    assert "translate_text" in selected
    assert len(selected) <= 3


def test_selection_is_bounded_by_top_k():
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    assert len(router.select("read write file shell deploy email database")) <= 2


def test_zero_overlap_query_promotes_nothing_rather_than_padding():
    """A lexical miss must send no schemas, not an arbitrary top_k of them."""
    router = ToolRouter(tools=CATALOGUE, top_k=5)
    assert router.select("zzzz qqqq vvvv") == []


def test_always_is_pinned_even_when_irrelevant():
    router = ToolRouter(tools=CATALOGUE, top_k=2, always=["search_web"])
    assert "search_web" in names(router.select("resize the image"))


def test_always_ignores_a_name_that_is_not_in_the_catalogue():
    router = ToolRouter(tools=CATALOGUE, top_k=2, always=["not_a_tool"])
    assert "not_a_tool" not in names(router.select("resize the image"))


def test_small_catalogue_is_never_sliced():
    router = ToolRouter(tools=CATALOGUE[:3], top_k=5)
    assert names(router.select("zzzz")) == names(CATALOGUE[:3])


def test_empty_query_with_nothing_pinned_sends_a_bounded_prefix():
    router = ToolRouter(tools=CATALOGUE, top_k=4)
    assert names(router.select("")) == names(CATALOGUE[:4])


def test_selection_is_returned_in_catalogue_order_not_rank_order():
    router = ToolRouter(tools=CATALOGUE, top_k=3)
    selected = names(router.select("translate the file"))
    assert selected == [n for n in names(CATALOGUE) if n in selected]


def test_sticky_carries_previous_promotions_forward():
    router = ToolRouter(tools=CATALOGUE, top_k=2, sticky=True)
    assert "deploy_service" in names(router.select("resize image", promoted={"deploy_service"}))


def test_non_sticky_drops_previous_promotions():
    router = ToolRouter(tools=CATALOGUE, top_k=2, sticky=False)
    assert "deploy_service" not in names(router.select("resize image", promoted={"deploy_service"}))


def test_schemas_returns_wire_format():
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    schemas = router.schemas("translate this text")
    assert all(s.to_dict()["type"] == "function" for s in schemas)
    assert "translate_text" in [s.name for s in schemas]


def test_embeddings_path_fuses_and_still_ranks():
    """With embeddings supplied, every tool is scored, so top_k is always filled."""
    router = ToolRouter(tools=CATALOGUE, top_k=3, embeddings=FakeEmbeddings())
    selected = router.select("translate this paragraph")
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# Summary pool
# ---------------------------------------------------------------------------


def test_summary_pool_lists_every_tool():
    pool = ToolRouter(tools=CATALOGUE).summary_pool()
    for t in CATALOGUE:
        assert t.name in pool


def test_summary_pool_is_byte_stable_across_calls():
    """Stability is the whole point: a churning prefix defeats provider prompt caching."""
    router = ToolRouter(tools=CATALOGUE)
    router.select("deploy the service")
    assert router.summary_pool() == router.summary_pool()


def test_summary_pool_truncates_long_descriptions():
    long_tool = Tool(name="verbose", description="word " * 200, func=lambda: None)
    pool = ToolRouter(tools=[*CATALOGUE, long_tool], summary_chars=40).summary_pool()
    line = next(line for line in pool.splitlines() if line.startswith("- verbose"))
    assert line.endswith("...")
    assert len(line) < 60


def test_summary_pool_is_far_smaller_than_the_full_schemas():
    import json

    router = ToolRouter(tools=CATALOGUE)
    full = json.dumps([t.to_schema().to_dict() for t in CATALOGUE])
    assert len(router.summary_pool()) < len(full) / 2


# ---------------------------------------------------------------------------
# Agent-loop wiring
# ---------------------------------------------------------------------------


class RecordingModel:
    """Deterministic model that records the tool schemas it was offered on each turn."""

    model_name = "recording"

    def __init__(self, call: tuple[str, dict] | None = None, prose: str = ""):
        self.offered: list[list[str]] = []
        self.call = call
        self.prose = prose

    def _next(self, messages, tools):
        self.offered.append([t.name for t in (tools or [])])
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="done")
        if self.call is not None:
            name, args = self.call
            return AIMessage(
                content=self.prose, tool_calls=[ToolCall(name=name, args=args, id="c0")]
            )
        return AIMessage(content=self.prose)

    def generate(self, messages, *, tools=None, **kwargs):
        return self._next(messages, tools)

    async def agenerate(self, messages, *, tools=None, **kwargs):
        return self._next(messages, tools)


def test_agent_without_router_still_sends_every_schema():
    model = RecordingModel()
    create_agent(model, CATALOGUE)("translate this")
    assert model.offered[0] == names(CATALOGUE)


def test_agent_with_router_sends_only_the_promoted_schemas():
    model = RecordingModel()
    router = ToolRouter(tools=CATALOGUE, top_k=3)
    create_agent(model, CATALOGUE, tool_router=router)("translate this paragraph")
    assert "translate_text" in model.offered[0]
    assert len(model.offered[0]) <= 3


def test_router_injects_the_summary_pool_into_the_prefix():
    model = RecordingModel()
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    result = create_agent(model, CATALOGUE, tool_router=router)("translate this")
    assert any(router.summary_pool() == m.text for m in result.messages)


def test_summary_pool_sits_after_the_system_prompt():
    model = RecordingModel()
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    agent = create_agent(model, CATALOGUE, system_prompt="You are terse.", tool_router=router)
    result = agent("translate this")
    assert result.messages[0].text == "You are terse."
    assert result.messages[1].text == router.summary_pool()


def test_an_unpromoted_tool_still_executes_if_the_model_calls_it():
    """Routing is a context optimisation, not a security boundary — governance is that."""
    model = RecordingModel(call=("deploy_service", {"name": "api"}))
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    result = create_agent(model, CATALOGUE, tool_router=router)("translate this paragraph")
    assert "deploy_service" not in model.offered[0]
    assert result.tool_calls_made == 1
    assert any(m.content == "deployed" for m in result.messages if isinstance(m, ToolMessage))


def test_naming_a_tool_in_prose_promotes_it_on_the_next_iteration():
    """The escape hatch the summary pool advertises has to actually work inside a run."""
    model = RecordingModel(
        call=("read_file", {"path": "a.txt"}), prose="Then I will use deploy_service."
    )
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    create_agent(model, CATALOGUE, tool_router=router, max_iterations=2)("read the file")
    assert "deploy_service" not in model.offered[0]
    assert "deploy_service" in model.offered[1]


def test_naming_a_tool_in_prose_promotes_it_on_the_next_conversation_turn():
    """The same escape hatch across runs, when the caller replays the history."""
    from infy.messages import HumanMessage

    history = [
        HumanMessage(content="do the thing"),
        AIMessage(content="I should use deploy_service for that."),
        HumanMessage(content="go ahead"),
    ]
    model = RecordingModel()
    router = ToolRouter(tools=CATALOGUE, top_k=2)
    create_agent(model, CATALOGUE, tool_router=router)(history)
    assert "deploy_service" in model.offered[0]


def test_sticky_promotions_only_grow_across_turns():
    model = RecordingModel(call=("read_file", {"path": "a.txt"}))
    router = ToolRouter(tools=CATALOGUE, top_k=3, sticky=True)
    create_agent(model, CATALOGUE, tool_router=router)("read the file")
    assert set(model.offered[0]).issubset(set(model.offered[1]))


async def test_async_agent_routes_identically_to_the_sync_agent():
    sync_model, async_model = RecordingModel(), RecordingModel()
    make = lambda: ToolRouter(tools=CATALOGUE, top_k=3)  # noqa: E731
    create_agent(sync_model, CATALOGUE, tool_router=make())("translate this paragraph")
    agent = create_async_agent(async_model, CATALOGUE, tool_router=make())
    await agent("translate this paragraph")
    assert sync_model.offered == async_model.offered
