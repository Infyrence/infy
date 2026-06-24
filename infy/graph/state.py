"""StateGraph builder and CompiledGraph — the user-facing API.

Build a graph: StateGraph(State) -> add_node -> add_edge -> compile -> invoke.

The executor is a Bulk-Synchronous-Parallel (Pregel-style) loop:
every superstep runs the whole active frontier against one frozen snapshot of
the channels, batches their writes, then commits them through the channels
(so reducers actually reduce). The next frontier is the union of every active
node's successors, which is what makes real fan-out and diamond joins work.

LangGraph: 1,964 lines. infy: ~360 lines.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from infy.graph.channels import BaseChannel, BinOp, LastValue
from infy.graph.checkpoint import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
)
from infy.graph.errors import (
    GraphInterrupt,
    GraphRecursionError,
    NodeError,
    NodeInterrupt,
)
from infy.graph.types import (
    END,
    START,
    Command,
    Interrupt,
    Send,
    StateSnapshot,
)

# A unit of work: the node to run and an optional per-task input override (set
# by Send for dynamic fan-out; None for ordinary edge traversal).
Task = tuple[str, Any]


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass
class NodeSpec:
    """Specification for a graph node."""

    name: str
    func: Callable[..., Any]
    retry: int = 0
    timeout: float | None = None


@dataclass
class EdgeSpec:
    """Specification for a graph edge."""

    source: str
    target: str


@dataclass
class ConditionalEdgeSpec:
    """Specification for a conditional edge."""

    source: str
    condition: Callable[..., Any]
    path_map: dict[str, str] | None = None
    default: str | None = None


# ---------------------------------------------------------------------------
# StateGraph builder
# ---------------------------------------------------------------------------


class StateGraph:
    """Build a stateful graph by adding nodes and edges.

    Usage:
        from infy.graph import StateGraph, START, END

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            next_node: str

        graph = StateGraph(State)
        graph.add_node("agent", call_model)
        graph.add_node("tools", execute_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        app = graph.compile()
        result = app.invoke({"messages": [HumanMessage(content="Hi")]})
    """

    def __init__(self, state_schema: type | dict[str, Any] | None = None) -> None:
        self.state_schema = state_schema
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: list[EdgeSpec] = []
        self.conditional_edges: list[ConditionalEdgeSpec] = []

    def add_node(self, name: str, func: Callable[..., Any], **kwargs: Any) -> StateGraph:
        """Add a processing node to the graph.

        The function receives the current state dict and returns a partial state
        dict to merge (applied through the channels), a Command, or None.
        """
        if name in (START, END):
            raise ValueError(f"Cannot add node with reserved name '{name}'")
        self.nodes[name] = NodeSpec(name=name, func=func, **kwargs)
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        """Add an unconditional edge from source to target.

        A node may have several outgoing edges; all of their targets become part
        of the next frontier (fan-out).
        """
        self.edges.append(EdgeSpec(source=source, target=target))
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: Callable[..., Any],
        path_map: dict[str, str] | None = None,
        default: str | None = None,
    ) -> StateGraph:
        """Add conditional routing from a source node.

        The condition receives the current state and returns a node name, a list
        of node names, or a (list of) Send object(s) for dynamic fan-out.
        """
        self.conditional_edges.append(
            ConditionalEdgeSpec(
                source=source,
                condition=condition,
                path_map=path_map,
                default=default,
            )
        )
        return self

    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        recursion_limit: int = 25,
    ) -> CompiledGraph:
        """Compile the graph into an executable CompiledGraph."""
        if not self.nodes:
            raise ValueError("Graph has no nodes")

        channels = self._build_channels()
        self._validate()

        return CompiledGraph(
            nodes=dict(self.nodes),
            edges=list(self.edges),
            conditional_edges=list(self.conditional_edges),
            channels=channels,
            state_schema=self.state_schema,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or [],
            interrupt_after=interrupt_after or [],
            recursion_limit=recursion_limit,
        )

    def _build_channels(self) -> dict[str, BaseChannel[Any]]:
        """Infer one channel per state field.

        ``Annotated[type, reducer]`` -> BinOp(reducer) so the field accumulates;
        every other field -> LastValue (single write per superstep).
        """
        channels: dict[str, BaseChannel[Any]] = {}
        schema = self.state_schema
        if schema is None:
            return channels

        try:
            hints = typing.get_type_hints(schema, include_extras=True)
        except Exception:
            hints = dict(getattr(schema, "__annotations__", {}) or {})

        for name, field_type in hints.items():
            metadata = getattr(field_type, "__metadata__", None)
            if metadata and callable(metadata[0]):
                channels[name] = BinOp(metadata[0], key=name)
            else:
                channels[name] = LastValue(key=name)

        return channels

    def _validate(self) -> None:
        """Validate the static graph structure."""
        node_names = set(self.nodes.keys())

        for edge in self.edges:
            if edge.source != START and edge.source not in node_names:
                raise ValueError(f"Edge source '{edge.source}' is not a node")
            if edge.target != END and edge.target not in node_names:
                raise ValueError(f"Edge target '{edge.target}' is not a node")

        for cedge in self.conditional_edges:
            if cedge.source != START and cedge.source not in node_names:
                raise ValueError(f"Conditional edge source '{cedge.source}' is not a node")
            if cedge.path_map:
                for target in cedge.path_map.values():
                    if target != END and target not in node_names:
                        raise ValueError(f"Conditional edge target '{target}' is not a node")


# ---------------------------------------------------------------------------
# CompiledGraph — the runtime executor
# ---------------------------------------------------------------------------


class CompiledGraph:
    """A compiled, executable graph with checkpoint and interrupt support.

    Supports:
    - invoke — run a frontier of nodes per superstep until it drains
    - get_state / update_state — inspect and modify persisted state
    - interrupt_before / interrupt_after + resume for human-in-the-loop
    """

    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        edges: list[EdgeSpec],
        conditional_edges: list[ConditionalEdgeSpec],
        channels: dict[str, BaseChannel[Any]],
        state_schema: type | dict[str, Any] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        recursion_limit: int = 25,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.channels = channels
        self.state_schema = state_schema
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self.recursion_limit = recursion_limit

        # Adjacency: a source may map to many targets / many conditional edges.
        self._edges: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            self._edges[edge.source].append(edge.target)
        self._conditional: dict[str, list[ConditionalEdgeSpec]] = defaultdict(list)
        for cedge in conditional_edges:
            self._conditional[cedge.source].append(cedge)

    # -- Public API --------------------------------------------------------

    def _prepare(
        self, input: Any, config: dict[str, Any] | None
    ) -> tuple[dict[str, BaseChannel[Any]], list[Task], bool, dict[str, Any], int]:
        """Shared setup for invoke/ainvoke/astream: fresh channels, restored state,
        the initial (or resumed) frontier, and whether to skip the resume interrupt."""
        config = config or {}
        recursion_limit = config.get("recursion_limit", self.recursion_limit)
        thread_config = self._thread_config(config)

        channels = {key: ch.copy() for key, ch in self.channels.items()}

        # Restore persisted state (and a pending frontier, if we were interrupted).
        checkpoint = self._load_checkpoint(config)
        resume_frontier: list[Task] | None = None
        if checkpoint is not None:
            self._restore_channels(channels, checkpoint.channel_values)
            if checkpoint.next_tasks:
                resume_frontier = [(t[0], t[1]) for t in checkpoint.next_tasks]

        self._apply_input(channels, input)

        if resume_frontier is not None:
            frontier = resume_frontier
        else:
            frontier = self._normalize(self._successors(START, self._state_dict(channels)))

        # On resume the first superstep must NOT re-trigger the interrupt that
        # paused us — the caller already chose to continue.
        skip_interrupt_before = resume_frontier is not None
        return channels, frontier, skip_interrupt_before, thread_config, recursion_limit

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the graph and return the final materialized state."""
        channels, frontier, skip_interrupt_before, thread_config, recursion_limit = self._prepare(
            input, config
        )

        step = 0
        while frontier:
            if step >= recursion_limit:
                raise GraphRecursionError(recursion_limit, step)

            if not skip_interrupt_before:
                paused = [t for t in frontier if t[0] in self.interrupt_before]
                if paused:
                    self._save_checkpoint(thread_config, channels, step, "loop", frontier)
                    raise GraphInterrupt(
                        interrupts=[
                            Interrupt(value={"node": t[0], "state": self._state_dict(channels)})
                            for t in paused
                        ]
                    )
            skip_interrupt_before = False

            snapshot = self._state_dict(channels)
            writes: dict[str, list[Any]] = defaultdict(list)
            gotos: dict[str, str] = {}
            ran: list[str] = []

            for node_name, arg in frontier:
                if node_name == END:
                    continue
                node_spec = self.nodes.get(node_name)
                if node_spec is None:
                    raise ValueError(f"Node '{node_name}' not found")

                try:
                    result = node_spec.func(self._node_input(snapshot, arg))
                except NodeInterrupt as e:
                    # More specific than GraphInterrupt (its base) — must be caught first,
                    # otherwise the in-node interrupt() never checkpoints and resume breaks.
                    self._save_checkpoint(thread_config, channels, step, "loop", frontier)
                    raise GraphInterrupt(interrupts=e.interrupts) from e
                except GraphInterrupt:
                    raise
                except Exception as e:
                    raise NodeError(node_name, e) from e

                ran.append(node_name)
                self._collect(node_name, result, writes, gotos)

            # Commit every node's writes at once — reducers fire here.
            self._apply_writes(channels, writes)
            updated = self._state_dict(channels)

            after = [n for n in ran if n in self.interrupt_after]
            next_frontier = self._next_frontier(ran, gotos, updated)

            if after:
                self._save_checkpoint(thread_config, channels, step, "loop", next_frontier)
                raise GraphInterrupt(
                    interrupts=[Interrupt(value={"node": n, "state": updated}) for n in after]
                )

            frontier = next_frontier
            step += 1

        self._save_checkpoint(thread_config, channels, step, "loop", [])
        return self._state_dict(channels)

    async def ainvoke(self, input: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Async counterpart to ``invoke``: awaits async node callables (and calls sync
        nodes directly), reusing the same channel/reducer/checkpoint/interrupt machinery."""
        final: dict[str, Any] = {}
        async for state in self.astream(input, config):
            final = state
        return final

    async def astream(
        self, input: Any, config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Async streaming execution — yields the materialized state after each superstep.

        Within a superstep the active frontier runs CONCURRENTLY (real parallel fan-out
        for async nodes), but writes are folded in frontier order so order-sensitive
        reducers behave identically to the sync ``invoke`` path. Because the frontier runs
        concurrently, a node raising does NOT cancel its already-running siblings (unlike
        the sequential sync path); the first error in frontier order is surfaced, so nodes
        should be side-effect-free or idempotent.
        """
        channels, frontier, skip_interrupt_before, thread_config, recursion_limit = self._prepare(
            input, config
        )

        step = 0
        ran_any = False
        while frontier:
            if step >= recursion_limit:
                raise GraphRecursionError(recursion_limit, step)

            if not skip_interrupt_before:
                paused = [t for t in frontier if t[0] in self.interrupt_before]
                if paused:
                    self._save_checkpoint(thread_config, channels, step, "loop", frontier)
                    raise GraphInterrupt(
                        interrupts=[
                            Interrupt(value={"node": t[0], "state": self._state_dict(channels)})
                            for t in paused
                        ]
                    )
            skip_interrupt_before = False

            snapshot = self._state_dict(channels)
            writes: dict[str, list[Any]] = defaultdict(list)
            gotos: dict[str, str] = {}
            ran: list[str] = []

            active: list[Task] = []
            coros: list[Any] = []
            for node_name, arg in frontier:
                if node_name == END:
                    continue
                node_spec = self.nodes.get(node_name)
                if node_spec is None:
                    raise ValueError(f"Node '{node_name}' not found")
                active.append((node_name, arg))
                coros.append(
                    self._arun_node(node_spec.func, self._node_input(snapshot, arg), node_name)
                )

            # Run the frontier concurrently; gather preserves frontier order in results.
            results = await asyncio.gather(*coros, return_exceptions=True)
            for (node_name, _arg), result in zip(active, results, strict=True):
                if isinstance(result, NodeInterrupt):
                    # Mirror the sync path: persist exactly once, for the first
                    # interrupting node in frontier order, then pause.
                    self._save_checkpoint(thread_config, channels, step, "loop", frontier)
                    raise GraphInterrupt(interrupts=result.interrupts) from result
                if isinstance(result, BaseException):
                    raise result
                ran.append(node_name)
                self._collect(node_name, result, writes, gotos)

            # Commit every node's writes at once — reducers fire here.
            self._apply_writes(channels, writes)
            updated = self._state_dict(channels)

            after = [n for n in ran if n in self.interrupt_after]
            next_frontier = self._next_frontier(ran, gotos, updated)

            if after:
                self._save_checkpoint(thread_config, channels, step, "loop", next_frontier)
                raise GraphInterrupt(
                    interrupts=[Interrupt(value={"node": n, "state": updated}) for n in after]
                )

            ran_any = True
            yield updated
            frontier = next_frontier
            step += 1

        self._save_checkpoint(thread_config, channels, step, "loop", [])
        if not ran_any:
            yield self._state_dict(channels)

    async def _arun_node(
        self, func: Callable[..., Any], node_input: dict[str, Any], node_name: str
    ) -> Any:
        """Run one node, awaiting it if it is async. GraphInterrupt and NodeInterrupt
        propagate unchanged (the caller checkpoints exactly once, in frontier order); any
        other exception is wrapped as NodeError, exactly as the sync path does."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(node_input)
            result = func(node_input)
            if inspect.isawaitable(result):
                return await result
            return result
        except (GraphInterrupt, NodeInterrupt):
            raise
        except Exception as e:
            raise NodeError(node_name, e) from e

    def get_state(self, config: dict[str, Any] | None = None) -> StateSnapshot:
        """Return the persisted state and the nodes that would run next."""
        config = config or {}
        checkpoint = self._load_checkpoint(config)
        if checkpoint is None:
            return StateSnapshot(values={}, next=[], config=config)

        values = {k: v for k, v in checkpoint.channel_values.items() if not k.startswith("__")}
        next_nodes = [t[0] for t in checkpoint.next_tasks]
        return StateSnapshot(
            values=values,
            next=next_nodes,
            config=config,
            metadata={"step": checkpoint.channel_versions.get("__step__", 0)},
        )

    def update_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        as_node: str | None = None,
    ) -> dict[str, Any]:
        """Manually merge values into the persisted state as a new checkpoint."""
        checkpoint = self._load_checkpoint(config)
        checkpoint = checkpoint.copy() if checkpoint else Checkpoint()
        checkpoint.channel_values.update(values)

        thread_config = self._thread_config(config)
        if self.checkpointer:
            step = int(checkpoint.channel_versions.get("__step__", 0))
            metadata = CheckpointMetadata(source="update", step=step)
            thread_config = self.checkpointer.put(
                thread_config, checkpoint, metadata, {k: 1 for k in values}
            )
        return thread_config

    # -- Routing -----------------------------------------------------------

    def _successors(self, node: str, state: dict[str, Any]) -> list[Task]:
        """All tasks reachable from a node via its edges and conditional edges."""
        tasks: list[Task] = [(target, None) for target in self._edges.get(node, [])]
        for cedge in self._conditional.get(node, []):
            tasks.extend(self._eval_conditional(cedge, state))
        return tasks

    def _eval_conditional(self, cedge: ConditionalEdgeSpec, state: dict[str, Any]) -> list[Task]:
        try:
            result = cedge.condition(state)
        except Exception:
            if cedge.default is not None:
                return [(cedge.default, None)]
            raise

        if isinstance(result, Send):
            result = [result]
        if isinstance(result, list) and result and all(isinstance(s, Send) for s in result):
            return [(s.node, s.arg) for s in result]

        names = result if isinstance(result, list) else [result]
        tasks: list[Task] = []
        for name in names:
            if cedge.path_map and name in cedge.path_map:
                tasks.append((cedge.path_map[name], None))
            else:
                tasks.append((name, None))
        return tasks

    def _next_frontier(
        self, ran: list[str], gotos: dict[str, str], state: dict[str, Any]
    ) -> list[Task]:
        """Union the successors of every node that ran this superstep."""
        tasks: list[Task] = []
        for node_name in ran:
            if node_name in gotos:
                target = gotos[node_name]
                if target != END:
                    tasks.append((target, None))
            else:
                tasks.extend(self._successors(node_name, state))
        return self._normalize(tasks)

    def _normalize(self, tasks: list[Task]) -> list[Task]:
        """Drop END targets and de-duplicate plain edge traversals.

        De-duping joins diamonds (B->D and C->D run D once) while leaving each
        Send task intact, since fan-out tasks are meant to be distinct.
        """
        seen: set[str] = set()
        out: list[Task] = []
        for node, arg in tasks:
            if node == END:
                continue
            if arg is None:
                if node in seen:
                    continue
                seen.add(node)
            out.append((node, arg))
        return out

    # -- State / channels --------------------------------------------------

    def _collect(
        self,
        node_name: str,
        result: Any,
        writes: dict[str, list[Any]],
        gotos: dict[str, str],
    ) -> None:
        """Fold a node's return value into the superstep's pending writes."""
        if result is None:
            return
        if isinstance(result, Command):
            if result.update:
                for key, value in result.update.items():
                    writes[key].append(value)
            if result.goto is not None:
                gotos[node_name] = result.goto
        elif isinstance(result, dict):
            for key, value in result.items():
                writes[key].append(value)
        # Bare strings / Send lists are routing signals consumed by conditional
        # edges, not state writes — ignore them here.

    def _apply_writes(
        self, channels: dict[str, BaseChannel[Any]], writes: dict[str, list[Any]]
    ) -> None:
        for key, values in writes.items():
            if key not in channels:
                channels[key] = LastValue(key=key)
            channels[key].update(values)

    def _apply_input(self, channels: dict[str, BaseChannel[Any]], input: Any) -> None:
        if input is None:
            return
        if isinstance(input, dict):
            updates = input
        elif isinstance(input, list):
            updates = {"messages": input}
        else:
            updates = {"input": input}
        for key, value in updates.items():
            if key not in channels:
                channels[key] = LastValue(key=key)
            channels[key].update([value])

    def _restore_channels(
        self, channels: dict[str, BaseChannel[Any]], values: dict[str, Any]
    ) -> None:
        for key, value in values.items():
            if key.startswith("__"):
                continue
            if key not in channels:
                channels[key] = LastValue(key=key)
            channels[key].from_checkpoint(value)

    def _state_dict(self, channels: dict[str, BaseChannel[Any]]) -> dict[str, Any]:
        return {key: ch.get() for key, ch in channels.items() if ch.is_available()}

    def _node_input(self, snapshot: dict[str, Any], arg: Any) -> dict[str, Any]:
        node_input = dict(snapshot)
        if arg is not None:
            if isinstance(arg, dict):
                node_input.update(arg)
            node_input["input"] = arg
        return node_input

    # -- Checkpointing -----------------------------------------------------

    def _thread_config(self, config: dict[str, Any]) -> dict[str, Any]:
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id") or str(uuid.uuid4())
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            }
        }

    def _load_checkpoint(self, config: dict[str, Any]) -> Checkpoint | None:
        if not self.checkpointer:
            return None
        try:
            tup = self.checkpointer.get_tuple(config)
        except Exception:
            return None
        return tup.checkpoint if tup else None

    def _save_checkpoint(
        self,
        config: dict[str, Any],
        channels: dict[str, BaseChannel[Any]],
        step: int,
        source: str,
        frontier: list[Task],
    ) -> None:
        if not self.checkpointer:
            return
        values = self._state_dict(channels)
        checkpoint = Checkpoint(
            channel_values=values,
            channel_versions={"__step__": step},
            next_tasks=[[node, arg] for node, arg in frontier],
        )
        metadata = CheckpointMetadata(source=source, step=step)
        self.checkpointer.put(config, checkpoint, metadata, {k: step for k in values})
