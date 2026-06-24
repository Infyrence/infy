"""Core primitives: Runnable protocol, composition, context.

This is the heart of infy. Async-first design.
LangChain equivalent: 6,715 lines in runnables/base.py + 713 in config.py.
infy: ~300 lines with full async support.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

# Contravariant: T only appears in input position across the Runnable protocol.
T = TypeVar("T", contravariant=True)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=False)
class Context:
    """Minimal execution context. No Pydantic. No ContextVar magic."""

    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_concurrency: int | None = None

    def merge(self, other: Context | None) -> Context:
        if other is None:
            return self
        return Context(
            tags=sorted(set(self.tags + other.tags)),
            metadata={**self.metadata, **other.metadata},
            max_concurrency=other.max_concurrency or self.max_concurrency,
        )


# ---------------------------------------------------------------------------
# Runnable Protocol — the ENTIRE contract
# ---------------------------------------------------------------------------


@runtime_checkable
class Runnable(Protocol[T]):
    """A runnable unit of computation. Sync + async."""

    def invoke(self, input: T, ctx: Context | None = None) -> Any: ...

    async def ainvoke(self, input: T, ctx: Context | None = None) -> Any: ...

    def stream(self, input: T, ctx: Context | None = None) -> Iterator[Any]: ...

    def astream(self, input: T, ctx: Context | None = None) -> AsyncIterator[Any]: ...

    def __or__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence: ...

    def __ror__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence: ...


# ---------------------------------------------------------------------------
# coerce
# ---------------------------------------------------------------------------


def coerce(obj: Runnable[Any] | Callable[..., Any] | dict[Any, Any] | Any) -> Runnable[Any]:
    """Convert a callable, dict, or Runnable to a Runnable."""
    from infy.tools import Tool

    if isinstance(obj, Runnable):
        return obj
    if hasattr(obj, "invoke") and hasattr(obj, "__or__"):
        # StructuredOutputModel or similar duck-typed runnables.
        return cast("Runnable[Any]", obj)
    if isinstance(obj, Tool):
        return _ToolAsRunnable(obj)
    if isinstance(obj, dict):
        return Parallel({k: coerce(v) for k, v in obj.items()})
    if callable(obj):
        return Lambda(obj)
    raise TypeError(f"Cannot coerce {type(obj).__name__} to Runnable")


def acoerce(obj: Runnable[Any] | Callable[..., Any] | dict[Any, Any] | Any) -> Runnable[Any]:
    """Same as coerce, but wraps sync callables to run in executor."""
    from infy.tools import Tool

    if isinstance(obj, Runnable):
        return obj
    if isinstance(obj, Tool):
        return _ToolAsRunnable(obj)
    if isinstance(obj, dict):
        return Parallel({k: acoerce(v) for k, v in obj.items()})
    if callable(obj):
        if asyncio.iscoroutinefunction(obj):
            return AsyncLambda(obj)
        return Lambda(obj)
    raise TypeError(f"Cannot coerce {type(obj).__name__} to Runnable")


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------


@dataclass
class Sequence:
    """A chain of runnables executed in order: step1 | step2 | step3."""

    steps: list[Runnable[Any]]

    def invoke(self, input: Any, ctx: Context | None = None) -> Any:
        result = input
        for step in self.steps:
            result = step.invoke(result, ctx)
        return result

    async def ainvoke(self, input: Any, ctx: Context | None = None) -> Any:
        result = input
        for step in self.steps:
            result = await step.ainvoke(result, ctx)
        return result

    def stream(self, input: Any, ctx: Context | None = None) -> Iterator[Any]:
        it: Iterator[Any] = iter([input])
        for step in self.steps:
            it = _chain_stream(step, it, ctx)
        yield from it

    async def astream(self, input: Any, ctx: Context | None = None) -> AsyncIterator[Any]:
        current = input
        for step in self.steps:
            chunks: list[Any] = []
            async for chunk in step.astream(current, ctx):
                chunks.append(chunk)
                yield chunk
            if chunks:
                current = chunks[-1]

    def __or__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        r = coerce(other)
        if isinstance(r, Sequence):
            return Sequence(self.steps + r.steps)
        return Sequence(self.steps + [r])

    def __ror__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        left = coerce(other)
        if isinstance(left, Sequence):
            return Sequence(left.steps + self.steps)
        return Sequence([left] + self.steps)

    def __repr__(self) -> str:
        return " | ".join(type(s).__name__ for s in self.steps)


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


@dataclass
class Parallel:
    """Execute multiple runnables concurrently on the same input."""

    steps: dict[str, Runnable[Any]]

    def invoke(self, input: Any, ctx: Context | None = None) -> dict[str, Any]:
        max_workers = (ctx.max_concurrency if ctx else None) or len(self.steps)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                name: ex.submit(step.invoke, input, ctx) for name, step in self.steps.items()
            }
            return {name: f.result() for name, f in futures.items()}

    async def ainvoke(self, input: Any, ctx: Context | None = None) -> dict[str, Any]:
        sem = asyncio.Semaphore(
            ctx.max_concurrency if ctx and ctx.max_concurrency else len(self.steps)
        )

        async def _run(name: str, step: Runnable[Any]) -> tuple[str, Any]:
            async with sem:
                return name, await step.ainvoke(input, ctx)

        results = await asyncio.gather(*[_run(n, s) for n, s in self.steps.items()])
        return dict(results)

    def stream(self, input: Any, ctx: Context | None = None) -> Iterator[dict[str, Any]]:
        max_workers = (ctx.max_concurrency if ctx else None) or len(self.steps)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                name: ex.submit(step.invoke, input, ctx) for name, step in self.steps.items()
            }
            for name, future in futures.items():
                yield {name: future.result()}

    async def astream(
        self, input: Any, ctx: Context | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        sem = asyncio.Semaphore(
            ctx.max_concurrency if ctx and ctx.max_concurrency else len(self.steps)
        )

        async def _run(name: str, step: Runnable[Any]) -> dict[str, Any]:
            async with sem:
                result = await step.ainvoke(input, ctx)
                return {name: result}

        tasks = [asyncio.create_task(_run(n, s)) for n, s in self.steps.items()]
        for task in asyncio.as_completed(tasks):
            yield await task

    def __or__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([self, coerce(other)])

    def __ror__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([coerce(other), self])

    def __repr__(self) -> str:
        return "{" + ", ".join(self.steps.keys()) + "}"


# ---------------------------------------------------------------------------
# Lambda — sync callable
# ---------------------------------------------------------------------------


@dataclass
class Lambda:
    """Wrap a plain function as a Runnable."""

    func: Callable[..., Any]

    def invoke(self, input: Any, ctx: Context | None = None) -> Any:
        return self.func(input)

    async def ainvoke(self, input: Any, ctx: Context | None = None) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.func, input)

    def stream(self, input: Any, ctx: Context | None = None) -> Iterator[Any]:
        result = self.func(input)
        if (
            isinstance(result, Iterator)
            or hasattr(result, "__iter__")
            and not isinstance(result, (str, bytes))
        ):
            yield from result
        else:
            yield result

    async def astream(self, input: Any, ctx: Context | None = None) -> AsyncIterator[Any]:
        result = self.func(input)
        if isinstance(result, AsyncIterator):
            async for item in result:
                yield item
        elif isinstance(result, Iterator):
            for item in result:
                yield item
        else:
            yield result

    def __or__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([self, coerce(other)])

    def __ror__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([coerce(other), self])

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", None) or type(self.func).__name__
        return f"Lambda({name})"


# ---------------------------------------------------------------------------
# AsyncLambda — async callable
# ---------------------------------------------------------------------------


@dataclass
class AsyncLambda:
    """Wrap an async function as a Runnable."""

    func: Callable[..., Any]

    def invoke(self, input: Any, ctx: Context | None = None) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.func(input)).result()
        return asyncio.run(self.func(input))

    async def ainvoke(self, input: Any, ctx: Context | None = None) -> Any:
        return await self.func(input)

    def stream(self, input: Any, ctx: Context | None = None) -> Iterator[Any]:
        result = self.invoke(input)
        if isinstance(result, AsyncIterator):
            import warnings

            warnings.warn("Use astream() for async iterables", stacklevel=2)
        elif isinstance(result, Iterator):
            yield from result
        else:
            yield result

    async def astream(self, input: Any, ctx: Context | None = None) -> AsyncIterator[Any]:
        result = await self.func(input)
        if isinstance(result, AsyncIterator) or hasattr(result, "__aiter__"):
            async for item in result:
                yield item
        else:
            yield result

    def __or__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([self, coerce(other)])

    def __ror__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([coerce(other), self])

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", None) or type(self.func).__name__
        return f"AsyncLambda({name})"


# ---------------------------------------------------------------------------
# _ToolAsRunnable
# ---------------------------------------------------------------------------


@dataclass
class _ToolAsRunnable:
    """Internal adapter: Tool -> Runnable."""

    tool: Any

    def invoke(self, input: str | dict[str, Any], ctx: Context | None = None) -> Any:
        return self.tool.invoke(input)

    async def ainvoke(self, input: str | dict[str, Any], ctx: Context | None = None) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.tool.invoke, input)

    def stream(self, input: str | dict[str, Any], ctx: Context | None = None) -> Iterator[Any]:
        yield self.tool.invoke(input)

    async def astream(
        self, input: str | dict[str, Any], ctx: Context | None = None
    ) -> AsyncIterator[Any]:
        yield self.tool.invoke(input)

    def __or__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([self, coerce(other)])

    def __ror__(self, other: Runnable[Any] | Callable[..., Any]) -> Sequence:
        return Sequence([coerce(other), self])


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _chain_stream(
    step: Runnable[Any], upstream: Iterator[Any], ctx: Context | None
) -> Iterator[Any]:
    if hasattr(step, "transform"):
        return cast("Iterator[Any]", step.transform(upstream, ctx))
    items = list(upstream)
    if len(items) == 1:
        return step.stream(items[0], ctx)
    return _flatten(step.stream(item, ctx) for item in items)


def _flatten(iterables: Iterator[Iterator[Any]]) -> Iterator[Any]:
    for it in iterables:
        yield from it
