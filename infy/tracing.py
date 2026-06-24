"""Pluggable tracing — no LangSmith lock-in.

LangChain equivalent: 5,209 lines (tracers/) + 4,850 lines (callbacks/) +
    deeply coupled langsmith dependency.
infy: ~70 lines, zero dependencies.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Tracer protocol — implement this for any observability backend
# ---------------------------------------------------------------------------


@runtime_checkable
class Tracer(Protocol):
    """Implement this to plug in LangSmith, OpenTelemetry, or anything."""

    def on_start(self, name: str, input: Any, metadata: dict[str, Any] | None = None) -> str:
        """Called when a traced function starts. Returns a run_id."""
        ...

    def on_end(self, run_id: str, output: Any, duration_ms: float) -> None:
        """Called when a traced function completes successfully."""
        ...

    def on_error(self, run_id: str, error: Exception, duration_ms: float) -> None:
        """Called when a traced function raises an exception."""
        ...


# ---------------------------------------------------------------------------
# NoOpTracer — default, does nothing
# ---------------------------------------------------------------------------


class NoOpTracer:
    """Default tracer that does nothing. Zero overhead."""

    def on_start(self, name: str, input: Any, metadata: dict[str, Any] | None = None) -> str:
        return ""

    def on_end(self, run_id: str, output: Any, duration_ms: float) -> None:
        pass

    def on_error(self, run_id: str, error: Exception, duration_ms: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Global tracer — set once, used everywhere
# ---------------------------------------------------------------------------

_tracer: Tracer = NoOpTracer()


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer for all infy operations."""
    global _tracer
    _tracer = tracer


def get_tracer() -> Tracer:
    """Get the current global tracer."""
    return _tracer


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def traced(
    func: Any = None,
    *,
    name: str | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Decorator to trace function execution.

    Usage:
        @traced
        def my_function(x):
            return x * 2

        @traced(name="custom_name")
        def another(x):
            return x
    """

    def decorator(f: Any) -> Any:
        run_name = name or f.__name__

        # Resolve the tracer per call, not at decoration time, so a later
        # set_tracer() applies to functions that were decorated at import time.
        def _resolve() -> Tracer:
            return tracer or get_tracer()

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t = _resolve()
            run_id = t.on_start(run_name, {"args": args, "kwargs": kwargs})
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                t.on_end(run_id, result, duration)
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                t.on_error(run_id, e, duration)
                raise

        @functools.wraps(f)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            t = _resolve()
            run_id = t.on_start(run_name, {"args": args, "kwargs": kwargs})
            start = time.perf_counter()
            try:
                result = await f(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                t.on_end(run_id, result, duration)
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                t.on_error(run_id, e, duration)
                raise

        import asyncio

        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
