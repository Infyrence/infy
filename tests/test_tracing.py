"""Tests for infy tracing."""

import contextlib

from infy.tracing import NoOpTracer, get_tracer, set_tracer, traced


class RecordTracer:
    """A test tracer that records all events."""

    def __init__(self):
        self.events = []

    def on_start(self, name, input, metadata=None):
        self.events.append(("start", name, input))
        return "run_123"

    def on_end(self, run_id, output, duration_ms):
        self.events.append(("end", run_id, output, duration_ms))

    def on_error(self, run_id, error, duration_ms):
        self.events.append(("error", run_id, error, duration_ms))


def test_noop_tracer():
    tracer = NoOpTracer()
    run_id = tracer.on_start("test", {})
    assert run_id == ""
    tracer.on_end("", None, 0)
    tracer.on_error("", Exception("test"), 0)


def test_traced_decorator():
    tracer = RecordTracer()
    set_tracer(tracer)

    @traced
    def my_func(x):
        return x * 2

    result = my_func(5)
    assert result == 10
    assert len(tracer.events) == 2
    assert tracer.events[0][0] == "start"
    assert tracer.events[1][0] == "end"

    # Reset
    set_tracer(NoOpTracer())


def test_traced_decorator_with_error():
    tracer = RecordTracer()
    set_tracer(tracer)

    @traced
    def failing():
        raise ValueError("oops")

    with contextlib.suppress(ValueError):
        failing()

    assert len(tracer.events) == 2
    assert tracer.events[0][0] == "start"
    assert tracer.events[1][0] == "error"
    assert isinstance(tracer.events[1][2], ValueError)

    set_tracer(NoOpTracer())


def test_traced_custom_name():
    tracer = RecordTracer()
    set_tracer(tracer)

    @traced(name="custom_name")
    def my_func():
        return 42

    my_func()
    assert tracer.events[0][1] == "custom_name"

    set_tracer(NoOpTracer())


def test_set_get_tracer():
    original = get_tracer()
    tracer = RecordTracer()
    set_tracer(tracer)
    assert get_tracer() is tracer
    set_tracer(original)
