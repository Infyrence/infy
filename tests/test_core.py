"""Tests for infy core — Runnable, composition, context."""

from infy import Lambda, Parallel, Sequence, coerce
from infy.core import Context


def test_lambda_invoke():
    fn = Lambda(lambda x: x + 1)
    assert fn.invoke(5) == 6


def test_lambda_stream():
    fn = Lambda(lambda x: [x, x * 2, x * 3])
    assert list(fn.stream(5)) == [5, 10, 15]


def test_lambda_generator_stream():
    def gen(x):
        yield from range(x)

    fn = Lambda(gen)
    assert list(fn.stream(3)) == [0, 1, 2]


def test_pipe_operator():
    chain = Lambda(lambda x: x + 1) | Lambda(lambda x: x * 2)
    assert chain.invoke(5) == 12  # (5+1)*2


def test_pipe_operator_three_steps():
    chain = (
        Lambda(lambda x: x.upper()) | Lambda(lambda x: f"Hello, {x}!") | Lambda(lambda x: x + " :)")
    )
    assert chain.invoke("world") == "Hello, WORLD! :)"


def test_pipe_flattening():
    s1 = Sequence([Lambda(lambda x: x + 1)])
    s2 = Sequence([Lambda(lambda x: x * 2)])
    combined = s1 | s2
    assert len(combined.steps) == 2  # Flattened, not nested


def test_ror_operator():
    fn = Lambda(lambda x: x + 1)
    chain = Lambda(lambda x: x * 2) | fn
    assert chain.invoke(5) == 11  # (5*2)+1


def test_parallel_invoke():
    p = Parallel(
        {
            "upper": Lambda(lambda x: x.upper()),
            "length": Lambda(lambda x: len(x)),
        }
    )
    result = p.invoke("hello")
    assert result == {"upper": "HELLO", "length": 5}


def test_parallel_stream():
    p = Parallel(
        {
            "double": Lambda(lambda x: x * 2),
            "triple": Lambda(lambda x: x * 3),
        }
    )
    results = list(p.stream(5))
    assert len(results) == 2


def test_coerce_callable():
    r = coerce(lambda x: x + 1)
    assert isinstance(r, Lambda)
    assert r.invoke(5) == 6


def test_coerce_dict():
    r = coerce({"a": lambda x: x, "b": lambda x: x * 2})
    assert isinstance(r, Parallel)
    result = r.invoke(5)
    assert result == {"a": 5, "b": 10}


def test_coerce_runnable():
    fn = Lambda(lambda x: x)
    assert coerce(fn) is fn


def test_context_merge():
    c1 = Context(tags=["a", "b"])
    c2 = Context(tags=["b", "c"], metadata={"key": "val"})
    merged = c1.merge(c2)
    assert "a" in merged.tags
    assert "b" in merged.tags
    assert "c" in merged.tags
    assert merged.metadata == {"key": "val"}


def test_context_merge_none():
    c = Context(tags=["a"])
    assert c.merge(None).tags == ["a"]


def test_sequence_invoke():
    s = Sequence([Lambda(lambda x: x + 1), Lambda(lambda x: x * 2)])
    assert s.invoke(5) == 12


def test_sequence_stream():
    s = Sequence([Lambda(lambda x: [x, x + 1]), Lambda(lambda x: x * 10)])
    result = list(s.stream(5))
    assert result == [50, 60]


def test_complex_chain():
    chain = Parallel(
        {
            "name": Lambda(lambda x: x.upper()),
            "length": Lambda(lambda x: len(x)),
        }
    ) | Lambda(lambda d: f"{d['name']} is {d['length']}")
    assert chain.invoke("alice") == "ALICE is 5"
