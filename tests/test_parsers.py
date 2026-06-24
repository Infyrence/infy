"""Tests for infy parsers."""

import pytest

from infy.parsers import JsonParser, StrParser


def test_str_parser_string():
    parser = StrParser()
    assert parser.parse("hello") == "hello"


def test_str_parser_message():
    from infy.messages import AIMessage

    parser = StrParser()
    msg = AIMessage(content="hello world")
    assert parser.parse(msg) == "hello world"


def test_json_parser_simple():
    parser = JsonParser()
    result = parser.parse('{"key": "value"}')
    assert result == {"key": "value"}


def test_json_parser_from_markdown():
    parser = JsonParser()
    result = parser.parse('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_json_parser_from_markdown_no_lang():
    parser = JsonParser()
    result = parser.parse('```\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_json_parser_invalid():
    parser = JsonParser()
    with pytest.raises(ValueError):
        parser.parse("not json at all")


def test_json_parser_partial():
    parser = JsonParser()
    # Partial parsing might return None or a partial result; the key thing is
    # that it does not raise.
    parser.parse('{"key":', partial=True)


def test_json_parser_stream():
    parser = JsonParser()
    chunks = ['{"name":', ' "test",', ' "value": 42}']
    results = list(parser.stream(iter(chunks)))
    # Should yield at least one parsed result
    assert len(results) >= 1
    assert results[-1] == {"name": "test", "value": 42}


def test_str_parser_stream():
    parser = StrParser()
    results = list(parser.stream(iter(["hello", " world"])))
    assert results == ["hello", " world"]


def test_json_parser_array():
    parser = JsonParser()
    result = parser.parse("[1, 2, 3]")
    assert result == [1, 2, 3]


def test_json_parser_nested():
    parser = JsonParser()
    result = parser.parse('{"nested": {"key": [1, 2, 3]}}')
    assert result == {"nested": {"key": [1, 2, 3]}}
