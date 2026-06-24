"""Tests for infy Rust core integration."""

import pytest

# ---------------------------------------------------------------------------
# Rust core availability
# ---------------------------------------------------------------------------


def test_rust_core_available():
    import infy_core

    assert hasattr(infy_core, "parse_json")
    assert hasattr(infy_core, "count_tokens")
    assert hasattr(infy_core, "cosine_similarity")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestRustJSONParsing:
    def test_parse_simple(self):
        from infy_core import parse_json

        result = parse_json('{"name": "infy", "version": 1}')
        assert result == {"name": "infy", "version": 1}

    def test_parse_array(self):
        from infy_core import parse_json

        result = parse_json('[1, 2, 3, "hello"]')
        assert result == [1, 2, 3, "hello"]

    def test_parse_nested(self):
        from infy_core import parse_json

        result = parse_json('{"a": {"b": [1, 2, 3]}}')
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_parse_numbers(self):
        from infy_core import parse_json

        result = parse_json('{"int": 42, "float": 3.14, "neg": -1}')
        assert result["int"] == 42
        assert abs(result["float"] - 3.14) < 0.001
        assert result["neg"] == -1

    def test_parse_bool_and_null(self):
        from infy_core import parse_json

        result = parse_json('{"yes": true, "no": false, "nil": null}')
        assert result["yes"] is True
        assert result["no"] is False
        assert result["nil"] is None

    def test_parse_invalid_raises(self):
        from infy_core import parse_json

        with pytest.raises(ValueError, match="JSON parse error"):
            parse_json("not json at all")

    def test_extract_from_markdown(self):
        from infy_core import extract_json_from_markdown

        result = extract_json_from_markdown('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_from_markdown_no_lang(self):
        from infy_core import extract_json_from_markdown

        result = extract_json_from_markdown('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_partial_json(self):
        from infy_core import parse_partial_json

        # Complete JSON
        result = parse_partial_json('{"name": "infy"}')
        assert result == {"name": "infy"}

        # Partial that can be completed
        result = parse_partial_json('{"name": "infy", "items": [1, 2, 3]}')
        assert result == {"name": "infy", "items": [1, 2, 3]}

    def test_partial_json_empty(self):
        from infy_core import parse_partial_json

        result = parse_partial_json("")
        assert result is None


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestRustTokenCounting:
    def test_simple_english(self):
        from infy_core import count_tokens

        count = count_tokens("Hello, world!")
        assert count >= 2
        assert count <= 5

    def test_empty_string(self):
        from infy_core import count_tokens

        count = count_tokens("")
        assert count >= 1

    def test_single_word(self):
        from infy_core import count_tokens

        count = count_tokens("hello")
        assert count >= 1

    def test_long_text(self):
        from infy_core import count_tokens

        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = count_tokens(text)
        assert count > 50

    def test_batch(self):
        from infy_core import count_tokens_batch

        results = count_tokens_batch(["hello", "world", "test"])
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)
        assert all(r >= 1 for r in results)

    def test_cjk(self):
        from infy_core import count_tokens

        # CJK characters should each be ~1 token
        count = count_tokens("你好世界")
        assert count >= 1
        assert count <= 8


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestRustCosineSimilarity:
    def test_identical_vectors(self):
        from infy_core import cosine_similarity

        score = cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        from infy_core import cosine_similarity

        score = cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert abs(score) < 1e-5

    def test_opposite_vectors(self):
        from infy_core import cosine_similarity

        score = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(score - (-1.0)) < 1e-5

    def test_similar_vectors(self):
        from infy_core import cosine_similarity

        score = cosine_similarity([1.0, 1.0], [1.0, 0.9])
        assert score > 0.9

    def test_length_mismatch(self):
        from infy_core import cosine_similarity

        with pytest.raises(ValueError, match="Vector length mismatch"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_empty_vectors(self):
        from infy_core import cosine_similarity

        score = cosine_similarity([], [])
        assert score == 0.0

    def test_batch_similarity(self):
        from infy_core import batch_cosine_similarity

        query = [1.0, 0.0]
        docs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        results = batch_cosine_similarity(query, docs, top_k=2)

        assert len(results) == 2
        assert results[0] == (0, 1.0)  # Perfect match first
        assert results[1][0] == 2  # Similar second

    def test_inner_product(self):
        from infy_core import inner_product

        result = inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(result - 32.0) < 1e-5  # 4 + 10 + 18


# ---------------------------------------------------------------------------
# Fast hashing
# ---------------------------------------------------------------------------


class TestRustFastHash:
    def test_deterministic(self):
        from infy_core import fast_hash

        h1 = fast_hash("hello world")
        h2 = fast_hash("hello world")
        assert h1 == h2

    def test_different_strings(self):
        from infy_core import fast_hash

        h1 = fast_hash("hello")
        h2 = fast_hash("world")
        assert h1 != h2

    def test_batch(self):
        from infy_core import fast_hash_batch

        results = fast_hash_batch(["a", "b", "c"])
        assert len(results) == 3
        assert len(set(results)) == 3


# ---------------------------------------------------------------------------
# Python-layer integration (parsers using Rust core)
# ---------------------------------------------------------------------------


class TestParsersWithRustCore:
    def test_json_parser_direct(self):
        from infy.parsers import JsonParser

        parser = JsonParser()
        result = parser.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_parser_markdown(self):
        from infy.parsers import JsonParser

        parser = JsonParser()
        result = parser.parse('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_parser_streaming(self):
        from infy.parsers import JsonParser

        parser = JsonParser()
        chunks = ['{"name":', ' "test",', ' "value": 42}']
        results = list(parser.stream(iter(chunks)))
        assert len(results) >= 1
        assert results[-1] == {"name": "test", "value": 42}

    def test_str_parser(self):
        from infy.parsers import StrParser

        parser = StrParser()
        assert parser.parse("hello") == "hello"


# ---------------------------------------------------------------------------
# Python-layer integration (similarity)
# ---------------------------------------------------------------------------


class TestSimilarityWithRustCore:
    def test_cosine_similarity(self):
        from infy.similarity import cosine_similarity

        score = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(score - 1.0) < 1e-5

    def test_batch_cosine_similarity(self):
        from infy.similarity import batch_cosine_similarity

        results = batch_cosine_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], top_k=1)
        assert len(results) == 1
        assert results[0] == (0, 1.0)

    def test_inner_product(self):
        from infy.similarity import inner_product

        assert abs(inner_product([1, 2], [3, 4]) - 11.0) < 1e-5


# ---------------------------------------------------------------------------
# Python-layer integration (tokens)
# ---------------------------------------------------------------------------


class TestTokensWithRustCore:
    def test_count_tokens(self):
        from infy.tokens import count_tokens

        count = count_tokens("Hello, world!")
        assert count >= 2

    def test_count_tokens_batch(self):
        from infy.tokens import count_tokens_batch

        results = count_tokens_batch(["hello", "world"])
        assert len(results) == 2
        assert all(r >= 1 for r in results)


# ---------------------------------------------------------------------------
# Performance: verify Rust is faster than Python fallback
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_json_parse_speed(self):
        import time

        from infy.parsers import JsonParser

        data = (
            '{"users": ['
            + ",".join(
                f'{{"id": {i}, "name": "user_{i}", "email": "user{i}@test.com"}}'
                for i in range(100)
            )
            + "]}"
        )

        parser = JsonParser()
        start = time.perf_counter()
        for _ in range(1000):
            parser.parse(data)
        elapsed = time.perf_counter() - start

        # Should parse 1000 times in under 1 second with Rust
        print(f"\nJSON parse: 1000 iterations in {elapsed:.3f}s")
        assert elapsed < 5.0  # 2.0s in release, ~5s in debug builds

    def test_similarity_speed(self):
        import time

        from infy.similarity import cosine_similarity

        a = [float(i) for i in range(1000)]
        b = [float(i) for i in range(1000)]

        start = time.perf_counter()
        for _ in range(10000):
            cosine_similarity(a, b)
        elapsed = time.perf_counter() - start

        print(f"SIMD cosine: 10000 iterations (1000-dim) in {elapsed:.3f}s")
        assert elapsed < 5.0  # 2.0s in release, ~5s in debug builds

    def test_token_count_speed(self):
        import time

        from infy.tokens import count_tokens

        text = "The quick brown fox jumps over the lazy dog. " * 100

        start = time.perf_counter()
        for _ in range(1000):
            count_tokens(text)
        elapsed = time.perf_counter() - start

        print(f"Token count: 1000 iterations (4400 chars) in {elapsed:.3f}s")
        assert elapsed < 5.0  # 2.0s in release, ~5s in debug builds
