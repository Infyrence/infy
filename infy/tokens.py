"""Token counting — fast approximation via Rust core.

LangChain relies on tiktoken (OpenAI) or heavy BPE libraries.
infy: ~30 lines, fast whitespace-based approximation via Rust.
For exact counts, use tiktoken directly.
"""

from __future__ import annotations

try:
    import infy_core

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def count_tokens(text: str) -> int:
    """Count tokens in text using a fast approximation.

    ~1 token per 4 chars for English, 1 token per CJK character.
    Uses Rust core when available (10-100x faster than Python).
    """
    if _HAS_RUST:
        return infy_core.count_tokens(text)
    return _py_count_tokens(text)


def count_tokens_batch(texts: list[str]) -> list[int]:
    """Count tokens for multiple texts."""
    if _HAS_RUST:
        return infy_core.count_tokens_batch(texts)
    return [_py_count_tokens(t) for t in texts]


# --- Pure Python fallback ---


def _py_count_tokens(text: str) -> int:
    count = 0
    i = 0
    n = len(text)

    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue

        if c.isascii() and (c.isalnum() or c == "_"):
            start = i
            while i < n and text[i].isascii() and (text[i].isalnum() or text[i] == "_"):
                i += 1
            count += max(1, (i - start + 3) // 4)
        elif c.isascii() and c in ".,!?;:()[]{}\"'-+/\\@#$%^&*~`<>=|":
            count += 1
            i += 1
        else:
            # CJK or other unicode
            cp = ord(c)
            if (0x4E00 <= cp <= 0x9FFF) or (0x3040 <= cp <= 0x30FF) or (0x3000 <= cp <= 0x303F):
                while i < n:
                    cp2 = ord(text[i])
                    if (0x4E00 <= cp2 <= 0x9FFF) or (0x3040 <= cp2 <= 0x30FF):
                        i += 1
                    else:
                        break
                count += 1
            else:
                count += 1
                i += 1

    return max(1, count)
