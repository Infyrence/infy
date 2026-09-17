# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor releases may include
breaking changes.

## [Unreleased]

### Added

- **Agno integration (`infy.integrations.agno`)** — `govern_hook` / `agovern_hook` build an Agno
  `tool_hook` that routes every tool call through infy governance. A denied or unapproved action
  never reaches the entrypoint; the model receives the block reason and adapts, and every decision
  lands in the tamper-evident audit chain. One agent-level hook covers plain callables, `Function`
  objects and every function of a `Toolkit`. Tested against the library, with
  `examples/agno_governance_demo.py`.

## [0.2.0] - 2026-09-16

First release published to PyPI. `0.1.0` was an internal milestone and was never tagged or
uploaded, so `pip install infy` starts here.

### Added

- **Governance (`infy.governance`)** — an optional, in-process control plane for the agent loop:
  - deny-by-default, forbid-overrides-permit, fail-closed `PolicyEngine` (pure Python, Cedar-style
    semantics) with an ergonomic `Policy(allow / deny / require_approval / approve_when)` surface;
  - `RiskEngine` risk tiering from static tool metadata (an unprofiled side-effecting tool is HIGH
    and escalates to approval by default);
  - human approval gate via a pluggable `Approver` (default `DenyAll`, fail-closed);
  - tamper-evident `AuditLog` — append-only, SHA-256 / optional HMAC hash chain, thread-safe,
    `verify()`-able;
  - four hook seams on `create_agent` / `create_async_agent` (before/after model and tool). The
    core loop is unchanged when `governance` is omitted.
- Optional risk metadata on `Tool` and the `@tool` decorator (`risk_tier`, `verb`, `side_effect`, …).
- `examples/governance_demo.py` and a governance overhead benchmark.
- **Tool routing (`infy.tool_router`)** — an optional `ToolRouter` for large catalogues. Keeps a
  compact, turn-stable one-line summary of every tool resident in the prompt prefix and promotes
  the full JSON Schema of only the top-k relevant tools per turn:
  - dependency-free hybrid ranking — BM25 over an inverted index of tool names and descriptions,
    optionally fused with dense embedding similarity by `reciprocal_rank_fusion` (RRF, k=60);
  - `always=[...]` pinning, `sticky` monotonic promotion (keeps the tool list append-only so a
    provider's prompt cache keeps hitting), and an escape hatch — a tool named in the model's
    prose is promoted on the next turn;
  - opt-in via `create_agent(..., tool_router=...)`. The loop is unchanged when omitted.
  - Measured on a 120-tool catalogue: **8.6x less tool-payload context and 5x faster per run**,
    at ~297 µs per selection. Routing is a **context optimisation, not a security boundary** — an
    unpromoted tool still executes if the model names it; `infy.governance` remains the authority.
- A tool-routing benchmark (`tool_routing_bench/`).
- **Intent structure (`infy.intent`)** — `Objective`: an immutable, structured statement of what
  a run is for (`goal`, `constraints`, `success_criteria`), held outside the message array and
  re-stated every turn to fight goal drift and multi-turn prompt injection. Rendered at the head
  of the conversation and moved as a single short reminder to just before each model call; the
  previous reminder is removed rather than accumulated, so the cost is flat. Opt-in via
  `create_agent(..., objective=...)`; the loop is unchanged when omitted.
- **`FactSheetMemory` (`infy.memory`)** — keeps recent turns verbatim and folds older tool output
  into a de-duplicated, bounded list of established facts. Uses a model to extract when given
  one, and falls back to recording tool results verbatim when not (or when extraction fails).
- **Bi-temporal memory (`infy.temporal`)** — `BiTemporalMemory` / `Fact`, storing facts on both
  valid time (when it was true in the world) and transaction time (when we believed it). Makes
  `retract` ("the world changed", history preserved) distinct from `correct` ("we were wrong",
  removed from belief but kept in `history()`), and supports `as_of` queries on either axis, so
  "what did we believe when we acted?" is answerable. `BiTemporalStore` is a protocol seam for a
  durable backend.

### Fixed

- **Anthropic provider dropped all but the last system message.** `_convert_messages` assigned
  rather than merged, so once the agent loop injected a second system message (a tool summary
  pool or an objective anchor) the caller's `system_prompt` was silently discarded. Now merged,
  matching the Gemini provider, with a regression test alongside the existing Gemini one.

### Changed

- **License: MIT → Apache-2.0** (adds an explicit patent grant). Added `LICENSE`, `NOTICE`.
- Repository hygiene: `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog, and
  GitHub issue / PR templates.

### Removed

- The hand-written static documentation site (`docs/`); the README and module READMEs are the
  source of truth.

## [0.1.0]

### Added

- Initial release: zero-dependency Python core — messages, `ChatModel` protocol, runnables
  (`Sequence` / `Parallel` / `Lambda`), parsers, tools, structured output.
- Tool-calling agents (`create_agent` / `create_async_agent`, sync + async).
- Pregel-style `StateGraph` runtime — typed channels, reducers, conditional routing, `Send`
  fan-out, checkpointing, and interrupt/resume for human-in-the-loop.
- Provider extras: OpenAI, Anthropic, Google Gemini / Vertex, Ollama.
- RAG, retrievers, memory, vector stores, text splitters, tokenisation.
- `infy_core` — a Rust (PyO3, `abi3`) SIMD extension for JSON parsing, cosine similarity, and
  tokenisation, with graceful pure-Python fallback.

[Unreleased]: https://github.com/Infyrence/infy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Infyrence/infy/releases/tag/v0.2.0
[0.1.0]: https://github.com/Infyrence/infy/releases/tag/v0.1.0
