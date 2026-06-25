# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor releases may include
breaking changes.

## [Unreleased]

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

[Unreleased]: https://github.com/Infyrence/infy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Infyrence/infy/releases/tag/v0.1.0
