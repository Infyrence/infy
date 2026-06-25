# Contributing to infy

Thanks for your interest in infy. This guide gets you from clone to a passing pull request.

## Philosophy

infy keeps a **deliberately small surface area**. Before adding an API, ask whether it earns its
weight against the project's goals: zero core dependencies, fast cold start, low memory, and a
typed, honest API. Bug fixes, performance work, provider parity, tests, and documentation are
always welcome. New top-level abstractions need a motivating use case first — please open an issue
to discuss before a large PR.

## Development setup

infy ships a Rust extension (`infy_core`), so a **release** build is needed for the full test suite
and for accurate performance numbers (debug builds intentionally fail the perf checks).

```bash
# 1. create an environment (Python 3.10+)
python -m venv .venv && . .venv/Scripts/activate    # Windows: .venv\Scripts\activate

# 2. install dev + build tooling
pip install -e ".[dev]" maturin

# 3. build the Rust core into the active environment (release)
maturin develop --release
```

The framework runs fully on pure Python with the extension absent; the release build is only
required to exercise and benchmark the accelerated paths.

## Quality gates

All four must pass before a PR is merged — they run in CI and should be run locally first:

```bash
ruff check infy tests examples
ruff format --check infy tests examples
mypy infy                 # strict
pytest -q
```

- **Style:** `ruff format` (line length 100). Match the surrounding code — comment density, naming,
  idiom.
- **Types:** `mypy --strict` must pass with no new ignores.
- **Tests:** add tests with every change. Assert real behavior, not just "does not raise."
  Security-relevant code (`infy.governance`) needs tests for the *deny* and *fail-closed* paths,
  not only the happy path.

## Pull requests

1. Fork and branch from `main` (`feature/...` or `fix/...`).
2. Keep PRs focused; one logical change per PR.
3. Write a clear description: what, why, and how it was verified.
4. Ensure the four gates are green and the diff is minimal.

## Developer Certificate of Origin (DCO)

Contributions are accepted under the [Apache-2.0](LICENSE) license. By contributing, you certify
the [DCO](https://developercertificate.org/). Sign off each commit:

```bash
git commit -s -m "fix: ..."
```

which adds a `Signed-off-by: Your Name <you@example.com>` trailer.

## Reporting bugs and vulnerabilities

- **Bugs / features:** open a GitHub issue using the templates.
- **Security vulnerabilities:** do **not** use public issues — see [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
