<!-- Thanks for contributing to infy. Keep PRs focused: one logical change. -->

## What

<!-- What does this change do? -->

## Why

<!-- Motivation / linked issue (Fixes #...). New abstractions should reference a discussion issue. -->

## How it was verified

<!-- Commands run, scenarios covered, before/after numbers if performance-related. -->

## Checklist

- [ ] `ruff check infy tests examples` passes
- [ ] `ruff format --check infy tests examples` passes
- [ ] `mypy infy` passes (strict, no new ignores)
- [ ] `pytest -q` passes; new tests added for the change (incl. deny / fail-closed paths if governance)
- [ ] Docs / CHANGELOG updated if user-facing
- [ ] Commits signed off (DCO): `git commit -s`
