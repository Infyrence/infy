# Security Policy

## Supported versions

infy is in **alpha** (0.x). Security fixes land on `main` and in the next release. Pin a version
and watch releases until 1.0.

| Version | Supported |
| --- | --- |
| `main` / latest `0.x` | ✅ |
| older `0.x` | ❌ (upgrade) |

## Reporting a vulnerability

**Please do not open a public issue for security reports.**

Report privately via GitHub Security Advisories:
[**Report a vulnerability**](https://github.com/Infyrence/infy/security/advisories/new). This
opens a private channel with the maintainers.

Please include:

- a description of the issue and its impact,
- the affected component (e.g. `infy.governance`, a provider, the Rust core),
- steps to reproduce or a proof of concept,
- any suggested remediation.

We aim to acknowledge within **3 business days**, agree on a disclosure timeline, and credit
reporters who wish to be named.

## Scope notes

- **Governance is a security boundary.** Bugs in `infy.governance` that allow a *fail-open* path
  (a tool executing when policy/risk/approval should have denied it), an audit-integrity break, or
  an approval bypass are treated as high severity. The module is designed to **fail closed**; a
  case where it does not is exactly what we want to hear about.
- **Tamper-evidence, not tamper-proofing.** The local `AuditLog` is tamper-*evident* (hash-chained,
  `verify()`-able) and optionally HMAC-keyed. It is **not** tamper-*proof* against an attacker who
  controls the process — that requires external anchoring (see `infy/governance/README.md`). This
  is a documented limitation, not a vulnerability.
- **The pure-Python core has no third-party runtime dependencies**, which shrinks the supply-chain
  surface. Provider SDKs and the Rust extension are optional and isolated.
