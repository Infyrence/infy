"""Shared, offline, deterministic scenario for the tool-routing benchmark.

One agent run = a support agent that issues a single tool call against a **120-tool catalogue**,
then writes a final answer. 120 is the catalogue size at which static schema injection is
reported to cost tens of thousands of tokens per turn; it is also roughly what an agent wired to
a handful of MCP servers ends up holding.

The catalogue is realistic rather than generated noise: twelve service domains crossed with ten
operations that all make sense for every domain, so every tool has a distinct name and a distinct
description and the router's ranking has real lexical signal to work against. A catalogue of
near-identical filler would flatter the router by making the target trivially separable.

Everything is mocked and deterministic: no network, no API keys. Every arm executes the same tool
call and reaches the same final answer, so the arms are byte-comparable.
"""

from __future__ import annotations

from typing import Any

from infy.tools import Tool

_DOMAINS: list[tuple[str, str]] = [
    ("github", "GitHub repository"),
    ("jira", "Jira project"),
    ("slack", "Slack workspace"),
    ("s3", "S3 bucket"),
    ("postgres", "Postgres database"),
    ("stripe", "Stripe billing"),
    ("kubernetes", "Kubernetes cluster"),
    ("datadog", "Datadog monitoring"),
    ("sendgrid", "SendGrid mailer"),
    ("twilio", "Twilio messaging"),
    ("salesforce", "Salesforce CRM"),
    ("notion", "Notion workspace"),
]

_OPERATIONS: list[tuple[str, str]] = [
    ("list", "List the {noun} records visible to the caller."),
    ("get", "Fetch a single {noun} record by its identifier."),
    ("create", "Create a new {noun} record from the supplied fields."),
    ("update", "Update fields on an existing {noun} record."),
    ("delete", "Permanently delete a {noun} record. This cannot be undone."),
    ("search", "Search {noun} records by a free-text query and return the matches."),
    ("export", "Export {noun} records to a downloadable file."),
    ("audit", "Read the audit history for a {noun} record."),
    ("permissions", "Read or change the access permissions on a {noun} record."),
    ("health", "Report the current health and rate-limit status of the {noun} integration."),
]

# A schema with real weight: four typed, described properties, which is what a genuine
# integration tool looks like on the wire. A one-argument toy tool would understate the tax.
_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "identifier": {"type": "string", "description": "Identifier of the target record."},
        "query": {"type": "string", "description": "Free-text query or filter expression."},
        "limit": {
            "type": "integer",
            "description": "Maximum number of records to return.",
            "default": 50,
        },
        "include_archived": {
            "type": "boolean",
            "description": "Whether to include archived records in the result set.",
            "default": False,
        },
    },
    "required": ["identifier"],
}

QUERY = "search our stripe billing records for the customer's failed charge and say why it failed"
TARGET = "stripe_search"
TARGET_ARGS = {"identifier": "cus_4821", "query": "failed charge"}
TARGET_RESULT = "charge ch_9f21: declined (insufficient_funds)"
FINAL = "The charge failed: the card was declined for insufficient funds."


def _stub(**kwargs: Any) -> str:
    return TARGET_RESULT


def build_catalogue() -> list[Tool]:
    """The 120-tool catalogue, in a fixed order."""
    return [
        Tool(
            name=f"{prefix}_{op}",
            description=template.format(noun=noun),
            func=_stub,
            args_schema=_ARGS_SCHEMA,
        )
        for prefix, noun in _DOMAINS
        for op, template in _OPERATIONS
    ]
