# Tool-routing benchmark

**Question:** what does static tool-schema injection actually cost, and does routing around it
pay for itself? infy's weight claim is about the costs paid on every request. For an agent wired
to a large catalogue, the biggest of those is the tool schemas themselves — re-serialised into
every model call, on every iteration of the loop. This measures the tax and the fix.

## Setup

One agent run = a support agent that issues **one tool call against a 120-tool catalogue**, then
writes a final answer (2 model turns). 120 is roughly what an agent wired to a handful of MCP
servers ends up holding.

The catalogue is realistic rather than filler: twelve service domains (GitHub, Jira, Slack, S3,
Postgres, Stripe, Kubernetes, Datadog, SendGrid, Twilio, Salesforce, Notion) crossed with ten
operations that make sense for all of them (`list`, `get`, `create`, `update`, `delete`, `search`,
`export`, `audit`, `permissions`, `health`). Every tool has a distinct name, a distinct
description, and a four-property typed argument schema. A catalogue of near-identical filler would
flatter the router by making the target trivially separable.

The model and tools are **mocked and deterministic** (`common.py`) — zero network. Three arms:

| arm | what it is |
| --- | --- |
| static (all 120 schemas) | today's behaviour: every schema, every turn |
| routed (top-5) | `ToolRouter` — one-line summary of all 120 + full schemas for the top 5 |
| routed (top-3) | same, tighter |

Tools run **sequentially** (`parallel_tools=False`) so the only variable between arms is routing.
Token counts come from infy's own `count_tokens` — a fast approximation, not a provider BPE
tokenizer — measured off the **real serialised payload**, not estimated from the catalogue.

Run it: `python bench.py` (from this directory).

## Results (representative; mocked, single machine)

```
Catalogue: 120 tools
Equivalence: every arm made 1 tool call over 2 turns to the same final answer = True
Routing found the needed tool (stripe_search) on turn 1 in every arm = True

arm                         turn-1 tok  tool tok/run  pool tok/run    total  cold ms  RSS MB   lat us
static (all 120 schemas)         23917         47834             0    47834      355      9.3   5993.7
routed (top-5)                    1000          2000          3532     5532      356      9.2   1187.1
routed (top-3)                     600          1200          3532     4732      311      9.4    994.7
```

**Context cost.** Static injection spends **23,917 tokens on turn 1** and **47,834 per run** — and
that is the floor, since it is re-sent every iteration. Routing at top-5 costs **5,532 tokens/run
all in**: **8.6× lighter, 88.4 % saved.** At top-3, **10.1× lighter, 90.1 % saved.**

**The summary pool is not free** and is charged above: 1,766 tokens, re-sent on both turns for
3,532 per run. It is 64 % of the routed arm's total cost. The saving comes from the schemas
(47,834 → 2,000), not from the pool being cheap.

**Routing is also faster, not a latency trade.** At top-5 the loop runs in **1,187 µs vs 5,994 µs**
— **5× faster per run**, because serialising 120 schemas twice costs far more than ranking 120
tools and serialising 5. Decomposed:

| cost | measured |
| --- | --- |
| per-turn selection (BM25 over 120 tools + rank) | **297 µs** |
| summary pool render | 0.2 µs (rendered once at construction) |
| one-time index build over 120 tools | ~4.0 ms, at startup, paid once |

Against the ~1–2 s LLM call it feeds, 297 µs of selection is **~0.02 % overhead** — the same order
as the governance layer's per-call cost, and for the same reason: it is a plain in-process
function call.

**Cold start and RSS are unchanged** (356 vs 355 ms, 9.2 vs 9.3 MB, both within run-to-run noise).
Routing changes what goes *in the prompt*, not what gets imported.

## Honest caveats

- **Token figures are approximations.** `count_tokens` is infy's fast heuristic, not `tiktoken`.
  The **ratios** are the portable result; treat the absolute counts as indicative.
- **This is a single-tool-call scenario.** A run that needs tools from several domains promotes
  more of them (and, with `sticky=True`, keeps them), so the saving shrinks. The 8.6× here is the
  favourable end of a real range, not a universal number.
- **The pool/schema crossover is catalogue-dependent.** Below roughly 15–20 tools the summary pool
  costs more than the schemas it replaces, and routing is a net loss. `ToolRouter` already refuses
  to slice when the catalogue is no larger than `top_k`, but between that and ~20 tools it is a
  judgement call — measure before enabling.
- **Retrieval can miss.** The router promoted the needed tool on turn 1 in every arm here, on a
  query with strong lexical overlap. A paraphrased query with no shared terms is the failure mode;
  that is what `always=[...]` pinning, the summary pool's escape hatch (name a tool in prose and it
  is promoted next turn), and the optional `embeddings=` dense path exist to cover.
- **Routing is not a security control.** A tool whose schema was never promoted still executes if
  the model names it. `infy.governance` remains the only authority on what is allowed to run.
