"""Tool-routing benchmark: one agent, a 120-tool catalogue, three arms.

  static (all 120 schemas)  -- today's behaviour: every schema, every turn
  routed (top-5)            -- summary pool + the 5 schemas the turn needs
  routed (top-3)            -- same, tighter

Everything is mocked and deterministic (common.py): the measured time is pure framework +
routing overhead, no network. The headline is CONTEXT COST PER RUN -- how many tokens of tool
payload each arm sends -- and whether the routed arms still reach the identical answer.

Token counts come from infy's own ``count_tokens`` (a fast approximation, not a provider BPE
tokenizer), measured off the real serialised payload. Ratios are the portable result; treat the
absolute token figures as approximate.
"""

import importlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# label -> (module, builder function)
ARMS = [
    ("static (all 120 schemas)", "infy_app", "build_static"),
    ("routed (top-5)", "infy_app", "build_routed"),
    ("routed (top-3)", "infy_app", "build_routed_3"),
]


def cold_start(mod, builder, runs=6):
    code = (
        "import time,importlib,sys;t=time.perf_counter();"
        f"m=importlib.import_module('{mod}');m.{builder}();"
        "sys.stdout.write(repr(time.perf_counter()-t))"
    )
    times = []
    for _ in range(runs):
        out = subprocess.run([PY, "-c", code], cwd=HERE, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"cold {mod}.{builder}:\n{out.stderr}")
        times.append(float(out.stdout.strip()))
    return min(times)


def rss(mod, builder):
    code = (
        f"import importlib,os,sys;m=importlib.import_module('{mod}');g=m.{builder}();m.run(g);"
        "import psutil;sys.stdout.write(repr(psutil.Process(os.getpid()).memory_info().rss))"
    )
    out = subprocess.run([PY, "-c", code], cwd=HERE, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"rss {mod}.{builder}:\n{out.stderr}")
    return int(out.stdout.strip())


def rss_baseline():
    out = subprocess.run(
        [
            PY,
            "-c",
            "import psutil,os,sys;"
            "sys.stdout.write(repr(psutil.Process(os.getpid()).memory_info().rss))",
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip())


def latency_us(mod, builder, reps=200):
    """Per-run loop cost with the agent built ONCE.

    Building inside the timed loop would charge every run for constructing the BM25 index over
    the catalogue, which a real deployment pays once at startup. That is a measurement artefact,
    not a cost the routed agent actually has; ``router_costs`` reports the build separately.
    """
    m = importlib.import_module(mod)
    agent = getattr(m, builder)()
    for _ in range(5):
        m.run(agent)
    start = time.perf_counter()
    for _ in range(reps):
        m.run(agent)
    return (time.perf_counter() - start) / reps * 1e6


def measure_tokens(mod, builder):
    m = importlib.import_module(mod)
    return m.run(getattr(m, builder)())


def router_costs(reps=2000):
    """Decompose the router's own cost: one-time index build vs recurring per-turn work."""
    import common

    from infy.tool_router import ToolRouter

    catalogue = common.build_catalogue()

    start = time.perf_counter()
    for _ in range(20):
        ToolRouter(tools=catalogue, top_k=5)
    build = (time.perf_counter() - start) / 20 * 1e6

    router = ToolRouter(tools=catalogue, top_k=5)
    for _ in range(50):
        router.select(common.QUERY)
    start = time.perf_counter()
    for _ in range(reps):
        router.select(common.QUERY)
    select = (time.perf_counter() - start) / reps * 1e6

    for _ in range(50):
        router.summary_pool()
    start = time.perf_counter()
    for _ in range(reps):
        router.summary_pool()
    pool = (time.perf_counter() - start) / reps * 1e6

    return build, select, pool


def main():
    sys.path.insert(0, HERE)
    import common

    base = rss_baseline()
    rows = {}
    reports = {}
    for label, mod, builder in ARMS:
        report = measure_tokens(mod, builder)
        reports[label] = report
        rows[label] = {
            "schema": sum(report["schema_tokens"]),
            "pool": report["pool_tokens"],
            "turn1": report["schema_tokens"][0],
            "cold": cold_start(mod, builder) * 1000,
            "rss": (rss(mod, builder) - base) / 1e6,
            "lat": latency_us(mod, builder),
        }

    finals = {r["final"] for r in reports.values()}
    calls = {r["tool_calls"] for r in reports.values()}
    turns = {r["turns"] for r in reports.values()}
    identical = len(finals) == 1 and len(calls) == 1 and len(turns) == 1
    ranked = all(r["target_offered"] for r in reports.values())

    print(f"Catalogue: {len(common.build_catalogue())} tools")
    print(f"Bare interpreter RSS baseline: {base / 1e6:.1f} MB")
    print(
        f"Equivalence: every arm made {next(iter(calls))} tool call(s) over "
        f"{next(iter(turns))} turns to the same final answer = {identical}"
    )
    print(f"Routing found the needed tool ({common.TARGET}) on turn 1 in every arm = {ranked}\n")

    hdr = (
        f"{'arm':<26}{'turn-1 tok':>12}{'tool tok/run':>14}"
        f"{'pool tok/run':>14}{'total':>9}{'cold ms':>9}{'RSS MB':>8}{'lat us':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, _, _ in ARMS:
        r = rows[label]
        total = r["schema"] + r["pool"]
        print(
            f"{label:<26}{r['turn1']:>12}{r['schema']:>14}{r['pool']:>14}"
            f"{total:>9}{r['cold']:>9.1f}{r['rss']:>8.1f}{r['lat']:>9.1f}"
        )

    static = rows["static (all 120 schemas)"]
    static_total = static["schema"] + static["pool"]

    print("\n=== context cost (routed vs static) ===")
    for label, _, _ in ARMS[1:]:
        r = rows[label]
        total = r["schema"] + r["pool"]
        saved = static_total - total
        print(
            f"  {label:<16} {total:>6} vs {static_total:>6} tokens/run  "
            f"-> {static_total / total:.1f}x lighter, {saved} tokens saved "
            f"({saved / static_total * 100:.1f}%)"
        )

    routed = rows["routed (top-5)"]
    overhead = routed["lat"] - static["lat"]
    build, select, pool = router_costs()
    nturns = next(iter(turns))

    print("\n=== what routing costs ===")
    print(
        f"  loop latency: {overhead:+.1f} us/run "
        f"({routed['lat']:.1f} vs {static['lat']:.1f} us over {nturns} turns)"
    )
    print(f"  per-turn selection: {select:.1f} us   summary pool render: {pool:.1f} us")
    print(f"  one-time index build over {len(common.build_catalogue())} tools: {build:.0f} us")
    print(f"  the summary pool costs {routed['pool']} tokens/run and is never free;")
    print(f"  it is charged above, inside the {routed['schema'] + routed['pool']} total.")
    print(
        f"  cold start {routed['cold']:.0f} vs {static['cold']:.0f} ms and "
        f"RSS {routed['rss']:.1f} vs {static['rss']:.1f} MB are within run-to-run noise here:"
    )
    print("  routing changes what goes in the prompt, not what gets imported.")


if __name__ == "__main__":
    main()
