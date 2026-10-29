"""Open-loop load generator: Poisson arrivals of math workflows, each with a
deadline, all sharing one endpoint.

  python -m math_agent.loadgen --rate 0.2 --n 100 --solo-ref runs/solo-x/workflows.jsonl

Deadline = arrival + slack * solo latency of that question (from --solo-ref;
falls back to the solo p50 for questions not in the reference, or to
--default-solo-s without a reference). Slack is drawn uniformly from --slack.

Arrivals are Poisson at --rate, or replayed from the BurstGPT trace (--arrivals
burstgpt) rescaled to the same mean rate. With --dataset mixed the stream draws
from GSM8K, MATH-500 and AIME 2026 at --mix weights, each source with its own
thinking budget (--budgets) and solo latency for deadlines (--solos).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

from . import metrics
from .common import add_common_args, client_from, config_from, examples_from
from .datasets import burstgpt
from .datasets.math import MIXED, source
from .workflow import run_workflow

DEFAULT_MIX = "gsm8k=0.5,math500=0.35,aime2026=0.15"
DEFAULT_BUDGETS = "gsm8k=1024,math500=4096,aime2026=6144"  # vLLM serves 8K of context: 6144 thinking + a 1024 answer fits
DEFAULT_SOLOS = "gsm8k=60,math500=240,aime2026=600"


def parse_map(spec: str | None, cast=float) -> dict:
    out = {}
    for part in (spec or "").split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = cast(v)
    return out


def draw_mixed(pool, n: int, mix: dict[str, float], rng: random.Random):
    """n examples with sources drawn at the mix weights, each uniformly within its source."""
    by = {s: [] for s in MIXED}
    for e in pool:
        by[source(e)].append(e)
    names = [s for s in MIXED if by[s] and mix.get(s, 0) > 0]
    weights = [mix[s] for s in names]
    return [rng.choice(by[rng.choices(names, weights)[0]]) for _ in range(n)]


def load_solo(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    ref = {}
    for line in open(path, encoding="utf-8"):
        w = json.loads(line)
        if not w.get("error"):
            ref[w["example_id"]] = w["latency_s"]
    return ref


async def main(args) -> None:
    rng = random.Random(args.seed)
    solo = load_solo(args.solo_ref)
    pool = examples_from(args)
    if solo and args.only_ref:
        pool = [e for e in pool if e.id in solo]
    mixed = args.dataset == "mixed"
    if mixed:
        examples = draw_mixed(pool, args.n, parse_map(args.mix or DEFAULT_MIX), rng)
    else:
        examples = [rng.choice(pool) for _ in range(args.n)]
    budgets = parse_map(args.budgets or (DEFAULT_BUDGETS if mixed else None), int)
    solos = parse_map(args.solos or (DEFAULT_SOLOS if mixed else None))
    slacks = [float(s) for s in args.slack.split(",")]
    default_solo = metrics.pct(list(solo.values()), 50) if solo else args.default_solo_s
    if args.arrivals == "burstgpt":
        offsets = burstgpt.arrivals(args.n, args.rate, args.seed)
    else:
        offsets, t = [], 0.0
        for _ in range(args.n):
            offsets.append(t)
            t += rng.expovariate(args.rate)

    llm = client_from(args)
    await llm.resolve_model()
    cfg = config_from(args)
    out = Path(args.out or f"runs/load-r{args.rate}-{time.strftime('%Y%m%d-%H%M%S')}")
    calls, wfs = metrics.Jsonl(out / "calls.jsonl"), metrics.Jsonl(out / "workflows.jsonl")
    cfg_dump = {k: v for k, v in vars(args).items() if k != "api_key"}
    (out / "config.json").write_text(json.dumps({**cfg_dump, "default_solo_s": default_solo, "budgets": budgets,
                                                 "solos": solos, "arrival_offsets": [round(x, 3) for x in offsets],
                                                 "burstiness": burstgpt.burstiness(offsets)}, indent=2))

    done: list[dict] = []

    async def one(i: int, ex) -> None:
        arrival = time.time()
        slack = rng.choice(slacks)
        src = source(ex)
        deadline = arrival + slack * solo.get(ex.id, solos.get(src, default_solo))
        wcfg = cfg
        if src in budgets:
            b = budgets[src]
            # retries grow the budget by the sweep's ratio but never past the sweep's own max budget (the context window)
            wcfg = dataclasses.replace(cfg, think_budget=b, max_budget=max(b, min(int(b * cfg.max_budget / max(cfg.think_budget, 1)), cfg.max_budget)))
        r = await run_workflow(ex, llm, wcfg, workflow_id=f"w{i:05d}", arrival=arrival,
                               deadline=deadline, on_call=lambda c: calls.write(c.record()))
        rec = {**r.record(), "slack": slack, "source": src, "budget": wcfg.think_budget}
        rec.pop("trace")
        wfs.write(rec)
        done.append(rec)
        print(f"{len(done):>4}/{args.n} {'OK' if r.correct else 'X '} "
              f"{'on-time' if r.met_deadline else 'LATE   '} {r.latency_s:6.1f}s slack={slack}"
              + (f" ERR {r.error}" if r.error else ""), flush=True)

    tasks = []
    t0 = time.time()
    try:
        for i, ex in enumerate(examples):
            await asyncio.sleep(max(0.0, t0 + offsets[i] - time.time()))
            tasks.append(asyncio.create_task(one(i, ex)))
        await asyncio.gather(*tasks)
    finally:
        calls.close()
        wfs.close()
        await llm.aclose()
    s = {**metrics.summarize(done), "rate": args.rate, "wall_s": round(time.time() - t0, 1),
         "arrivals": args.arrivals, "burstiness": burstgpt.burstiness(offsets)}
    by_source = {}
    for src in sorted({w["source"] for w in done}):
        sub = [w for w in done if w["source"] == src]
        by_source[src] = {k: metrics.summarize(sub).get(k) for k in ("n", "deadline_met", "accuracy",
                                                                     "correct_and_on_time", "thinking_tokens_mean")}
    s["by_source"] = by_source
    by_slack = {}
    for sl in slacks:
        sub = [w for w in done if w["slack"] == sl]
        if sub:
            by_slack[str(sl)] = {k: metrics.summarize(sub)[k] for k in ("n", "deadline_met", "accuracy")}
    s["by_slack"] = by_slack
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print("wrote", out)


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--rate", type=float, required=True, help="workflow arrivals per second")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--slack", default="1.5,3,6", help="comma list of deadline slack multipliers")
    p.add_argument("--solo-ref", help="workflows.jsonl from a solo run")
    p.add_argument("--only-ref", action="store_true", help="sample only questions in the solo reference")
    p.add_argument("--default-solo-s", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arrivals", default="poisson", choices=("poisson", "burstgpt"),
                   help="Poisson at --rate, or BurstGPT trace timing rescaled to --rate")
    p.add_argument("--mix", help=f"mixed: source weights (default {DEFAULT_MIX})")
    p.add_argument("--budgets", help=f"per-source thinking budgets, e.g. {DEFAULT_BUDGETS} (mixed default)")
    p.add_argument("--solos", help=f"per-source solo latency for deadlines, e.g. {DEFAULT_SOLOS} (mixed default)")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
