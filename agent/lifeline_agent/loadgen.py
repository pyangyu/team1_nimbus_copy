"""Open-loop load generator: Poisson arrivals of agentic workflows, each with a
deadline, all sharing one endpoint.

  python -m lifeline_agent.loadgen --rate 0.2 --n 100 --solo-ref runs/solo-x/workflows.jsonl

Deadline = arrival + slack * solo latency of that question (from --solo-ref;
falls back to the solo p50 for questions not in the reference, or to
--default-solo-s without a reference). Slack is drawn uniformly from --slack.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from . import metrics
from .common import add_common_args, client_from, config_from
from .datasets import finqa
from .retrieval import Retriever
from .workflow import run_workflow


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
    pool = finqa.load(args.split)
    if solo and args.only_ref:
        pool = [e for e in pool if e.id in solo]
    examples = [rng.choice(pool) for _ in range(args.n)]
    slacks = [float(s) for s in args.slack.split(",")]
    default_solo = metrics.pct(list(solo.values()), 50) if solo else args.default_solo_s

    retriever = Retriever(finqa.corpus())
    for e in examples:  # build indexes before the clock starts
        retriever.search(e.doc, "warmup", 1)
    llm = client_from(args)
    await llm.resolve_model()
    cfg = config_from(args)
    out = Path(args.out or f"runs/load-r{args.rate}-{time.strftime('%Y%m%d-%H%M%S')}")
    calls, wfs = metrics.Jsonl(out / "calls.jsonl"), metrics.Jsonl(out / "workflows.jsonl")
    (out / "config.json").write_text(json.dumps({**vars(args), "default_solo_s": default_solo}, indent=2))

    done: list[dict] = []

    async def one(i: int, ex) -> None:
        arrival = time.time()
        slack = rng.choice(slacks)
        deadline = arrival + slack * solo.get(ex.id, default_solo)
        r = await run_workflow(ex, retriever, llm, cfg, workflow_id=f"w{i:05d}", arrival=arrival,
                               deadline=deadline, on_call=lambda c: calls.write(c.record()))
        rec = {**r.record(), "slack": slack}
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
            tasks.append(asyncio.create_task(one(i, ex)))
            await asyncio.sleep(rng.expovariate(args.rate))
        await asyncio.gather(*tasks)
    finally:
        calls.close()
        wfs.close()
        await llm.aclose()
    s = {**metrics.summarize(done), "rate": args.rate, "wall_s": round(time.time() - t0, 1)}
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
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
