"""Run workflows one at a time (no contention).

  python -m lifeline_agent.run --index 0 -v        # one question, full trace
  python -m lifeline_agent.run --n 50              # solo calibration run

The solo run's workflows.jsonl is the reference the load generator uses to
set deadlines (slack x solo latency).
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


def pick(examples, args):
    if args.id:
        return [e for e in examples if e.id == args.id]
    if args.index is not None:
        return [examples[args.index]]
    rng = random.Random(args.seed)
    return rng.sample(examples, min(args.n, len(examples)))


async def main(args) -> None:
    examples = pick(finqa.load(args.split), args)
    retriever = Retriever(finqa.corpus())
    llm = client_from(args)
    cfg = config_from(args)
    out = Path(args.out or f"runs/solo-{time.strftime('%Y%m%d-%H%M%S')}")
    calls, wfs = metrics.Jsonl(out / "calls.jsonl"), metrics.Jsonl(out / "workflows.jsonl")
    done = []
    try:
        for i, ex in enumerate(examples):
            r = await run_workflow(ex, retriever, llm, cfg, on_call=lambda c: calls.write(c.record()))
            rec = r.record()
            wfs.write(rec)
            done.append(rec)
            mark = "OK " if r.correct else ("~  " if r.correct_lenient else "X  ")
            print(f"{i + 1:>3}/{len(examples)} {mark} {r.latency_s:6.1f}s rounds={r.rounds} "
                  f"think={r.thinking_tokens:<5} pred={r.pred!s:<12.12} gold={r.gold!s:<12.12} {ex.id}"
                  + (f"  ERR {r.error}" if r.error else ""))
            if args.verbose:
                print("  Q:", ex.question)
                print("  gold program:", ex.gold_program)
                for t in r.trace:
                    print("   ", json.dumps(t, default=str)[:400])
    finally:
        calls.close()
        wfs.close()
        await llm.aclose()
    s = metrics.summarize(done)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print("wrote", out)


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--id")
    p.add_argument("--index", type=int)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
