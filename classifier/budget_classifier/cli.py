from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .buckets import DEFAULT_EDGES, Buckets
from .classifier import BudgetClassifier
from .jev import JevPrior


def build(args: argparse.Namespace) -> BudgetClassifier:
    buckets = Buckets(tuple(int(e) for e in args.edges.split(","))) if args.edges else Buckets()
    jev = None
    if args.jev_url:
        jev = JevPrior(buckets, args.jev_url, api_key=os.environ.get(args.jev_key_env) or None,
                       model=args.jev_model, timeout=args.jev_timeout)
    clf = BudgetClassifier(buckets, jev=jev, warmup=args.warmup, quantile=args.quantile,
                           censored=args.censored, active_at=args.active_at,
                           reset_on_upgrade=not args.no_reset_on_upgrade)
    if args.state and Path(args.state).exists():
        clf.load(args.state)
    return clf


async def replay(args: argparse.Namespace) -> None:
    clf = build(args)
    rows = (json.loads(line) for line in open(args.data) if line.strip())
    out = open(args.out, "w") if args.out else sys.stdout
    i = 0
    for i, r in enumerate(rows, 1):
        rid = str(r.get("id", i))
        await clf.predict(rid, r["prompt"], r.get("model"))
        res = clf.observe(rid, int(r["thinking_tokens"]), finished=bool(r.get("finished", True)), model=r.get("model"))
        if args.trace:
            out.write(json.dumps({"i": i, "model": clf.current_model, **res}) + "\n")
        if args.report_every and i % args.report_every == 0:
            v = clf.stats()["versions"][str(clf.current_model)]
            out.write(json.dumps({"i": i, "model": clf.current_model, **v}) + "\n")
    out.write(json.dumps({"i": i, "final": clf.stats()}) + "\n")
    if args.state:
        clf.save(args.state)
    if clf.jev is not None:
        await clf.jev.aclose()


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(build(args), state_path=args.state, save_every=args.save_every),
                host=args.host, port=args.port, log_level="warning")


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="budget-classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", help="npz file to load at start and save to")
    common.add_argument("--edges", help=f"bucket edges, default {','.join(map(str, DEFAULT_EDGES))}")
    common.add_argument("--warmup", type=int, default=200)
    common.add_argument("--quantile", type=float, default=0.5)
    common.add_argument("--censored", choices=["update", "skip"], default="update")
    common.add_argument("--active-at", type=float, default=0.05)
    common.add_argument("--no-reset-on-upgrade", action="store_true")
    common.add_argument("--jev-url", help="Jev /v1/systemone endpoint (hosted Jev or Open-Jev); off if unset")
    common.add_argument("--jev-model", default="open-jev")
    common.add_argument("--jev-key-env", default="JEV_API_KEY")
    common.add_argument("--jev-timeout", type=float, default=2.0)

    s = sub.add_parser("serve", parents=[common])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8792)
    s.add_argument("--save-every", type=int, default=100)

    r = sub.add_parser("replay", parents=[common])
    r.add_argument("data", help="JSONL rows: prompt, thinking_tokens, optional id, model, finished")
    r.add_argument("--out")
    r.add_argument("--report-every", type=int, default=100)
    r.add_argument("--trace", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "serve":
        serve(args)
    else:
        asyncio.run(replay(args))


if __name__ == "__main__":
    cli()
