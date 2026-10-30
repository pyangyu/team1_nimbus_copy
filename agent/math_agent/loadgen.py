"""Open-loop load generator: Poisson arrivals of math workflows, each with a
deadline, all sharing one endpoint.

  python -m math_agent.loadgen --rate 0.2 --n 100 --solo-ref runs/solo-x/workflows.jsonl

Deadline = arrival + slack * solo latency of that question (from --solo-ref;
falls back to the solo p50 for questions not in the reference, or to
--default-solo-s without a reference). Slack is drawn uniformly from --slack.

  python -m math_agent.loadgen --workload math500:1-5 --level-budgets --closed 32
  python -m math_agent.loadgen --workload math500:1-5 --level-budgets --capacity 0.05 \\
      --profile 180:0.7,120:1.5,300:0.7

  python -m math_agent.loadgen --workload math500:1-5 --level-budgets --capacity 0.05 --trace burstgpt --load 0.8

--closed measures capacity: that many workflows stay in flight, and completions
after --warmup-s over --duration-s give capacity_wf_s. --profile replaces the
fixed --rate with Poisson segments at rate * --capacity; arrival times, problems
and slacks are fixed by --seed. --trace replays a window of a real trace at a mean rate of
--load * --capacity with problems woven in by response length (see trace.py).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import metrics, trace, workload
from .common import add_common_args, client_from, config_from, examples_from
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
    if args.workload:
        comps = workload.parse_workload(args.workload)
        pools = workload.pools(comps)
        if solo and args.only_ref:
            pools = [[e for e in pool if e.id in solo] for pool in pools]
        sampler = workload.Sampler(comps, pools, args.seed)
    else:
        pool = examples_from(args)
        if solo and args.only_ref:
            pool = [e for e in pool if e.id in solo]
        sampler = workload.Sampler([workload.Component(args.dataset)], [pool], args.seed)
        sampler.rng = rng
    slacks = [float(s) for s in args.slack.split(",")]
    default_solo = metrics.pct(list(solo.values()), 50) if solo else args.default_solo_s
    profile = workload.parse_profile(args.profile) if args.profile else None
    mode = "closed" if args.closed else "profile" if profile else "trace" if args.trace else "rate"
    trace_tags: list[dict] = []
    trace_meta: dict = {}
    if mode == "rate":
        examples = [sampler.draw() for _ in range(args.n)]
        total = args.n
    elif mode == "profile":
        arr = workload.arrivals(profile, args.capacity, args.seed)
        examples = [sampler.draw() for _ in arr]
        total = len(arr)
    elif mode == "trace":
        rp = trace.replay(trace.load(args.trace_log), [e for pool in sampler.pools for e in pool], args.capacity,
                          start_h=args.trace_start_h, window_h=args.trace_window_h, duration_s=args.trace_s,
                          load=args.load, bin_s=args.bin_s, seed=args.seed)
        arr, examples, profile, total = rp.arrivals, rp.examples, rp.profile, len(rp.arrivals)
        trace_tags = [{"trace_t": r.t, "trace_model": r.model, "trace_response_tokens": r.response_tokens}
                      for r in rp.requests]
        trace_meta = {"trace": args.trace, "log": args.trace_log, "start_h": args.trace_start_h,
                      "window_h": args.trace_window_h, "window_n": rp.window_n, "load": args.load,
                      "bin_s": args.bin_s, "compression": round(args.trace_window_h * 3600 / args.trace_s, 3)}
    else:
        examples, total = [], 0
    srng = random.Random(f"{args.seed}-slack")

    llm = client_from(args)
    await llm.resolve_model()
    cfg = config_from(args)
    name = f"r{args.rate}" if mode == "rate" else mode
    out = Path(args.out or f"runs/load-{name}-{time.strftime('%Y%m%d-%H%M%S')}")
    calls, wfs = metrics.Jsonl(out / "calls.jsonl"), metrics.Jsonl(out / "workflows.jsonl")
    cfg_dump = {k: v for k, v in vars(args).items() if k != "api_key"}
    (out / "config.json").write_text(json.dumps({**cfg_dump, "default_solo_s": default_solo}, indent=2))

    done: list[dict] = []

    def plan(ex):
        if args.level_budgets:
            budget, max_budget, solo_s = workload.budget_for(ex)
            return replace(cfg, think_budget=budget, max_budget=max_budget), solo.get(ex.id, solo_s), budget
        return cfg, solo.get(ex.id, default_solo), cfg.think_budget

    async def one(i: int, ex, slack: float | None = None, extra: dict | None = None) -> None:
        arrival = time.time()
        slack = rng.choice(slacks) if slack is None else slack
        wcfg, solo_s, budget = plan(ex)
        deadline = arrival + slack * solo_s
        tags = {"slack": slack, "band": workload.band(ex), "level": ex.level, "budget": budget, **(extra or {})}
        try:
            r = await run_workflow(ex, llm, wcfg, workflow_id=f"w{i:05d}", arrival=arrival,
                                   deadline=deadline, on_call=lambda c: calls.write(c.record()))
        except asyncio.CancelledError:
            now = time.time()
            rec = {"workflow_id": f"w{i:05d}", "example_id": ex.id, "correct": False, "arrival": arrival,
                   "start": arrival, "end": now, "deadline": deadline, "met_deadline": False,
                   "latency_s": now - arrival, "error": "cancelled", "cancelled": True, **tags}
            wfs.write(rec)
            done.append(rec)
            raise
        rec = {**r.record(), **tags}
        rec.pop("trace")
        wfs.write(rec)
        done.append(rec)
        print(f"{len(done):>4}/{total or '-'} {'OK' if r.correct else 'X '} "
              f"{'on-time' if r.met_deadline else 'LATE   '} {r.latency_s:6.1f}s slack={slack}"
              + (f" ERR {r.error}" if r.error else ""), flush=True)

    async def cancel(tasks) -> None:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    tasks = []
    t0 = time.time()
    try:
        if mode == "rate":
            for i, ex in enumerate(examples):
                tasks.append(asyncio.create_task(one(i, ex)))
                await asyncio.sleep(rng.expovariate(args.rate))
            await asyncio.gather(*tasks)
        elif mode in ("profile", "trace"):
            for i, ((off, seg), ex) in enumerate(zip(arr, examples)):
                await asyncio.sleep(max(0.0, t0 + off - time.time()))
                extra = {"segment": seg, "phase": profile[seg].phase, "offset_s": round(off, 3),
                         **(trace_tags[i] if trace_tags else {})}
                tasks.append(asyncio.create_task(one(i, ex, srng.choice(slacks), extra)))
            await asyncio.sleep(max(0.0, t0 + workload.total_s(profile) - time.time()))
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=args.tail_s)
                await cancel(pending)
        else:
            stop_at = t0 + args.warmup_s + args.duration_s
            counter = itertools.count()

            async def worker() -> None:
                while time.time() < stop_at:
                    i = next(counter)
                    await one(i, sampler.draw(), srng.choice(slacks))

            tasks = [asyncio.create_task(worker()) for _ in range(args.closed)]
            await asyncio.sleep(max(0.0, stop_at - time.time()))
            await cancel(tasks)
    finally:
        calls.close()
        wfs.close()
        await llm.aclose()
    if mode == "closed":
        lo = t0 + args.warmup_s
        win = [w for w in done if not w.get("cancelled") and lo <= w["end"] < stop_at]
        s = {**metrics.summarize(win), "mode": mode, "t0": t0, "concurrency": args.closed,
             "warmup_s": args.warmup_s, "duration_s": args.duration_s, "started": len(done),
             "capacity_wf_s": round(len(win) / args.duration_s, 5),
             "capacity_think_tok_s": round(sum(w["thinking_tokens"] for w in win) / args.duration_s, 2),
             "by_band": metrics.breakdown(win, "band")}
    else:
        s = {**metrics.summarize(done), "mode": mode, "t0": t0, "wall_s": round(time.time() - t0, 1)}
        if mode == "rate":
            s["rate"] = args.rate
        else:
            s.update({"capacity": args.capacity, "profile": args.profile or trace.spec(profile),
                      "profile_s": workload.total_s(profile),
                      "arrivals": total, "tail_s": args.tail_s,
                      "by_phase": metrics.breakdown(done, "phase"), "by_segment": metrics.breakdown(done, "segment"),
                      "by_band": metrics.breakdown(done, "band"),
                      "by_band_phase": {b: metrics.breakdown([w for w in done if w["band"] == b], "phase")
                                        for b in sorted({w["band"] for w in done})}})
            if trace_meta:
                s["trace"] = trace_meta
    by_slack = {}
    for sl in slacks:
        sub = [w for w in (win if mode == "closed" else done) if w["slack"] == sl]
        if sub:
            by_slack[str(sl)] = {k: metrics.summarize(sub).get(k) for k in ("n", "deadline_met", "accuracy")}
    s["by_slack"] = by_slack
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print("wrote", out)


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--rate", type=float, help="workflow arrivals per second (fixed-rate mode)")
    p.add_argument("--n", type=int, default=100, help="workflows in fixed-rate mode")
    p.add_argument("--workload", help="dataset[:levels][=weight],... e.g. math500:1-5 or gsm8k=0.5,math500:4-5=0.5 "
                                      "(replaces --dataset/--levels)")
    p.add_argument("--level-budgets", action="store_true",
                   help="per-problem budget and solo latency by MATH level (1-2: 2048, 3: 4096, 4-5: 6144; GSM8K 1024)")
    p.add_argument("--profile", help="rate profile duration_s:rate,... with rates in units of --capacity")
    p.add_argument("--capacity", type=float, help="capacity C in workflows/s (profile and trace modes)")
    p.add_argument("--trace", choices=trace.TRACES, help="replay arrivals from this trace (trace mode)")
    p.add_argument("--trace-log", default=trace.DEFAULT_LOG, choices=tuple(trace.LOGS),
                   help="BurstGPT log type: conversation (human chat), api (batch API calls) or all")
    p.add_argument("--trace-start-h", type=float, default=trace.DEFAULT_START_H,
                   help="trace mode: window start, hours from the start of the trace")
    p.add_argument("--trace-window-h", type=float, default=trace.DEFAULT_WINDOW_H, help="trace mode: window length")
    p.add_argument("--trace-s", type=float, default=trace.DEFAULT_S, help="trace mode: seconds to replay the window in")
    p.add_argument("--load", type=float, default=trace.DEFAULT_LOAD, help="trace mode: mean arrival rate in units of C")
    p.add_argument("--bin-s", type=float, default=trace.DEFAULT_BIN_S,
                   help="trace mode: bin length for segments and burst/baseline phases")
    p.add_argument("--tail-s", type=float, help="profile mode: after the profile ends, cancel workflows still running "
                                                "after this many seconds (default: wait for all)")
    p.add_argument("--closed", type=int, help="closed loop: keep this many workflows in flight (capacity mode)")
    p.add_argument("--warmup-s", type=float, default=180.0, help="closed loop: seconds before counting completions")
    p.add_argument("--duration-s", type=float, default=600.0, help="closed loop: seconds of counted completions")
    p.add_argument("--slack", default="1.5,3,6", help="comma list of deadline slack multipliers")
    p.add_argument("--solo-ref", help="workflows.jsonl from a solo run")
    p.add_argument("--only-ref", action="store_true", help="sample only questions in the solo reference")
    p.add_argument("--default-solo-s", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if sum(x is not None for x in (args.rate, args.profile, args.closed, args.trace)) != 1:
        p.error("give exactly one of --rate, --profile, --closed, --trace")
    if (args.profile or args.trace) and not args.capacity:
        p.error("--profile and --trace need --capacity")
    if args.trace and not (args.workload and all(c.dataset == "math500" for c in workload.parse_workload(args.workload))):
        p.error("--trace weaves in MATH-500 problems: give --workload math500[:levels]")
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
