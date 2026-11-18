#!/usr/bin/env python3
"""Replay cuts on uncut traces, with NO training: the baseline for Thinking Brownout.

For each chosen trace and each cut point u (tokens of thinking already used):
  hard        close the thinking at u, the model must answer now
  notice:G    inject a text notice at u, allow up to G more thinking tokens, then force the close
  budget:G    same, and the notice says "at most G tokens left" (not used by default: the plan
              gives the model no budget, only "finish as fast as you can"; G is a safety cap)
Every run is graded with the same grader as the uncut answer and saved to a JSONL file
(resumable: run the same command again to continue).

  # 1. check the plumbing on one trace (prints what the model does after the notice)
  python cut_bench.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B --smoke
  # 2. size and time of the run, without running it
  python cut_bench.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B --plan
  # 3. the baseline benchmark (test problems)
  python cut_bench.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B
  python cut_report.py runs/Qwen__Qwen3.8-27B.bench.jsonl

  # candidates for SFT data (train problems, several samples, several budgets)
  python cut_bench.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B --split train \\
      --conds hard --k 2 --tag sft
  python make_sft.py runs/Qwen__Qwen3.8-27B.sft.jsonl traces/Qwen__Qwen3.8-27B.jsonl
"""
import argparse
import asyncio
import json
import os
import time
import zlib

import numpy as np

from cutlib import (CLOSE, Server, clean, cut_points, final_box, find_think_open, graded,
                    messages_for, notice_text, parse_conds, read_traces, same, test_problem_ids)
from analyze_boxed import boxed_all


def choose(rows, args):
    test = test_problem_ids(rows)
    keep = []
    for r in rows:
        if not r.get("thinking") or r["L"] < args.min_think:
            continue
        if args.split != "all" and (r["id"] in test) != (args.split == "test"):
            continue
        if r.get("sample", 0) >= args.samples_per_problem:
            continue
        keep.append(r)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(keep)
    if args.max_traces:
        keep = keep[: args.max_traces]
    return keep


def plan(rows, items, conds, args):
    fracs, absolutes = args.fracs, args.abs
    n_cuts = sum(len(cut_points(r["L"], fracs, absolutes)) for r in items)
    ans = np.mean([r.get("answer_tokens") or 0 for r in rows if r.get("finished")])
    per_cut = sum(args.k * ((0.7 * g if kind != "hard" else 0) + 1.2 * ans) for _, kind, g in conds)
    out_tokens = n_cuts * per_cut
    runs = n_cuts * len(conds) * args.k
    prefix = sum(u for r in items for u in cut_points(r["L"], fracs, absolutes))
    rates = [r["completion_tokens"] / r["latency_s"] for r in rows
             if r.get("latency_s") and r.get("completion_tokens") and r["completion_tokens"] > 200]
    orig = sum(r.get("completion_tokens") or 0 for r in rows)
    print(f"traces chosen: {len(items)}   cut points: {n_cuts}   runs: {runs}")
    print(f"thinking prefix to re-read (prefill, cached across the runs of one cut point): {prefix / 1e6:.1f}M tokens")
    print(f"tokens to generate, estimate: {out_tokens / 1e6:.1f}M "
          f"(the uncut collection in this file generated {orig / 1e6:.1f}M, so about {out_tokens / max(orig, 1):.0%} of it)")
    if rates:
        rate = float(np.median(rates))
        print(f"one request decoded at {rate:.0f} tokens/s during collection (median).")
        for c in sorted({args.concurrency, 16, 64}):
            print(f"  if {c} requests run at that speed in parallel: about {out_tokens / (rate * c) / 3600:.1f} hours")
        print("  (rough: the per-request speed drops when more requests run together, and prefill time is not counted;")
        print("   the real run prints a measured estimate after the first runs finish)")


async def prepare(srv, r, args, think_open):
    msgs = messages_for(r, args.system)
    gen = await srv.tok_chat(msgs, args.template_kwargs)
    open_ids = await srv.tok(think_open, cache=True)
    think_ids = await srv.tok(r["thinking"])
    return gen + open_ids, think_ids


async def one_run(srv, r, base_ids, think_ids, u, cond, k, args, verbose=False):
    name, kind, grace = cond
    prefix = base_ids + think_ids[:u]
    seed = args.seed * 100003 + zlib.crc32(f"{r['id']}|{r.get('sample', 0)}|{u}|{name}|{k}".encode()) % 1_000_000
    think_after, n_after, self_term = "", 0, None
    if kind == "hard":
        ids = prefix + await srv.tok("\n" + CLOSE + "\n\n", cache=True)
    else:
        ids = prefix + await srv.tok(notice_text(kind, grace), cache=True)
        p1 = await srv.complete(ids, grace, seed, stop=[CLOSE])
        think_after, n_after = p1["text"], p1["n"]
        self_term = p1["stop_reason"] == CLOSE or (p1["finish"] == "stop" and p1["stop_reason"] is None
                                                   and n_after < grace)
        tail = think_after + (CLOSE + "\n\n" if self_term else "\n" + CLOSE + "\n\n")
        ids = ids + await srv.tok(tail)
    p2 = await srv.complete(ids, args.answer_max, seed + 1)
    answer = clean(p2["text"])
    if verbose:
        print(f"\n--- {name} | cut at {u} of {len(think_ids)} thinking tokens ---")
        if kind != "hard":
            print(f"thinking after the notice ({n_after} tokens, closed by itself: {self_term}):")
            print("   ", clean(think_after)[:600].replace("\n", "\n    "))
        print(f"answer ({p2['n']} tokens, finish {p2['finish']}):")
        print("   ", answer[:500].replace("\n", "\n    "))
    return {"think_after": clean(think_after) if args.keep_text else "", "think_after_tok": n_after,
            "self_term": self_term, "answer": answer, "answer_tok": p2["n"], "answer_finish": p2["finish"],
            "correct": graded(answer, r["gold"])}


async def main_async(args):
    rows = read_traces(args.traces)
    conds = parse_conds(args.conds)
    items = choose(rows, args)
    name = os.path.basename(args.traces).replace(".jsonl", "")
    out_path = args.out or os.path.join("runs", f"{name}.{args.tag}.jsonl")
    if args.plan:
        plan(rows, items, conds, args)
        return
    if not items:
        raise SystemExit("no traces match (check --split, --min-think, --samples-per-problem)")
    srv = Server(args.base_url, args.model, args.concurrency)
    think_open = args.think_open
    if think_open == "auto":
        think_open = await find_think_open(srv, messages_for(items[0], args.system), args.template_kwargs,
                                           verbose=args.smoke)
    if args.smoke:
        r = max(items[:50], key=lambda x: min(x["L"], 4000))        # a reasonably long trace
        base_ids, think_ids = await prepare(srv, r, args, think_open)
        print(f"\nproblem: {r['prompt'][:200]!r}\nuncut: {r['L']} thinking tokens, correct: {graded(r.get('answer'), r['gold'])}")
        print(f"tokenized thinking: {len(think_ids)} tokens (the trace recorded {r.get('think_tokens')})")
        u = cut_points(len(think_ids), [0.5], [])[0]
        for cond in conds:
            res = await one_run(srv, r, base_ids, think_ids, u, cond, 0, args, verbose=True)
            print(f"graded correct: {res['correct']} (gold {r['gold']!r})")
        await srv.close()
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    lock, t0, state = asyncio.Lock(), time.time(), {"n": 0, "tok": 0}
    fout = open(out_path, "a")

    async def do_item(r):
        base_ids, think_ids = await prepare(srv, r, args, think_open)
        L = len(think_ids)
        final = final_box(r.get("answer"))
        tasks = []
        for u in cut_points(L, args.fracs, args.abs):
            keys = [(cond, k, f"{r['id']}|{r.get('sample', 0)}|{u}|{cond[0]}|{k}") for cond in conds for k in range(args.k)]
            if all(key in done for _, _, key in keys):
                continue
            prefix_text = await srv.detok(think_ids[:u])
            n_chars = len(prefix_text) if r["thinking"].startswith(prefix_text) else int(len(r["thinking"]) * u / L)
            has_final = final is not None and any(same(v, final) for _, v in boxed_all(prefix_text))
            meta = {"id": r["id"], "sample": r.get("sample", 0), "source": r.get("source"), "gold": r["gold"],
                    "L": L, "u": u, "frac": round(u / L, 3), "prefix_chars": n_chars,
                    "uncut_correct": graded(r.get("answer"), r["gold"]), "uncut_finished": bool(r.get("finished")),
                    "uncut_answer_tok": r.get("answer_tokens") or 0, "prefix_has_final_box": has_final}

            async def run(cond, k, key, u=u, meta=meta):
                res = await one_run(srv, r, base_ids, think_ids, u, cond, k, args)
                rec = dict(meta, key=key, cond=cond[0], kind=cond[1], grace=cond[2], k=k, **res)
                async with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    state["n"] += 1
                    state["tok"] += res["think_after_tok"] + res["answer_tok"]
                    if state["n"] % 50 == 0:
                        el = time.time() - t0
                        left = (total - len(done) - state["n"]) * el / state["n"]
                        print(f"  {state['n'] + len(done)}/{total} runs | {state['tok'] / el:.0f} tokens/s generated | "
                              f"about {left / 60:.0f} min left", flush=True)

            tasks += [run(cond, k, key) for cond, k, key in keys if key not in done]
        await asyncio.gather(*tasks)

    total = sum(len(cut_points(r["L"], args.fracs, args.abs)) for r in items) * len(conds) * args.k
    print(f"{len(items)} traces, about {total} runs ({len(done)} already done) -> {out_path}")
    gate = asyncio.Semaphore(max(4, args.concurrency // 2))      # traces being worked on at once

    async def gated(r):
        async with gate:
            await do_item(r)

    await asyncio.gather(*[gated(r) for r in items])
    fout.close()
    await srv.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min. Next: python cut_report.py {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"],
                    help="test = the 20%% of problems the predictors never saw (benchmark); train = the rest (SFT data)")
    ap.add_argument("--samples-per-problem", type=int, default=2, help="use samples 0..N-1 of each problem")
    ap.add_argument("--min-think", type=int, default=512, help="skip traces with less thinking than this")
    ap.add_argument("--max-traces", type=int, default=0, help="0 = no limit")
    ap.add_argument("--fracs", type=lambda s: [float(x) for x in s.split(",") if x], default=[0.25, 0.5, 0.75],
                    help="cut after this share of the trace's own thinking")
    ap.add_argument("--abs", type=lambda s: [int(x) for x in s.split(",") if x], default=[],
                    help="also cut after this many tokens, e.g. 2048,8192")
    ap.add_argument("--conds", default="hard,notice:2048")
    ap.add_argument("--k", type=int, default=1, help="samples per condition and cut point")
    ap.add_argument("--answer-max", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--template-kwargs", type=json.loads, default={"enable_thinking": True})
    ap.add_argument("--system", default=None)
    ap.add_argument("--think-open", default="auto", help="text placed before the thinking prefix; auto = detect")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-text", dest="keep_text", action="store_false", help="do not store the generated thinking")
    ap.add_argument("--plan", action="store_true", help="print the size and a time estimate, run nothing")
    ap.add_argument("--smoke", action="store_true", help="one trace, print everything, save nothing")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
