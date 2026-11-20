#!/usr/bin/env python3
"""SFT targets for the cut adapter: the shortest continuation that is correct, DERIVED, and DECISIVE.
Only traces the model solved uncut are used.

At every cut point the model gets a data-generation instruction (never seen at deployment; the
adapter only sees [cut]) that asks for efficient, decisive thinking, and we sample K continuations.
A continuation qualifies only if all of these hold:
  closed      it ended its thinking by itself before the safety cap
  correct     the answer written after it is graded correct
  derived     with the answer HIDDEN inside the new thinking (replaced by "?"), the model still
              answers correctly in >= 3 of 4 tries, without redoing the work in the answer
              -> the steps lead to the answer; a lucky early guess does not pass
  decisive    few hesitation phrases (wait, hmm, alternatively, let me double-check, ...)
Candidate 0 is "stop right now" (close the thinking at the cut): it qualifies if all 4 answers
are correct. The target is the qualified candidate with the lowest cost: thinking tokens plus the
answer tokens beyond the uncut answer's length (moving work into the answer is charged; a bare answer
earns nothing).

  # smoke: 20 train traces, cuts at 25% and 50%, then read decisive_smoke.md
  python decisive_sample.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B --smoke 20
  # full run, train problems of the main collection (resumable); 3 random cut points per trace
  python decisive_sample.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B
  # the extra DeepScaleR problems are all training problems: use --split all
  python decisive_sample.py traces/Qwen__Qwen3.8-27B.more.jsonl --model Qwen/Qwen3.8-27B --split all
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
import zlib
from collections import defaultdict

import numpy as np

from cut_bench import prepare
from cutlib import (CLOSE, Server, clean, cut_points, dither_count, final_box, find_think_open, graded,
                    messages_for, read_traces, test_problem_ids)
from analyze_boxed import boxed_all, same
from prune_tail import mask_answer

INSTRUCTION = ("\n\n[Your thinking is being cut short. From here, think efficiently and decisively: commit to "
               "the most direct approach and carry it through. Do every step the answer actually depends on, "
               "but do not re-verify results you already have, do not try alternative methods, and do not "
               "second-guess. Backtrack only if you find a concrete error. When the answer is determined, "
               "stop and give it.]\n\n")

def is_decisive(count, n_tok, args):
    free, rate = getattr(args, "dither_free", 2), getattr(args, "max_dither", 8.0)   # make_sft has no such flags
    return count <= free or count * 1000 / max(n_tok, 1) <= rate


def state_of(meta):
    if not meta["uncut_correct"]:
        return "unsolved"
    return "concluded" if meta["prefix_has_final_box"] else "midway"


async def answers(srv, ids, n, max_tokens, seed):
    outs = await asyncio.gather(*[srv.complete(ids, max_tokens, seed + j) for j in range(n)])
    return [(clean(o["text"]), o["n"], o["finish"]) for o in outs]


async def one_cut(srv, r, base_ids, think_ids, u, args, inst_ids, close_ids):
    key = f"{r['id']}|{r.get('sample', 0)}|{u}"
    seed0 = args.seed * 100003 + zlib.crc32(key.encode()) % 1_000_000
    gold = r["gold"]
    prefix = base_ids + think_ids[:u]
    prefix_text = await srv.detok(think_ids[:u])
    uncut_final = final_box(r.get("answer"))
    meta = {"key": key, "id": r["id"], "sample": r.get("sample", 0), "source": r.get("source"), "gold": gold,
            "L": len(think_ids), "u": u, "frac": round(u / len(think_ids), 3),
            "prefix_chars": len(prefix_text) if r["thinking"].startswith(prefix_text)
            else int(len(r["thinking"]) * u / len(think_ids)),
            "uncut_correct": graded(r.get("answer"), gold), "uncut_answer_tok": r.get("answer_tokens") or 0,
            "prefix_has_final_box": uncut_final is not None and any(same(v, uncut_final) for _, v in boxed_all(prefix_text))}
    meta["state"] = state_of(meta)
    redo_limit = max(2 * meta["uncut_answer_tok"], args.min_redo)

    # candidate 0: stop right now
    hard = await answers(srv, prefix + close_ids, args.checks, args.answer_max, seed0 + 500)
    hard_hits = sum(graded(a, gold) for a, _, _ in hard)
    hard_tok = float(np.median([n for _, n, _ in hard]))
    hard_rec = {"hits": hard_hits, "answer_tok": hard_tok, "answer": hard[0][0]}

    async def cand(j):
        s = seed0 + 17 * (j + 1)
        p1 = await srv.complete(prefix + inst_ids, args.cap, s, stop=[CLOSE])
        text, n = p1["text"], p1["n"]
        closed = p1["stop_reason"] == CLOSE or (p1["finish"] == "stop" and n < args.cap)
        head = text.lstrip()
        c = {"j": j, "think": clean(text), "think_tok": n, "closed": closed,
             "echoed": head.startswith(INSTRUCTION.strip()[:40]) or INSTRUCTION.strip()[:60] in text,
             "fake_note": head.startswith("[") or (len(r["prompt"].strip()) > 30 and r["prompt"].strip()[:60] in text),
             "dither": dither_count(text)}
        c["decisive"] = is_decisive(c["dither"], n, args)
        tail = text + (CLOSE + "\n\n" if closed else "\n" + CLOSE + "\n\n")
        p2 = await srv.complete(prefix + inst_ids + await srv.tok(tail), args.answer_max, s + 1)
        c["answer"], c["answer_tok"] = clean(p2["text"]), p2["n"]
        c["correct"] = graded(c["answer"], gold)
        c["derived_hits"], c["check_tok"], c["n_masked"] = None, None, 0
        if closed and c["correct"]:
            masked, nm = mask_answer(text, final_box(c["answer"]) or gold)
            masked, nm2 = mask_answer(masked, gold)
            c["n_masked"] = nm + nm2
            ids = prefix + inst_ids + await srv.tok(masked + CLOSE + "\n\n")
            chk = await answers(srv, ids, args.checks, args.check_max, s + 101)
            c["derived_hits"] = sum(graded(a, gold) for a, _, _ in chk)
            c["check_tok"] = float(np.median([m for _, m, _ in chk]))
        c["derived"] = bool(c["derived_hits"] is not None and c["derived_hits"] >= args.need
                            and c["check_tok"] <= redo_limit)
        c["ok"] = (c["closed"] and c["correct"] and c["derived"] and c["decisive"]
                   and not c["echoed"] and not c["fake_note"])
        return c

    cands = await asyncio.gather(*[cand(j) for j in range(args.k)])
    return select(dict(meta, hard=hard_rec, cands=cands), args)


def select(rec, args):
    """Pick the target: the qualified candidate with the lowest cost after the cut, where
        cost = thinking tokens + answer tokens BEYOND the uncut answer's length.
    Moving work from the thinking into the answer is charged (the answer grows past the uncut one),
    but a short answer earns nothing: a bare \\boxed{} cannot beat a candidate that thinks less and
    keeps the uncut model's explained answer. Ties go to fewer total tokens, then fewer thinking tokens.
    A continuation that thinks longer than the uncut run still did after the cut is never a target, and
    cut points of unsolved traces (the uncut answer was wrong) have none.
    "Stop now" must be correct in --need-stop of 4 tries. Re-run with --report to re-select."""
    ua = rec["uncut_answer_tok"]
    if rec["state"] == "unsolved":                      # uncut run was wrong: never a target
        rec["chosen"] = rec["chosen_cost"] = rec["chosen_tok"] = rec["chosen_total"] = None
        return rec
    left = rec["L"] - rec["u"]                          # thinking the uncut run still did after the cut
    redo_limit = max(2 * ua, args.min_redo)
    h = rec["hard"]
    h["ok"] = h["hits"] >= args.need_stop and h["answer_tok"] <= redo_limit
    over = lambda a: max(0, a - ua)
    for c in rec["cands"]:                              # re-apply the CURRENT hesitation list, so a
        if c.get("think") is not None and c["closed"]:  # wider list takes effect without re-sampling
            c["dither"] = dither_count(c["think"])
            if re.match(r"\s*\W{0,6}\]", c["think"]):     # e.g. "...]": the model closes the instruction's
                c["fake_note"] = True                     # bracket instead of thinking; never a target
            c["decisive"] = is_decisive(c["dither"], c["think_tok"], args)
            c["ok"] = (c["closed"] and c["correct"] and c["derived"] and c["decisive"]
                       and not c["echoed"] and not c.get("fake_note", False))
    pool = ([(over(h["answer_tok"]), h["answer_tok"], 0, -1)] if h["ok"] else []) + \
           [(c["think_tok"] + over(c["answer_tok"]), c["think_tok"] + c["answer_tok"], c["think_tok"], c["j"])
            for c in rec["cands"] if c["ok"] and c["think_tok"] <= left]
    best = min(pool) if pool else None
    rec["chosen"] = best[3] if best else None
    rec["chosen_cost"] = best[0] if best else None
    rec["chosen_tok"] = best[2] if best else None
    rec["chosen_total"] = best[1] if best else None
    return rec


def pick(rows, args):
    if args.split == "all":
        keep = list(rows)
    else:
        test = test_problem_ids(rows)
        keep = [r for r in rows if (r["id"] in test) == (args.split == "test")]
    keep = [r for r in keep if r.get("thinking") and r["L"] >= args.min_think
            and r.get("sample", 0) < args.samples_per_problem
            and graded(r.get("answer"), r["gold"])]          # only traces the model solved uncut
    rng = random.Random(args.seed)
    rng.shuffle(keep)
    if args.smoke:
        keep = keep[: args.smoke]
    elif args.max_traces:
        keep = keep[: args.max_traces]
    return keep


ORDER = {"early": 0, "middle": 1, "late": 2}


def bucket(f):
    return "early" if f < 0.35 else "middle" if f < 0.6 else "late"


def report(recs, path_md=None, n_show=0):
    n_uns = sum(r["state"] == "unsolved" for r in recs)
    recs = [r for r in recs if r["state"] != "unsolved"]
    print(f"\n{len(recs)} cut points" + (f" ({n_uns} more from unsolved traces left out: never a target)" if n_uns else ""))
    allc = [c for r in recs for c in r["cands"]]
    if allc:
        n = len(allc)
        f = lambda k: sum(1 for c in allc if k(c))
        print(f"candidates {n}: closed {f(lambda c: c['closed']) / n:.0%} | correct {f(lambda c: c['correct']) / n:.0%} | "
              f"derived {f(lambda c: c['derived']) / n:.0%} | decisive {f(lambda c: c['decisive']) / n:.0%} | "
              f"all four (qualified) {f(lambda c: c['ok']) / n:.0%} | echoed the instruction {f(lambda c: c['echoed']) / n:.0%} "
              f"| invented a [note] or copied the problem {f(lambda c: c.get('fake_note', False)) / n:.0%}")
        lucky = [c for c in allc if c["closed"] and c["correct"] and not c["derived"]]
        print(f"  correct but NOT derived (lucky / answer redone in the answer): {len(lucky)} "
              f"({len(lucky) / max(f(lambda c: c['correct']), 1):.0%} of correct ones)")
    print(f"\n{'state':<11}{'cut':>7}{'cuts':>6}{'acc(all samples)':>18}{'med think':>11}"
          f"{'stop-now ok':>13}{'has target':>12}{'target med':>12}{'dither/1k all':>15}{'chosen':>8}")
    g = defaultdict(list)
    for r in recs:
        g[(r["state"], bucket(r["frac"]))].append(r)
    for (s, fr), rs in sorted(g.items(), key=lambda kv: (kv[0][0], ORDER.get(kv[0][1], 9))):
        cs = [c for r in rs for c in r["cands"]]
        tg = [r["chosen_tok"] for r in rs if r["chosen"] is not None]
        ch = [r["cands"][r["chosen"]] for r in rs if r["chosen"] not in (None, -1)]
        rate = lambda xs: np.median([c["dither"] * 1000 / max(c["think_tok"], 1) for c in xs]) if xs else float("nan")
        print(f"{s:<11}{fr:>7}{len(rs):>6}{np.mean([c['correct'] for c in cs]):>18.2f}"
              f"{np.median([c['think_tok'] for c in cs]):>11.0f}{np.mean([r['hard']['ok'] for r in rs]):>13.0%}"
              f"{len(tg) / len(rs):>12.0%}{(np.median(tg) if tg else float('nan')):>12.0f}"
              f"{rate(cs):>15.1f}{rate(ch):>8.1f}")
    print("acc(all samples) = accuracy of the instruction itself (compare: notice at 25%, mid-way = 0.90)")
    print(f"\n{'state':<11}{'cut':>7}{'stop-now':>10}{'sample':>8}{'none':>6}{'sample think med':>18}"
          f"{'target total med':>18}{'uncut left med':>16}{'saved':>7}")
    for (s, fr), rs in sorted(g.items(), key=lambda kv: (kv[0][0], ORDER.get(kv[0][1], 9))):
        n = len(rs)
        sn = sum(r["chosen"] == -1 for r in rs)
        sm = [r for r in rs if r["chosen"] not in (None, -1)]
        tot = [r["chosen_total"] for r in rs if r["chosen"] is not None]
        left = [r["L"] - r["u"] + r["uncut_answer_tok"] for r in rs if r["chosen"] is not None]
        print(f"{s:<11}{fr:>7}{sn / n:>10.0%}{len(sm) / n:>8.0%}{(n - sn - len(sm)) / n:>6.0%}"
              f"{(np.median([r['chosen_tok'] for r in sm]) if sm else float('nan')):>18.0f}"
              f"{(np.median(tot) if tot else float('nan')):>18.0f}{(np.median(left) if left else float('nan')):>16.0f}"
              f"{(1 - sum(tot) / sum(left) if tot else float('nan')):>7.0%}")
    print("target total = thinking + answer tokens after the cut; uncut left = what the uncut run still produced")
    if path_md:
        write_md(recs, path_md, n_show)
        print(f"examples -> {path_md}")


def _clip(s, head, tail=0):
    s = s or ""
    if len(s) <= head + tail + 40:
        return s
    return s[:head] + f"\n[... {len(s) - head - tail} chars skipped ...]\n" + (s[-tail:] if tail else "")


def write_md(recs, path, n_show):
    md = ["# Decisive sampling: smoke examples\n",
          "Per cut point: the 8 candidates, the chosen target, the shortest correct one that was REJECTED "
          "(and why), and what the uncut run did next. Check: is the chosen target really efficient and "
          "decisive, and complete (no skipped step it needed)?\n"]
    order = sorted(recs, key=lambda r: (r["state"] != "midway", r["frac"]))
    for r in order[: n_show or len(order)]:
        md.append(f"\n## {r['state']} | cut at {r['frac']:.0%} ({r['u']} of {r['L']}) | id {r['id']} s{r['sample']} "
                  f"| gold `{r['gold']}`\n")
        md.append(f"stop-now: {r['hard']['hits']}/4 correct, answer {r['hard']['answer_tok']:.0f} tok "
                  f"-> {'OK' if r['hard']['ok'] else 'no'}\n")
        md.append("| # | think tok | closed | correct | derived | dither | qualified |\n|---|---|---|---|---|---|---|")
        for c in sorted(r["cands"], key=lambda c: c["think_tok"]):
            md.append(f"| {c['j']}{' ★' if r['chosen'] == c['j'] else ''} | {c['think_tok']} | {c['closed']} | "
                      f"{c['correct']} | {c['derived_hits'] if c['derived_hits'] is not None else '-'}/4"
                      f"{' (redo)' if c['check_tok'] and not c['derived'] and (c['derived_hits'] or 0) >= 3 else ''} | "
                      f"{c['dither']} | {'✓' if c['ok'] else ''} |")
        if r["chosen"] == -1:
            md.append("\n**Chosen: stop right now.**")
        elif r["chosen"] is not None:
            c = r["cands"][r["chosen"]]
            md.append(f"\n**Chosen target ({c['think_tok']} tok)**\n\n```\n{_clip(c['think'], 2500, 500)}\n```")
            md.append(f"\nAnswer ({c['answer_tok']} tok):\n```\n{_clip(c['answer'], 600)}\n```")
        else:
            md.append("\n**No target** (nothing qualified).")
        rej = [c for c in r["cands"] if c["correct"] and not c["ok"]]
        if rej:
            c = min(rej, key=lambda c: c["think_tok"])
            why = [w for w, bad in [("not closed", not c["closed"]), ("not derived", not c["derived"]),
                                    ("dithers", not c["decisive"])] if bad]
            md.append(f"\n**Shortest correct but rejected ({c['think_tok']} tok; {', '.join(why)})**\n\n"
                      f"```\n{_clip(c['think'], 1500, 300)}\n```")
        wrong = [c for c in r["cands"] if not c["correct"]]
        if wrong:
            c = min(wrong, key=lambda c: c["think_tok"])
            md.append(f"\n**Shortest WRONG one ({c['think_tok']} tok, answered `{final_box(c['answer'])}`)**\n\n"
                      f"```\n{_clip(c['think'], 1200, 300)}\n```")
        md.append(f"\n**Uncut run after the cut (first 1,500 chars)**\n\n```\n{r.get('_rest', '')}\n```\n")
    open(path, "w").write("\n".join(md))


async def main_async(args):
    rows = read_traces(args.traces)
    items = pick(rows, args)
    if not items:
        raise SystemExit("no traces match (check --split / --min-think / --samples-per-problem)")
    name = os.path.basename(args.traces).replace(".jsonl", "")
    out = args.out or os.path.join("runs", f"{name}.decisive{'_smoke' if args.smoke else ''}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fixed = [0.25, 0.5] if args.smoke and args.fracs is None else args.fracs

    def fracs_for(r):
        """Fixed --fracs, or (default) --cuts random positions per trace, one in each equal slice of
        [--lo, --hi]: every trace gets an early, a middle and a late cut, at positions that differ from
        trace to trace (deployment cuts at any point, not on a grid). Same trace -> same positions,
        so a resumed run picks the same cut points."""
        if fixed:
            return fixed
        rng = random.Random(zlib.crc32(f"{r['id']}|{r.get('sample', 0)}|{args.seed}".encode()))
        w = (args.hi - args.lo) / args.cuts
        return [args.lo + w * (i + rng.random()) for i in range(args.cuts)]

    srv = Server(args.base_url, args.model, args.concurrency)
    think_open = await find_think_open(srv, messages_for(items[0], args.system), args.template_kwargs)
    inst_ids = await srv.tok(INSTRUCTION, cache=True)
    close_ids = await srv.tok("\n" + CLOSE + "\n\n", cache=True)

    done = {}
    if os.path.exists(out):
        for line in open(out):
            try:
                rec = json.loads(line)
                done[rec["key"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    total = sum(len(cut_points(r["L"], fracs_for(r), [])) for r in items)
    mine = {(r["id"], r.get("sample", 0)) for r in items}
    n_done = sum(1 for rec in done.values() if (rec["id"], rec.get("sample", 0)) in mine)   # older runs also
    print(f"{len(items)} traces, {total} cut points x {args.k} samples ({n_done} of them already done"  # hold unsolved
          + (f"; {len(done) - n_done} older records in the file are not in this run" if len(done) > n_done else "")
          + f") -> {out}")
    fout, lock, t0, st = open(out, "a"), asyncio.Lock(), time.time(), {"n": 0}
    recs = []

    async def do_cut(r, base_ids, think_ids, u):
        key = f"{r['id']}|{r.get('sample', 0)}|{u}"
        if key in done:
            rec = done[key]
        else:
            rec = await one_cut(srv, r, base_ids, think_ids, u, args, inst_ids, close_ids)
            async with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                st["n"] += 1
                if st["n"] % 20 == 0:
                    el = time.time() - t0
                    left = max(total - n_done - st["n"], 0) * el / st["n"]
                    print(f"  {st['n'] + n_done}/{total} cut points | about {left / 60:.0f} min left", flush=True)
        if args.smoke:
            rest_text = r["thinking"][len(await srv.detok(think_ids[:u])):]
            rec = dict(rec, _rest=rest_text[:1500])
        recs.append(rec)

    async def do_item(r):
        base_ids, think_ids = await prepare(srv, r, args, think_open)
        await asyncio.gather(*[do_cut(r, base_ids, think_ids, u) for u in cut_points(len(think_ids), fracs_for(r), [])])

    gate = asyncio.Semaphore(max(4, args.concurrency // args.k))

    async def gated(r):
        async with gate:
            await do_item(r)

    await asyncio.gather(*[gated(r) for r in items])
    fout.close()
    await srv.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    report(recs, args.md if args.smoke else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--samples-per-problem", type=int, default=4)
    ap.add_argument("--min-think", type=int, default=512)
    ap.add_argument("--max-traces", type=int, default=0)
    ap.add_argument("--fracs", type=lambda s: [float(x) for x in s.split(",") if x], default=None,
                    help="fixed cut positions for every trace (smoke default: 0.25,0.5); "
                         "default: --cuts random positions per trace")
    ap.add_argument("--cuts", type=int, default=3, help="random cut points per trace (one per slice of [lo, hi])")
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=8, help="samples per cut point")
    ap.add_argument("--cap", type=int, default=2048, help="safety cap on thinking after the cut")
    ap.add_argument("--checks", type=int, default=4, help="answers per derived / stop-now check")
    ap.add_argument("--need", type=int, default=3, help="correct answers needed out of --checks (derived check)")
    ap.add_argument("--need-stop", type=int, default=4, help="correct answers needed for 'stop right now'")
    ap.add_argument("--answer-max", type=int, default=2048)
    ap.add_argument("--check-max", type=int, default=1024)
    ap.add_argument("--min-redo", type=int, default=400,
                    help="a check answer longer than max(2x the uncut answer, this) counts as redoing the work")
    ap.add_argument("--dither-free", type=int, default=2, help="this many hesitation phrases are always allowed")
    ap.add_argument("--max-dither", type=float, default=8.0, help="else at most this many per 1,000 tokens")
    ap.add_argument("--system", default=None)
    ap.add_argument("--template-kwargs", type=json.loads, default={"enable_thinking": True})
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="N solved traces, writes decisive_smoke.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true",
                    help="re-select and report an existing --out file (no generation); writes --md")
    ap.add_argument("--md", default="decisive_smoke.md")
    args = ap.parse_args()
    if args.report:
        if not args.out:
            raise SystemExit("--report needs --out <the decisive jsonl>; the positional argument is the traces file")
        recs = [select(json.loads(l), args) for l in open(args.out) if l.strip()]
        tr = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
        for r in recs:
            th = tr.get((r["id"], r["sample"]), {}).get("thinking", "")
            r["_rest"] = th[int(len(th) * r["frac"]):][:1500]
        report(recs, args.md)
        return
    if not args.model:
        raise SystemExit("--model is required")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
