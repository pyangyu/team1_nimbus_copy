#!/usr/bin/env python3
"""Tables for the cut benchmark (reads the JSONL written by cut_bench.py; no model is run).

  python cut_report.py runs/Qwen__Qwen3.8-27B.bench.jsonl
"""
import json
import sys

import numpy as np

from cutlib import dither_rate


def load(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if rows and "cond" not in rows[0]:
        raise SystemExit(f"{path} is not a cut_bench result (it looks like a trace file).\n"
                         f"Run cut_bench.py on the traces first, then pass its output, e.g.\n"
                         f"  python cut_report.py runs/<traces name>.bench.jsonl")
    for r in rows:
        r["state"] = ("has conclusion" if r["prefix_has_final_box"] else
                      "mid-way, solvable" if r["uncut_correct"] else "unsolved when uncut")
        r["bucket"] = f"{r['frac']:.2f}" if abs(r["frac"] - round(r["frac"] * 4) / 4) < 0.02 else f"{r['u']} tok"
    return rows


def line(label, g):
    n = len(g)
    unc = np.mean([r["uncut_correct"] for r in g])
    cut = np.mean([r["correct"] for r in g])
    st = [r["self_term"] for r in g if r["self_term"] is not None]
    after = np.median([r["think_after_tok"] for r in g])
    rest = sum(r["L"] - r["u"] for r in g)                               # thinking the uncut run still had to do
    saved = 1 - sum(r["think_after_tok"] for r in g) / max(rest, 1)
    a_cut, a_unc = np.mean([r["answer_tok"] for r in g]), np.mean([r["uncut_answer_tok"] for r in g])
    net = 1 - sum(r["think_after_tok"] + r["answer_tok"] for r in g) / max(
        sum(r["L"] - r["u"] + r["uncut_answer_tok"] for r in g), 1)
    lost = np.mean([r["uncut_correct"] and not r["correct"] for r in g])
    won = np.mean([r["correct"] and not r["uncut_correct"] for r in g])
    dr = [dither_rate(r.get("think_after"), r["think_after_tok"]) for r in g
          if r["think_after_tok"] >= 20 and r.get("think_after")]
    return (f"{label:<34}{n:>6}{unc:>8.3f}{cut:>8.3f}{cut - unc:>+8.3f}{lost:>7.1%}{won:>7.1%}"
            f"{(format(np.mean(st), '.0%') if st else '-'):>9}{after:>8.0f}{saved:>8.0%}{a_unc:>8.0f}{a_cut:>8.0f}{net:>8.0%}"
            f"{(format(np.mean(dr), '.1f') if dr else '-'):>8}")


HEAD = (f"{'':<34}{'runs':>6}{'uncut':>8}{'cut':>8}{'change':>8}{'lost':>7}{'won':>7}"
        f"{'self-end':>9}{'after':>8}{'saved':>8}{'ans unc':>8}{'ans cut':>8}{'net':>8}{'dither':>8}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = load(sys.argv[1])
    conds = sorted({r["cond"] for r in rows}, key=lambda c: (c != "hard", c))
    n_cut = len({(r["id"], r["sample"], r["u"]) for r in rows})
    print(f"{len(rows)} runs, {n_cut} cut points, {len({(r['id'], r['sample']) for r in rows})} traces\n")

    print("1) By condition (all cut points)")
    print(HEAD)
    for c in conds:
        print(line(c, [r for r in rows if r["cond"] == c]))

    for c in conds:
        g = [r for r in rows if r["cond"] == c and "echoed_notice" in r]
        if g and any(r["echoed_notice"] for r in g):
            print(f"{c}: the model copied the notice back before wrapping up in "
                  f"{sum(r['echoed_notice'] for r in g) / len(g):.0%} of runs (those tokens count as thinking after the cut)")

    print("\n2) By condition and where the cut happened")
    print(HEAD)
    for c in conds:
        for b in sorted({r["bucket"] for r in rows}):
            g = [r for r in rows if r["cond"] == c and r["bucket"] == b]
            if g:
                print(line(f"{c} | cut at {b}", g))

    print("\n3) By condition and the state of the request when it was cut")
    print(HEAD)
    for c in conds:
        for s in ("has conclusion", "mid-way, solvable", "unsolved when uncut"):
            g = [r for r in rows if r["cond"] == c and r["state"] == s]
            if g:
                print(line(f"{c} | {s}", g))


    print("""
uncut / cut   accuracy of the original full run / of the cut run (same grader)
lost / won    share of runs that went correct -> wrong / wrong -> correct
self-end      the model closed its thinking by itself before the grace ran out (notice conditions)
after         median thinking tokens used after the cut
saved         share of the thinking the uncut run still had ahead that the cut removed
ans unc/cut   mean answer length, uncut and cut (does the model redo the thinking in the answer?)
net           share of ALL tokens after the cut point that was removed (thinking + answer)
dither        hesitation phrases (wait, let me double-check, alternative method, ...) per 1,000 thinking
              tokens after the cut (runs with at least 20 such tokens); lower = more decisive
states        has conclusion = the prefix already contains the final answer in a \\boxed{}
              mid-way, solvable = no boxed answer yet, and the uncut run ended correct
              unsolved when uncut = the uncut run was wrong or hit the token cap""")


if __name__ == "__main__":
    main()
