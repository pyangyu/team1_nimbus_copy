#!/usr/bin/env python3
"""Sanity check of the baseline benchmark: is "saved 89%" real, and how does the model manage it?

  python sample_bench.py runs/Qwen__Qwen3.8-27B.bench.jsonl traces/Qwen__Qwen3.8-27B.jsonl
  -> prints the checks, writes bench_samples.md (examples to read)

Checks
  A. where the saved tokens come from: capped traces (hit the token limit) vs normal ones
  B. in the ORIGINAL uncut traces: how early the final answer already appears in the thinking
     (if it is usually there long before the end, a cut that says "wrap up" loses little)
  C. examples, side by side: end of the kept prefix | what the model thought after the notice | answer
"""
import argparse
import json
import random
import re
import sys  # noqa: F401
from collections import defaultdict

import numpy as np

from cutlib import CAP, final_box, read_traces, same          # noqa: E402
from analyze_boxed import boxed_all, norm                       # noqa: E402


def state(r):
    if not r["uncut_correct"]:
        return "unsolved"
    return "concluded" if r["prefix_has_final_box"] else "midway"


def pct(x):
    return f"{x:.0%}"


def saved(g):
    rest = sum(r["L"] - r["u"] for r in g)
    return 1 - sum(r["think_after_tok"] for r in g) / max(rest, 1)


def clip(s, head, tail=0):
    s = s or ""
    if len(s) <= head + tail + 40:
        return s
    return s[:head] + f"\n\n[... {len(s) - head - tail} chars skipped ...]\n\n" + (s[-tail:] if tail else "")


def first_answer_pos(thinking, final):
    """Fraction of the thinking (by chars) at which the final answer first shows up in a \\boxed{},
    or, failing that, as plain text (only for answers of 3+ chars, to avoid matching '2' anywhere)."""
    found = []
    for end, v in boxed_all(thinking):
        if same(v, final):
            found.append((end / max(len(thinking), 1), "boxed"))
            break
    n = norm(final)
    flat = thinking.replace(" ", "")
    if len(n) >= 3:
        i = flat.find(n)
        if i >= 0:
            found.append((i / max(len(flat), 1), "text"))
    else:                                    # short answers: only "= 5" / "is 5" style statements
        m = re.search(r"(?:=|\bis|\bare|answer:?)\s*\$?" + re.escape(n) + r"(?![\d./])", thinking)
        if m:
            found.append((m.start() / max(len(thinking), 1), "text"))
    return min(found) if found else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("traces")
    ap.add_argument("--n", type=int, default=3, help="examples per group")
    ap.add_argument("--out", default="bench_samples.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = [json.loads(l) for l in open(args.runs) if l.strip()]
    tr = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
    for r in runs:
        r["state"] = state(r)
        r["capped"] = r["L"] >= CAP
    notice = [r for r in runs if r["kind"] != "hard"]
    hard = [r for r in runs if r["kind"] == "hard"]

    # ---------------- A
    print("A. Where the saved thinking comes from (notice runs)")
    print(f"{'group':<34}{'runs':>6}{'mean L':>9}{'mean left':>11}{'after med':>11}{'saved':>8}")
    groups = [("all", notice),
              ("traces that hit the cap (32K)", [r for r in notice if r["capped"]]),
              ("traces that ended by themselves", [r for r in notice if not r["capped"]])]
    for s in ("concluded", "midway", "unsolved"):
        groups.append((f"  {s}, not capped", [r for r in notice if r["state"] == s and not r["capped"]]))
    for f in (0.25, 0.5, 0.75):
        groups.append((f"  not capped, cut at {f}", [r for r in notice if not r["capped"] and abs(r["frac"] - f) < 0.02]))
    for name, g in groups:
        if not g:
            continue
        print(f"{name:<34}{len(g):>6}{np.mean([r['L'] for r in g]):>9.0f}"
              f"{np.mean([r['L'] - r['u'] for r in g]):>11.0f}{np.median([r['think_after_tok'] for r in g]):>11.0f}"
              f"{pct(saved(g)):>8}")
    capped_share = sum(r["L"] - r["u"] for r in notice if r["capped"]) / max(sum(r["L"] - r["u"] for r in notice), 1)
    print(f"capped traces hold {pct(capped_share)} of all the thinking that was still ahead.")
    acc = lambda g: np.mean([r["correct"] for r in g]) if g else float("nan")
    nc = [r for r in notice if not r["capped"]]
    print(f"not capped only: uncut acc {np.mean([r['uncut_correct'] for r in nc]):.3f}, "
          f"notice acc {acc(nc):.3f}, saved {pct(saved(nc))}")

    # ---------------- B
    print("\nB. In the ORIGINAL correct traces: where the final answer first appears in the thinking")
    pos, kinds = [], defaultdict(int)
    seen = set()
    for r in runs:
        key = (r["id"], r["sample"])
        if key in seen or not r["uncut_correct"]:
            continue
        seen.add(key)
        t = tr.get(key)
        if not t:
            continue
        fb = final_box(t.get("answer"))
        if fb is None:
            continue
        p, k = first_answer_pos(t["thinking"], fb)
        kinds[k] += 1
        if p is not None:
            pos.append(p)
    n = len(seen)
    pos = np.array(pos)
    print(f"{n} correct traces; answer found in thinking: boxed {kinds['boxed']}, as text {kinds['text']}, "
          f"not found {kinds[None]}")
    if len(pos):
        print(f"first appearance, as a share of the thinking: median {np.median(pos):.0%}")
        for f in (0.25, 0.5, 0.75):
            print(f"  already there before {f:.0%} of the thinking: {np.mean(pos <= f) * len(pos) / n:.0%} of correct traces")
    print("  (earliest of: boxed, or the answer as text; short answers only as '= 5' / 'is 5'. Rough, but no longer boxed-only)")

    # ---------------- C
    rng = random.Random(args.seed)
    picks = [
        ("notice | mid-way, solvable | cut at 0.25 | still correct",
         [r for r in notice if r["state"] == "midway" and abs(r["frac"] - 0.25) < 0.02 and r["correct"]]),
        ("notice | mid-way, solvable | cut at 0.25 | LOST (became wrong)",
         [r for r in notice if r["state"] == "midway" and abs(r["frac"] - 0.25) < 0.02 and not r["correct"]]),
        ("notice | mid-way, solvable | cut at 0.50 | still correct",
         [r for r in notice if r["state"] == "midway" and abs(r["frac"] - 0.5) < 0.02 and r["correct"]]),
        ("notice | has conclusion | still correct",
         [r for r in notice if r["state"] == "concluded" and r["correct"]]),
        ("notice | unsolved when uncut | WON (became correct)",
         [r for r in notice if r["state"] == "unsolved" and r["correct"]]),
        ("hard | mid-way, solvable | cut at 0.25 | still correct",
         [r for r in hard if r["state"] == "midway" and abs(r["frac"] - 0.25) < 0.02 and r["correct"]]),
    ]
    md = ["# Baseline bench: examples\n",
          "Each example: the end of the kept thinking prefix, what the model thought after the cut, "
          "and its answer. Compare `thinking left` (what the uncut run still used) with `after`.\n"]
    for title, g in picks:
        md.append(f"\n## {title}  ({len(g)} runs in this group)\n")
        for r in rng.sample(g, min(args.n, len(g))):
            t = tr.get((r["id"], r["sample"]), {})
            th = t.get("thinking", "")
            prefix = th[: r["prefix_chars"]]
            md.append(f"\n### id {r['id']} sample {r['sample']} — gold `{r['gold']}`\n")
            md.append(f"- thinking: {r['L']} tokens total{' (HIT CAP)' if r['capped'] else ''}; cut at {r['u']} "
                      f"({r['frac']:.0%}); thinking left in the uncut run: **{r['L'] - r['u']}**")
            md.append(f"- after the cut: **{r['think_after_tok']}** thinking tokens"
                      + (f", closed by itself: {r['self_term']}" if r["kind"] != "hard" else "")
                      + (", copied the notice back first" if r.get("echoed_notice") else ""))
            md.append(f"- answer tokens: uncut {r['uncut_answer_tok']} → cut {r['answer_tok']}; "
                      f"uncut correct {r['uncut_correct']} → cut correct {r['correct']}; "
                      f"uncut final `{final_box(t.get('answer'))}`, cut final `{final_box(r['answer'])}`")
            md.append(f"\n**Problem**\n\n```\n{clip(t.get('prompt', ''), 600)}\n```")
            md.append(f"\n**End of the kept prefix (last 1,200 chars)**\n\n```\n{prefix[-1200:]}\n```")
            if r["kind"] != "hard":
                md.append(f"\n**Thinking after the notice**\n\n```\n{clip(r.get('think_after', ''), 1500, 600)}\n```")
            md.append(f"\n**Answer**\n\n```\n{clip(r['answer'], 1200, 300)}\n```")
            md.append(f"\n**What the uncut run did with the rest of its thinking (next 1,500 chars after the cut)**"
                      f"\n\n```\n{th[r['prefix_chars']: r['prefix_chars'] + 1500]}\n```\n")
    open(args.out, "w").write("\n".join(md))
    print(f"\nC. examples -> {args.out}")


if __name__ == "__main__":
    main()
