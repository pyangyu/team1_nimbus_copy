#!/usr/bin/env python3
"""Re-grade saved results with the fixed grader (analyze_boxed.same), in place (a .bak copy is kept).

The old grader compared strings, so \\frac12 vs \\frac{1}{2}, \\frac13 vs \\dfrac{1}{3}, or 119/407 vs
28917/98901 counted as WRONG. Nothing is re-generated: only the stored answers are graded again.

  python regrade.py traces/Qwen__Qwen3.8-27B.jsonl                 # trace files: field "correct"
  python regrade.py runs/Qwen__Qwen3.8-27B.bench.jsonl --traces traces/Qwen__Qwen3.8-27B.jsonl
                                                                    # bench runs: "correct" and "uncut_correct"
"""
import argparse
import json
import os
import shutil
from collections import Counter

from analyze_boxed import boxed_all
from cutlib import graded


def last(ans):
    b = boxed_all(ans or "")
    return b[-1][1] if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--traces", default=None, help="trace file, needed for bench runs (uncut answers)")
    ap.add_argument("--dry", action="store_true", help="only report, do not write")
    args = ap.parse_args()
    uncut = {}
    if args.traces:
        for l in open(args.traces):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            uncut[(r["id"], r.get("sample", 0))] = r
    for path in args.files:
        rows, flips, ex = [], Counter(), []
        for l in open(path):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                continue
        for r in rows:
            fields = []
            if "cond" in r:                                   # a bench run
                fields.append(("correct", r.get("answer")))
                t = uncut.get((r["id"], r.get("sample", 0)))
                if t is not None:
                    fields.append(("uncut_correct", t.get("answer")))
                elif args.traces is None:
                    raise SystemExit("bench runs need --traces (for the uncut answers)")
            else:                                             # a trace
                fields.append(("correct", r.get("answer")))
            for f, ans in fields:
                new = graded(ans, r["gold"])
                old = bool(r.get(f))
                if new != old:
                    flips[(f, old, new)] += 1
                    if len(ex) < 12:
                        ex.append((f, r["gold"], last(ans), old, new))
                r[f] = new
        n = len(rows)
        print(f"\n{path}: {n} rows")
        for (f, o, nw), c in sorted(flips.items()):
            print(f"  {f}: {'right' if o else 'wrong'} -> {'right' if nw else 'wrong'}: {c} ({c / n:.1%})")
        if not flips:
            print("  nothing changed")
        for f, g, p, o, nw in ex:
            print(f"    {f}: gold {g!r}  answer {p!r}  {o} -> {nw}")
        if flips and not args.dry:
            bak = path + ".bak"
            if not os.path.exists(bak):
                shutil.copy(path, bak)
            with open(path, "w") as out:
                for r in rows:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  written (original kept as {bak})")


if __name__ == "__main__":
    main()
