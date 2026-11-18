#!/usr/bin/env python3
"""Build SFT examples for cut-aware training from the runs of cut_bench.py (no model is run).

There is no budget: for each cut point the target is the SHORTEST continuation that
  - ends with an acceptable answer (see --accept),
  - closed its thinking by itself (or is the immediate close),
  - and comes from a condition that was acceptable in at least --min-agree of its samples
    (so a lucky guess does not become a training target).

--accept correct (default): the answer must be correct.
--accept either: correct, OR the same final answer the uncut run gave (needs no gold label,
and lets traces the uncut model got wrong still teach "stop early"; those count half).

  python make_sft.py runs/Qwen__Qwen3.8-27B.sft.jsonl traces/Qwen__Qwen3.8-27B.jsonl
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from cutlib import SUFFIX, final_box, notice_text, read_traces, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("traces")
    ap.add_argument("--min-agree", type=int, default=0, help="0 = 2 when there are several samples, else 1")
    ap.add_argument("--accept", default="correct", choices=["correct", "agree", "either"])
    ap.add_argument("--max-think", type=int, default=2048,
                    help="a target may use at most this many thinking tokens after the notice (a safety cap, "
                         "not a budget the model is told)")
    ap.add_argument("--cap-concluded", type=float, default=0.4,
                    help="keep at most this share of 'has conclusion' examples (0 = keep all), so the adapter "
                         "does not learn to always stop at once")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = []
    for line in open(args.runs):
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    traces = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
    kmax = max(r["k"] for r in runs) + 1
    need = args.min_agree or (2 if kmax >= 2 else 1)
    levels = [args.max_think]            # no budget levels: one target per cut point, the shortest correct one
    by_cut = defaultdict(list)
    for r in runs:
        by_cut[(r["id"], r["sample"], r["u"])].append(r)

    out_path = args.out or os.path.join("sft", os.path.basename(args.runs))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stats = defaultdict(lambda: defaultdict(list))        # state -> level -> [(has target, cost, uncut rest)]
    examples = []
    if True:
        for (pid, sample, u), g in by_cut.items():
            t = traces.get((pid, sample))
            if t is None:
                continue
            m = g[0]
            state = ("has conclusion" if m["prefix_has_final_box"] else
                     "mid-way, solvable" if m["uncut_correct"] else "unsolved when uncut")
            uncut_final = final_box(t.get("answer"))

            def ok(r):
                same_as_uncut = uncut_final is not None and (final_box(r["answer"]) is not None
                                                             and same(final_box(r["answer"]), uncut_final))
                return {"correct": r["correct"], "agree": same_as_uncut,
                        "either": r["correct"] or same_as_uncut}[args.accept]

            agree, tried = defaultdict(int), defaultdict(int)
            for r in g:
                agree[r["cond"]] += bool(ok(r))
                tried[r["cond"]] += 1
            usable = [r for r in g if ok(r) and agree[r["cond"]] >= min(need, tried[r["cond"]])
                      and (r["grace"] == 0 or r["self_term"])]
            rest = m["L"] - u + m["uncut_answer_tok"]
            for B in levels:
                cand = [r for r in usable if r["think_after_tok"] <= B and (r["kind"] == "prune" or r["grace"] <= B)]
                best = min(cand, key=lambda r: r["think_after_tok"] + r["answer_tok"]) if cand else None
                stats[state][B].append((best is not None,
                                        best["think_after_tok"] + best["answer_tok"] if best else None, rest))
                if best is None:
                    continue
                examples.append({
                    "id": pid, "sample": sample, "source": m.get("source"), "state": state,
                    "cut_tokens": u, "uncut_thinking_tokens": m["L"],
                    "user": t["prompt"] + (SUFFIX if t.get("suffix") is None else t["suffix"]),
                    "thinking_prefix": t["thinking"][: m["prefix_chars"]],
                    "notice": notice_text("notice", 0),
                    "target_thinking": best["think_after"],          # what the model should think after the notice
                    "target_answer": best["answer"],
                    "target_tokens": best["think_after_tok"] + best["answer_tok"],
                    "target_correct": bool(best["correct"]),
                    "weight": 1.0 if best["correct"] else 0.5,      # same-as-uncut-but-wrong targets count half
                    "uncut_rest_tokens": rest, "from_cond": best["cond"],
                })

    easy = [e for e in examples if e["state"] == "has conclusion"]
    other = [e for e in examples if e["state"] != "has conclusion"]
    cap = args.cap_concluded
    if 0 < cap < 1 and other and len(easy) > cap / (1 - cap) * len(other):
        keep = int(cap / (1 - cap) * len(other))
        np.random.default_rng(0).shuffle(easy)
        print(f"'has conclusion' examples: kept {keep} of {len(easy)} (cap {cap:.0%} of the set)")
        easy = easy[:keep]
    examples = sorted(easy + other, key=lambda e: (e["id"], e["sample"], e["cut_tokens"]))
    n_out = len(examples)
    with open(out_path, "w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    by_src = defaultdict(int)
    for e in examples:
        by_src[e["from_cond"]] += 1
    print("targets by source: " + ", ".join(f"{k} {v}" for k, v in sorted(by_src.items())))

    print(f"{len(by_cut)} cut points, {kmax} sample(s) per condition, a condition must be correct in >= {need} of them")
    print(f"{n_out} SFT examples -> {out_path}\n")
    print("Share of cut points that have a usable target, by state")
    print(f"{'state':<24}{'cut pts':>8}{'target':>10}")
    for state in ("has conclusion", "mid-way, solvable", "unsolved when uncut"):
        if state in stats:
            n = len(stats[state][levels[0]])
            print(f"{state:<24}{n:>8}" + "".join(f"{np.mean([x[0] for x in stats[state][B]]):>10.0%}" for B in levels))
    print("\nMedian tokens after the cut: target vs what the uncut run used (thinking + answer)")
    print(f"{'state':<24}{'target':>10}{'uncut':>10}")
    for state in stats:
        cells = []
        for B in levels:
            c = [x[1] for x in stats[state][B] if x[0]]
            cells.append(f"{np.median(c):>10.0f}" if c else f"{'-':>10}")
        print(f"{state:<24}" + "".join(cells) + f"{np.median([x[2] for x in stats[state][levels[0]]]):>10.0f}")
    print("\nA low share in a row means no short correct wrap-up was found from that state: add cut")
    print("points or problems, or check the pruning pass rate.")


if __name__ == "__main__":
    main()
