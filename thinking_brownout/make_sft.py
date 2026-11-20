#!/usr/bin/env python3
"""SFT examples for the cut adapter, from the output of decisive_sample.py (no model is run).

One example per cut point that has a target. The target is the qualified candidate with the lowest
cost after the cut: thinking tokens + answer tokens beyond the uncut answer's length (so moving work
into the answer is charged, but stripping the answer down to a bare box earns nothing):
  stop now      close the thinking at the cut; correct in 4 of 4 tries, answer not redoing the work
  a sample      one of the K continuations written under the data-generation instruction that is
                closed, correct, derived (correct with its answer hidden) and decisive
In the example, the instruction is replaced by the bare [cut] marker: the adapter learns to act like the
instruction's best continuations without ever seeing it. Loss is on the target only (cut_alora.py).

Balance
  --cap-concluded   at most this share of examples from cuts where the answer was already boxed
  --max-stop-share  at most this share of "stop now" targets, so the adapter does not learn to always stop
Validation is split by PROBLEM inside cut_alora.py train, so all cut points of a problem stay on one side.

  python make_sft.py runs/Qwen__Qwen3.8-27B.decisive.jsonl runs/Qwen__Qwen3.8-27B.more.decisive.jsonl \\
      --traces traces/Qwen__Qwen3.8-27B.jsonl traces/Qwen__Qwen3.8-27B.more.jsonl
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from cutlib import SUFFIX, read_traces
from decisive_sample import bucket, select

STATES = {"concluded": "has conclusion", "midway": "mid-way, solvable", "unsolved": "unsolved when uncut"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="output files of decisive_sample.py")
    ap.add_argument("--traces", nargs="+", required=True, help="the trace files those runs were made from")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B", help="tokenizer, only for records without prefix_chars")
    ap.add_argument("--need-stop", type=int, default=4, help="'stop now' must be correct in this many of 4 tries")
    ap.add_argument("--min-redo", type=int, default=400,
                    help="an answer longer than max(2x the uncut answer, this) counts as redoing the work")
    ap.add_argument("--cap-concluded", type=float, default=0.4)
    ap.add_argument("--max-stop-share", type=float, default=0.5,
                    help="cap on the share of 'stop now' targets (0.5 = no more than continuations)")
    ap.add_argument("--stop-below", type=int, default=32,
                    help="a continuation with fewer thinking tokens than this is a stop in all but name: it counts "
                         "as 'stop now' for the balance, so real continuations get their share")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = {}
    for path in args.traces:
        for r in read_traces(path):
            traces[(r["id"], r.get("sample", 0))] = r
    recs, seen, n_unsolved = [], set(), 0
    for path in args.runs:
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            if r["state"] == "unsolved":                    # older runs sampled these too; never a target
                n_unsolved += 1
                continue
            recs.append(select(r, args))

    tok = None
    examples, no_trace = [], 0
    stats = defaultdict(lambda: [0, 0])                     # (state, bucket) -> [cut points, with target]
    for r in recs:
        b = bucket(r["frac"])
        stats[(r["state"], b)][0] += 1
        if r["chosen"] is None:
            continue
        t = traces.get((r["id"], r["sample"]))
        if t is None:
            no_trace += 1
            continue
        chars = r.get("prefix_chars")
        if chars is None:                                   # older records: rebuild the prefix from tokens
            if tok is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(args.model)
            ids = tok.encode(t["thinking"], add_special_tokens=False)[: r["u"]]
            pre = tok.decode(ids)
            chars = len(pre) if t["thinking"].startswith(pre) else int(len(t["thinking"]) * r["u"] / max(len(ids), 1))
        if r["chosen"] == -1:
            think, answer, src = "", r["hard"]["answer"], "stop-now"
        else:
            c = r["cands"][r["chosen"]]
            think, answer, src = c["think"], c["answer"], "sample"
        stats[(r["state"], b)][1] += 1
        examples.append({
            "id": r["id"], "sample": r["sample"], "source": r.get("source"), "state": STATES[r["state"]],
            "cut": b, "frac": r["frac"], "cut_tokens": r["u"], "uncut_thinking_tokens": r["L"],
            "user": t["prompt"] + (SUFFIX if t.get("suffix") is None else t["suffix"]),
            "thinking_prefix": t["thinking"][:chars],
            "target_thinking": think, "target_answer": answer,
            "target_think_tokens": r["chosen_tok"], "target_tokens": r["chosen_total"],
            "target_correct": True, "weight": 1.0, "from_cond": src,
            "uncut_rest_tokens": r["L"] - r["u"] + r["uncut_answer_tok"],
        })

    rng = random.Random(args.seed)
    n_before = len(examples)

    def cap(pool_pred, share, label):
        nonlocal examples
        pool = [e for e in examples if pool_pred(e)]
        rest = [e for e in examples if not pool_pred(e)]
        if 0 < share < 1 and rest and len(pool) > share / (1 - share) * len(rest):
            keep = int(share / (1 - share) * len(rest))
            rng.shuffle(pool)
            print(f"{label}: kept {keep} of {len(pool)} (cap {share:.0%})")
            pool = pool[:keep]
        examples = pool + rest

    cap(lambda e: e["state"] == "has conclusion", args.cap_concluded, "'has conclusion' examples")
    is_stop = lambda e: e["from_cond"] == "stop-now" or (e["target_think_tokens"] or 0) < args.stop_below
    cap(is_stop, args.max_stop_share, f"'stop now' targets (incl. continuations under {args.stop_below} tokens)")
    examples.sort(key=lambda e: (e["id"], e["sample"], e["cut_tokens"]))

    out = args.out or os.path.join("sft", os.path.basename(args.runs[0]).replace(".jsonl", "") + ".sft.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n{len(recs)} cut points read ({n_unsolved} from unsolved traces skipped), {n_before} with a target, "
          f"{len(examples)} examples written -> {out}")
    if no_trace:
        print(f"!! {no_trace} cut points had no trace in --traces (pass every trace file the runs came from)")
    print(f"\n{'state':<12}{'cut':>8}{'cut pts':>9}{'target':>9}")
    for (s, b), (n, k) in sorted(stats.items(), key=lambda kv: (kv[0][0], {"early": 0, "middle": 1, "late": 2}[kv[0][1]])):
        print(f"{s:<12}{b:>8}{n:>9}{k / n:>9.0%}")

    src = Counter(e["from_cond"] for e in examples)
    st = Counter(e["state"] for e in examples)
    per_problem = Counter(e["id"] for e in examples)
    print("\nexamples by source: " + ", ".join(f"{k} {v} ({v / len(examples):.0%})" for k, v in src.most_common()))
    print("examples by state:  " + ", ".join(f"{k} {v}" for k, v in st.most_common()))
    print(f"distinct problems: {len(per_problem)}; examples per problem: mean {np.mean(list(per_problem.values())):.1f}, "
          f"max {max(per_problem.values())}")
    th = np.array([e["target_think_tokens"] for e in examples])
    tot = np.array([e["target_tokens"] for e in examples])
    rest = np.array([e["uncut_rest_tokens"] for e in examples])
    cont = np.array([e["target_think_tokens"] for e in examples if e["from_cond"] == "sample"])
    if len(cont):
        print(f"continuations: {len(cont)}; thinking tokens <{args.stop_below} {np.mean(cont < args.stop_below):.0%} | "
              f"{args.stop_below}-199 {np.mean((cont >= args.stop_below) & (cont < 200)):.0%} | 200-599 "
              f"{np.mean((cont >= 200) & (cont < 600)):.0%} | 600+ {np.mean(cont >= 600):.0%} | median {np.median(cont):.0f}")
    print(f"target thinking tokens: p25 {np.percentile(th, 25):.0f} | median {np.median(th):.0f} | "
          f"p75 {np.percentile(th, 75):.0f} | p90 {np.percentile(th, 90):.0f}")
    print(f"tokens after the cut, target vs uncut (thinking + answer): median {np.median(tot):.0f} vs {np.median(rest):.0f} "
          f"| total saved {1 - tot.sum() / rest.sum():.0%}")
    mid = st.get("mid-way, solvable", 0)
    print(f"\nmid-way examples: {mid}")


if __name__ == "__main__":
    main()
