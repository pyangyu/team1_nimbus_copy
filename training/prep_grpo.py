#!/usr/bin/env python3
"""Build the GRPO prompts: problem + thinking up to the cut + the countdown notice with W (same as the SFT
input), plus W and the gold answer for the reward.

Two ways to get the cut points:
  --examples  reuse the cut examples made by make_sft.py (the first GRPO run, MATH train traces)
  no --examples  draw them here from traces with several samples per problem (gen_traces.py --samples 4).
              Problems the model never solves are dropped (--pass-range): after a cut they are wrong anyway,
              so their GRPO groups have no learning signal. Problems it always solves stay in: a cut still
              makes it fail some of the time, and keeping its accuracy after a cut is the point. The cut lands on any trace
              that finished thinking, right or wrong, the same way the AIME eval cuts its traces.

Examples whose prefix already has the answer are capped at --seen-share, since they mostly give groups where
every sample is right and AIME traces rarely look like that.

  python prep_grpo.py --traces traces --examples sft/examples.jsonl --out grpo_prompts.jsonl
  python prep_grpo.py --traces hard --out hard_prompts.jsonl --pass-range 0.25 1.0 --cuts 3
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

import thinking_budget_bench as TB
from budget_drop_bench import WARNS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--traces", type=Path, required=True)
ap.add_argument("--examples", type=Path, help="cut examples from make_sft.py; without it, cuts are drawn here")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--pass-range", type=float, nargs=2, default=[0.25, 1.0],
                help="without --examples: keep problems whose samples are right this share of the time")
ap.add_argument("--cuts", type=int, default=3, help="without --examples: cuts per kept problem")
ap.add_argument("--min-L", type=int, default=400, help="without --examples: skip shorter thinking")
ap.add_argument("--seen-share", type=float, default=0.3)
ap.add_argument("--max-prompt", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
enc = lambda s: tok.encode(s, add_special_tokens=False)
t_open, t_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
traces = {json.loads(l)["sid"]: json.loads(l) for l in (a.traces / "phase_a.jsonl").open()}


def prompt_ids(q):
    text = tok.apply_chat_template([{"role": "user", "content": q + TB.INSTR}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    return enc(text)


def answer_seen(gold, text):
    """Same rough check as make_sft.py: the gold answer shows up right after '=', 'is' or 'answer'."""
    g = re.escape(TB._norm(gold))
    return re.search(rf"(=|is|answer:?)\$?{g}(?![\d.])", TB._norm(text)) is not None


def draw_examples():
    """Cut points for problems the model solves only part of the time."""
    by_problem = collections.defaultdict(list)
    for sid, r in traces.items():
        g = r["gen"]
        finished = t_close in g
        start = g.index(t_open) + 1 if t_open in g else 0
        end = g.index(t_close) if finished else None
        right = finished and TB.grade("math500", tok.decode(g[end + 1:], skip_special_tokens=True), r["gold"])
        by_problem[r["id"]].append({"sid": sid, "finished": finished, "right": right, "start": start,
                                    "L": (end - start) if finished else None})
    lo, hi = a.pass_range
    kept = {pid: ts for pid, ts in by_problem.items() if lo <= sum(t["right"] for t in ts) / len(ts) <= hi}
    out = []
    for pid, ts in kept.items():
        cands = [t for t in ts if t["finished"] and t["L"] >= a.min_L]
        rr = random.Random(f"{a.seed}-{pid}")
        for c in range(min(a.cuts, 2 * len(cands))):
            t = cands[c % len(cands)] if c < len(cands) else rr.choice(cands)
            g, s = traces[t["sid"]]["gen"], t["start"]
            raw = int(rr.uniform(0.3, 0.7) * t["L"])
            cut = next((j for j in range(raw, min(raw + 64, t["L"] - 64))
                        if "\n" in tok.decode(g[s + j - 1:s + j])), raw)
            W = max(64, int(rr.uniform(0.2, 0.6) * (t["L"] - cut)))
            out.append({"ex": f"{t['sid']}#c{c}", "sid": t["sid"], "cut": cut, "W": W,
                        "answer_seen": answer_seen(traces[t["sid"]]["gold"], tok.decode(g[s:s + cut]))})
    print(f"{len(by_problem)} problems, {len(kept)} solved {lo:.0%}-{hi:.0%} of the time, "
          f"{len(out)} cut points", flush=True)
    return out


examples = [json.loads(l) for l in a.examples.open()] if a.examples else draw_examples()

seen, unseen = [], []
for e in examples:
    r = traces.get(e["sid"])
    if r is None:
        continue
    g = r["gen"]
    start = g.index(t_open) + 1 if t_open in g else 0
    ids = prompt_ids(r["q"]) + g[:start + e["cut"]] + enc(WARNS["warn_count"].format(W=e["W"]))
    if len(ids) > a.max_prompt:
        continue
    row = {"id": e["ex"], "input_ids": ids, "W": e["W"], "gold": r["gold"], "answer_seen": e["answer_seen"]}
    (seen if e["answer_seen"] else unseen).append(row)

rng = random.Random(a.seed)
rng.shuffle(seen)
keep_seen = int(a.seen_share / (1 - a.seen_share) * len(unseen))
rows = unseen + seen[:keep_seen]
rng.shuffle(rows)
with a.out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} prompts ({len(unseen)} without the answer yet, {min(keep_seen, len(seen))} with) -> {a.out}")
