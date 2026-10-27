#!/usr/bin/env python3
"""Build the GRPO prompts from the MATH train traces and the cut examples made by make_sft.py.

Each prompt = problem + thinking up to the cut + the countdown notice with W (same as the SFT input), plus W and
the gold answer for the reward. Examples whose prefix already has the answer are capped at --seen-share, since
they mostly give groups where every sample is right (no learning signal) and AIME traces rarely look like that.

  python prep_grpo.py --traces traces --examples sft/examples.jsonl --out grpo_prompts.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import thinking_budget_bench as TB
from budget_drop_bench import WARNS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--traces", type=Path, required=True)
ap.add_argument("--examples", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seen-share", type=float, default=0.3)
ap.add_argument("--max-prompt", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
enc = lambda s: tok.encode(s, add_special_tokens=False)
t_open = tok.convert_tokens_to_ids("<think>")
traces = {json.loads(l)["sid"]: json.loads(l) for l in (a.traces / "phase_a.jsonl").open()}
examples = [json.loads(l) for l in a.examples.open()]


def prompt_ids(q):
    text = tok.apply_chat_template([{"role": "user", "content": q + TB.INSTR}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    return enc(text)


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
