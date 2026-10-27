#!/usr/bin/env python3
"""Step 1: natural Qwen3-4B traces on the MATH train split (levels 3-5), the raw material for the SFT data.

MATH-500 and AIME 2026 are kept for evaluation; MATH-500 comes from the MATH test split, so the train split
does not overlap with it (checked again below by problem text).

Writes --out/phase_a.jsonl in the same format as aime_drop_bench.py (plus q, gold, level), in chunks, so a
stopped run keeps what it finished and just continues when started again.

  python gen_traces.py --out traces                 (2000 problems, roughly 1-2 h on an RTX 4090)
  python gen_traces.py --out traces --n 300         (quick)
  python gen_traces.py --out mock_traces --n 20 --mock
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import thinking_budget_bench as TB

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus"]


def load_math_train(levels, seed):
    from datasets import load_dataset
    test_q = {r["problem"].strip() for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    items = []
    for s in SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="train")):
            try:
                lv = int(r["level"].split()[-1])
            except ValueError:
                continue
            gold = TB.last_boxed(r["solution"])
            if lv not in levels or gold is None or r["problem"].strip() in test_q:
                continue
            items.append({"id": f"mathtrain/{s}/{i}", "q": r["problem"], "gold": gold, "level": lv})
    random.Random(seed).shuffle(items)
    return items


class Mock:
    """No GPU: fake model with the real tokenizer. Only checks the pipeline."""

    def __init__(self, model, problems):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.gold = {p["q"][:60]: p["gold"] for p in problems}
        self.t_open = self.tok.convert_tokens_to_ids("<think>")
        self.t_close = self.tok.convert_tokens_to_ids("</think>")
        self.line = self.tok.encode(" so we try the next step.\n", add_special_tokens=False)

    def generate(self, prompts, max_tokens, seed):
        out = []
        for ids in prompts:
            h = random.Random(len(ids) * 7919 + seed)
            text = self.tok.decode(ids)
            gold = next((g for k, g in self.gold.items() if k in text), "0")
            body = self.line * h.randint(60, 700)
            ans = gold if h.random() < 0.85 else "wrong"
            seq = [self.t_open] + body + [self.t_close] + self.tok.encode(
                f"\n\nSo the answer is $\\boxed{{{ans}}}$.", add_special_tokens=False)
            out.append((seq[:max_tokens], "stop" if len(seq) <= max_tokens else "length"))
        return out


class Engine:
    def __init__(self, model, max_len, gpu_util, max_num_seqs):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=model, enable_prefix_caching=True, gpu_memory_utilization=gpu_util,
                       max_model_len=max_len, max_num_seqs=max_num_seqs)
        self.tok = AutoTokenizer.from_pretrained(model)

    def generate(self, prompts, max_tokens, seed):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=seed + i)
               for i in range(len(prompts))]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=2000, help="problems")
    ap.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--max-think", type=int, default=12000, help="longer traces are dropped later anyway")
    ap.add_argument("--answer-max", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=200, help="problems per save")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("traces"))
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    problems = load_math_train(set(a.levels), a.seed)[:a.n]
    path = a.out / "phase_a.jsonl"
    done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
    todo = [p for p in problems if p["id"] not in done]
    print(f"{len(problems)} MATH train problems (levels {a.levels}); {len(done)} already done, {len(todo)} to go",
          flush=True)

    max_len = 2048 + a.max_think + a.answer_max
    eng = Mock(a.model, problems) if a.mock else Engine(a.model, max_len, a.gpu_util, a.max_num_seqs)
    tok = eng.tok

    def prompt_ids(q):
        text = tok.apply_chat_template([{"role": "user", "content": q + TB.INSTR}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return tok.encode(text, add_special_tokens=False)

    t_all, ntok_all = time.time(), 0
    for c in range(0, len(todo), a.chunk):
        part = todo[c:c + a.chunk]
        t0 = time.time()
        outs = eng.generate([prompt_ids(p["q"]) for p in part], a.max_think + a.answer_max, a.seed + len(done) + c)
        with path.open("a") as f:
            for p, (ids, why) in zip(part, outs):
                f.write(json.dumps({"sid": p["id"] + "#s0", "id": p["id"], "k": 0, "q": p["q"], "gold": p["gold"],
                                    "level": p["level"], "finish": why, "gen": ids}) + "\n")
        ntok = sum(len(o[0]) for o in outs)
        ntok_all += ntok
        left = len(todo) - c - len(part)
        rate = ntok_all / (time.time() - t_all)
        print(f"  {c + len(part)}/{len(todo)} done, {ntok / (time.time() - t0):.0f} tok/s this chunk, "
              f"about {left * ntok_all / max(1, c + len(part)) / rate / 60:.0f} min left", flush=True)

    # quick summary of what we got
    t_open, t_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    Ls, ok, capped = [], 0, 0
    rows = [json.loads(l) for l in path.open()]
    for r in rows:
        g = r["gen"]
        if t_close not in g:
            capped += 1
            continue
        start = g.index(t_open) + 1 if t_open in g else 0
        end = g.index(t_close)
        Ls.append(end - start)
        ok += TB.grade("math500", tok.decode(g[end + 1:], skip_special_tokens=True), r["gold"])
    summ = {"traces": len(rows), "finished_thinking": len(Ls), "ran_out": capped,
            "correct_of_finished": round(ok / max(1, len(Ls)), 3),
            "thinking_tokens": {"median": statistics.median(Ls) if Ls else None,
                                "p10": sorted(Ls)[len(Ls) // 10] if Ls else None,
                                "p90": sorted(Ls)[9 * len(Ls) // 10] if Ls else None}}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
