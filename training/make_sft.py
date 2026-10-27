#!/usr/bin/env python3
"""Builds the SFT data from the natural MATH train traces (gen_traces.py).

For every correct trace with thinking length L we make --cuts examples:
  cut  at p ~ U(0.3, 0.7) * L, moved forward to the next line break (at most 64 tokens) so the notice
       does not land inside a word
  W    ~ U(0.2, 0.6) * (L - p) tokens of thinking left, at least 64
  input   = prompt + thinking up to p + notice ("I only have about W tokens of thinking left ...")
  target  = continuation + </think> + "The final answer is \\boxed{...}." + <|im_end|>

Where the continuation comes from (rejection sampling, everything from the 4B itself unless --rewrites):
  plain   the base model continues the same prefix with no notice (--plain samples)
  nudged  the base model continues after the soft "running out of time" notice (--nudged samples)
  guided  more samples after a "keep going, skip the double-checks" notice, only for examples with no
          target yet (--guided)
  plainx  more no-notice samples, again only for examples with no target yet (--extra-plain, off by default)
  rewrite optional: a 27B model's rewrite of the rest of the original trace (rewrite.py output, --rewrites)

A candidate is kept only if
  1. it closes </think> within W tokens
  2. the 4B, given prefix + notice + candidate + </think>, then writes the right final answer
  3. it uses at least half of W (--min-use; rewrites only need --rewrite-min-use). Our first try let examples
     whose prefix already had the answer stop right away, and the model learned to stop right away after
     every notice, so now the rule holds for every example (--seen-short-ok brings the old rule back).
Per example we keep the longest that passes (with --prefer-rewrite, the longest rewrite if one passes).
At most --seen-max-share of the cut examples may have the answer in the prefix already, since AIME traces
mostly do not.
--uncut-share of the final data are whole natural traces with no cut, so normal behaviour does not drift.

Every stage is saved in --out, so running it again continues where it stopped, and adding --rewrites
later only checks the new candidates.

  python make_sft.py --traces traces --out sft
  python make_sft.py --traces traces --out sft_rewrite --reuse sft --rewrites rewrite_output.jsonl --prefer-rewrite
  python make_sft.py --traces mock_traces --out mock_sft --mock          (no GPU, checks the pipeline)
Outputs: sft.jsonl (token ids, for train_lora.py), stats.json, rewrite_input.jsonl (for rewrite.py)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import time
from pathlib import Path

import thinking_budget_bench as TB
from budget_drop_bench import WARNS

FINAL_PREFIX = "The final answer is \\boxed{"
NOTICES = {"count": WARNS["warn_count"],            # what the model sees at train and eval time
           "tag": "\n\n<budget_cut left={W}>\n\n"}
NUDGE = WARNS["warn_soft"]                          # only used to sample "nudged" candidates
GUIDE = ("\n\nI have about {W} tokens of thinking left. That is enough to finish if I skip double-checking, "
         "so I'll keep going with the main line of reasoning and wrap up before it runs out.\n")


def answer_seen(gold, text):
    """Rough: the gold answer shows up right after '=', 'is' or 'answer'."""
    g = re.escape(TB._norm(gold))
    return re.search(rf"(=|is|answer:?)\$?{g}(?![\d.])", TB._norm(text)) is not None


def boxed_inner(text):
    """text starts right after '\\boxed{'; return what is inside the matching brace, or None."""
    depth = 1
    for i, ch in enumerate(text):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return text[:i]
    return None


class Engine:
    def __init__(self, model, max_len, gpu_util, max_num_seqs):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=model, enable_prefix_caching=True, gpu_memory_utilization=gpu_util,
                       max_model_len=max_len, max_num_seqs=max_num_seqs)
        self.tok = AutoTokenizer.from_pretrained(model)

    def generate(self, prompts, max_tokens, seeds, stop_ids=None):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=int(m), seed=int(s),
                              stop_token_ids=stop_ids) for m, s in zip(max_tokens, seeds)]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


class Mock:
    """No GPU: fake model, real tokenizer. Only checks the pipeline."""

    def __init__(self, model, golds):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.golds = golds
        self.t_close = self.tok.convert_tokens_to_ids("</think>")
        self.line = self.tok.encode(" and we check it.\n", add_special_tokens=False)

    def generate(self, prompts, max_tokens, seeds, stop_ids=None):
        out = []
        for ids, m, s in zip(prompts, max_tokens, seeds):
            h = random.Random(len(ids) * 31 + int(s))
            if stop_ids:
                n = h.randint(1, max(1, int(m) // len(self.line) + 2)) * len(self.line)
                out.append((self.line * (n // len(self.line)) + [self.t_close], "stop") if n < m
                           else ((self.line * m)[:int(m)], "length"))
            else:
                text = self.tok.decode(ids[:600])
                gold = next((g for q, g in self.golds.items() if q in text), "0")
                out.append((self.tok.encode((gold if h.random() < 0.7 else "7") + "}.", add_special_tokens=False),
                            "stop"))
        return out


def jl_read(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def jl_append(path, rows):
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("sft"))
    ap.add_argument("--cuts", type=int, default=2, help="examples per trace")
    ap.add_argument("--min-L", type=int, default=600)
    ap.add_argument("--max-seq", type=int, default=10240, help="longest training sequence (tokens)")
    ap.add_argument("--plain", type=int, default=3, help="samples per example with no notice")
    ap.add_argument("--nudged", type=int, default=2, help="samples per example after the soft notice")
    ap.add_argument("--rewrites", type=Path, default=None, help="rewrite.py output to add as candidates")
    ap.add_argument("--notice", choices=list(NOTICES), default="count")
    ap.add_argument("--min-use", type=float, default=0.5, help="share of W a target must use")
    ap.add_argument("--seen-short-ok", action="store_true", help="first-try rule: no --min-use if the answer is already in the prefix")
    ap.add_argument("--guided", type=int, default=3, help="extra samples for examples with no target yet")
    ap.add_argument("--rewrite-min-use", type=float, default=0.25,
                    help="rewrites are compressed on purpose, so they only need this share of W")
    ap.add_argument("--prefer-rewrite", action="store_true",
                    help="if a rewrite passes, use it over the 4B's own samples (rewrites are the compressed ones)")
    ap.add_argument("--extra-plain", type=int, default=0,
                    help="more no-notice samples for examples still without a target (finds shorter paths)")
    ap.add_argument("--seen-max-share", type=float, default=0.3, help="cap on answer-already-seen examples")
    ap.add_argument("--reuse", type=Path, default=None, help="copy examples/samples/checks from an earlier --out")
    ap.add_argument("--uncut-share", type=float, default=0.2)
    ap.add_argument("--final-max", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=300, help="examples per saved chunk")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="only the first N traces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    if a.reuse:
        import shutil
        for f in ("examples.jsonl", "samples.jsonl", "checks.jsonl"):
            if (a.reuse / f).exists() and not (a.out / f).exists():
                shutil.copyfile(a.reuse / f, a.out / f)

    traces = jl_read(a.traces / "phase_a.jsonl")[:a.limit]
    golds = {r["q"][:60]: r["gold"] for r in traces}
    _eng = []

    def eng():                                                     # vLLM only starts if something needs generating
        if not _eng:
            _eng.append(Mock(a.model, golds) if a.mock else Engine(a.model, a.max_seq + 2048, a.gpu_util,
                                                                    a.max_num_seqs))
        return _eng[0]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    t_open, t_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    nl2, fp = enc("\n\n"), enc(FINAL_PREFIX)
    notice_ids = lambda W: enc(NOTICES[a.notice].format(W=W))

    def prompt_ids(q):
        text = tok.apply_chat_template([{"role": "user", "content": q + TB.INSTR}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return enc(text)

    # ------------------------------------------------ traces we can use: finished, correct, long enough
    recs = {}
    n_skip = collections.Counter()
    for r in traces:
        g = r["gen"]
        if t_close not in g:
            n_skip["ran out of tokens"] += 1
            continue
        start = g.index(t_open) + 1 if t_open in g else 0
        end = g.index(t_close)
        ans_text = tok.decode(g[end + 1:], skip_special_tokens=True)
        if not TB.grade("math500", ans_text, r["gold"]):
            n_skip["wrong"] += 1
            continue
        if end - start < a.min_L:
            n_skip["too short"] += 1
            continue
        recs[r["sid"]] = {**r, "start": start, "end": end, "L": end - start, "P": prompt_ids(r["q"]),
                          "nat_box": TB.last_boxed(ans_text)}
    print(f"{len(recs)} usable traces of {len(traces)} (skipped: {dict(n_skip)})", flush=True)

    # ------------------------------------------------ examples (cut point + budget), saved once
    ex_path = a.out / "examples.jsonl"
    examples = jl_read(ex_path)
    have = {e["sid"] for e in examples}
    new = []
    for sid, r in recs.items():                                    # traces added since the last run get examples too
        if sid in have:
            continue
        g, s, rr = r["gen"], r["start"], random.Random(f"{a.seed}-{sid}")
        for c in range(a.cuts):
            raw = int(rr.uniform(0.3, 0.7) * r["L"])
            cut = next((j for j in range(raw, min(raw + 64, r["L"] - 64))
                        if "\n" in tok.decode(g[s + j - 1:s + j])), raw)
            rest = r["L"] - cut
            W = max(64, int(rr.uniform(0.2, 0.6) * rest))
            new.append({"ex": f"{sid}#c{c}", "sid": sid, "L": r["L"], "cut": cut, "W": W,
                        "answer_seen": answer_seen(r["gold"], tok.decode(g[s:s + cut]))})
    jl_append(ex_path, new)
    examples += new
    with (a.out / "rewrite_input.jsonl").open("w") as f:          # same format as make_rewrite_input.py
        for e in examples:
            if e["sid"] not in recs:
                continue
            r = recs[e["sid"]]
            g, s = r["gen"], r["start"]
            f.write(json.dumps({"id": e["ex"], "problem": r["q"], "gold": r["gold"], "L": e["L"],
                                "cut": e["cut"], "budget_tokens": e["W"], "approx_words": int(e["W"] * 0.7),
                                "notice": NOTICES["count"].format(W=e["W"]),
                                "has_answer_already": e["answer_seen"],
                                "prefix": tok.decode(g[s:s + e["cut"]]),
                                "original_rest": tok.decode(g[s + e["cut"]:r["end"]]),
                                "rest_tokens": e["L"] - e["cut"]}) + "\n")
    examples = [e for e in examples if e["sid"] in recs]
    exd = {e["ex"]: e for e in examples}
    print(f"{len(examples)} cut examples, answer already in the prefix for "
          f"{sum(e['answer_seen'] for e in examples)}", flush=True)

    def base_ids(e):
        r = recs[e["sid"]]
        return r["P"] + r["gen"][:r["start"] + e["cut"]]

    # ------------------------------------------------ sampling, answer check, verdict
    smp_path, chk_path = a.out / "samples.jsonl", a.out / "checks.jsonl"
    checked = {c["cand"]: c for c in jl_read(chk_path)}

    def sample(todo, specs, seed0, label):
        """specs: [(src, n, extra_ids(W))]. Appends to samples.jsonl and returns the new rows."""
        print(f"sampling {label} continuations for {len(todo)} examples", flush=True)
        t_all, out = time.time(), []
        for c in range(0, len(todo), a.chunk):
            part = todo[c:c + a.chunk]
            jobs = [(e, src, k, base_ids(e) + extra(e["W"]), e["W"]) for e in part for src, n, extra in specs
                    for k in range(n)]
            outs = eng().generate([j[3] for j in jobs], [j[4] for j in jobs],
                                  [seed0 + 101 * i for i in range(len(jobs))], stop_ids=[t_close])
            rows = []
            for (e, src, k, _, W), (ids, why) in zip(jobs, outs):
                closed = why == "stop" or (ids and ids[-1] == t_close)
                ids = ids[:-1] if ids and ids[-1] == t_close else ids
                rows.append({"cand": f"{e['ex']}#{src}{k}", "ex": e["ex"], "src": src, "closed": bool(closed),
                             "n": len(ids), "ids": ids if closed else None})
            jl_append(smp_path, rows)
            out += rows
            el = time.time() - t_all
            print(f"  {c + len(part)}/{len(todo)} examples, about "
                  f"{el / (c + len(part)) * (len(todo) - c - len(part)) / 60:.0f} min left", flush=True)
        return out

    def check(cs):
        """Final-answer check (prefix + notice + candidate + </think> -> answer) for candidates that fit W."""
        need = [c for c in cs if c["closed"] and c["n"] + 1 <= exd[c["ex"]]["W"] and c["cand"] not in checked]
        if not need:
            return
        print(f"checking final answers for {len(need)} candidates", flush=True)
        for c0 in range(0, len(need), 4 * a.chunk):
            part = need[c0:c0 + 4 * a.chunk]
            prompts = [base_ids(exd[c["ex"]]) + notice_ids(exd[c["ex"]]["W"]) + c["ids"] + [t_close] + nl2 + fp
                       for c in part]
            outs = eng().generate(prompts, [a.final_max] * len(part), [a.seed + 7] * len(part))
            rows = []
            for c, (ids, _) in zip(part, outs):
                text = tok.decode(ids, skip_special_tokens=True)
                gold = recs[exd[c["ex"]]["sid"]]["gold"]
                rows.append({"cand": c["cand"], "correct": TB.grade("math500", "\\boxed{" + text, gold),
                             "box": boxed_inner(text)})
            jl_append(chk_path, rows)
            checked.update({r["cand"]: r for r in rows})

    def verdict(c):
        e = exd[c["ex"]]
        if not c["closed"]:
            return "did not close in W"
        if c["n"] + 1 > e["W"]:
            return "longer than W"
        if not checked.get(c["cand"], {}).get("correct"):
            return "wrong answer"
        if not checked[c["cand"]].get("box"):
            return "answer not closed"
        need = a.rewrite_min_use if c["src"] == "rewrite" else a.min_use
        if not (a.seen_short_ok and e["answer_seen"]) and c["n"] + 1 < need * e["W"]:
            return "stopped too early"
        return "PASS"

    # stage 1: plain + nudged for every example
    done_ex = {s["ex"] for s in jl_read(smp_path) if s["src"] in ("plain", "nudged")}
    todo = [e for e in examples if e["ex"] not in done_ex]
    if todo and (a.plain or a.nudged):
        sample(todo, [("plain", a.plain, lambda W: []), ("nudged", a.nudged, lambda W: enc(NUDGE))],
               a.seed, f"{a.plain} plain + {a.nudged} nudged")
    cands = [s for s in jl_read(smp_path) if s["ex"] in exd]
    if a.rewrites:                                                 # 27B rewrites, if we have them
        n_rw = 0
        for r in jl_read(a.rewrites):
            if r["id"] not in exd or not r.get("rewrite"):
                continue
            text = r["rewrite"].replace("</think>", "").strip()
            ids = enc("\n" + text + "\n")
            cands.append({"cand": f"{r['id']}#rewrite{r.get('sample', 0)}", "ex": r["id"], "src": "rewrite",
                          "closed": r.get("finish", "stop") == "stop", "n": len(ids), "ids": ids})
            n_rw += 1
        print(f"added {n_rw} rewrite candidates from {a.rewrites}", flush=True)
    check(cands)

    # stage 2: examples with no target yet get --guided samples after a "keep going" notice
    have_pass = {c["ex"] for c in cands if verdict(c) == "PASS"}
    tried = {c["ex"] for c in cands if c["src"] == "guided"}
    todo = [e for e in examples if e["ex"] not in have_pass and e["ex"] not in tried]
    if todo and a.guided:
        new_c = sample(todo, [("guided", a.guided, lambda W: enc(GUIDE.format(W=W)))], a.seed + 7919, "guided")
        cands += new_c
        check(new_c)

    # stage 3 (v3): still nothing -> more plain samples; the base model sometimes finds a short enough path
    have_pass = {c["ex"] for c in cands if verdict(c) == "PASS"}
    tried = {c["ex"] for c in cands if c["src"] == "plainx"}
    todo = [e for e in examples if e["ex"] not in have_pass and e["ex"] not in tried]
    if todo and a.extra_plain:
        new_c = sample(todo, [("plainx", a.extra_plain, lambda W: [])], a.seed + 104729, "extra plain")
        cands += new_c
        check(new_c)

    # ------------------------------------------------ filter and pick one target per example (the longest)
    why = collections.Counter()
    by_ex = collections.defaultdict(list)
    for c in cands:
        v = verdict(c)
        why[f"{c['src']}: {v}"] += 1
        if v == "PASS":
            by_ex[c["ex"]].append(c)

    seen_c, unseen_c = [], []
    for ex, cs in by_ex.items():
        e = exd[ex]
        if a.prefer_rewrite and any(c["src"] == "rewrite" for c in cs):
            cs = [c for c in cs if c["src"] == "rewrite"]
        c = max(cs, key=lambda c: c["n"])
        inp = base_ids(e) + notice_ids(e["W"])
        tgt = c["ids"] + [t_close] + enc("\n\n" + FINAL_PREFIX + checked[c["cand"]]["box"] + "}.") + [im_end]
        if len(inp) + len(tgt) > a.max_seq:
            why["too long for --max-seq"] += 1
            continue
        (seen_c if e["answer_seen"] else unseen_c).append(
            ({"id": ex, "kind": "cut", "src": c["src"], "W": e["W"], "answer_seen": e["answer_seen"],
              "input_ids": inp, "target_ids": tgt}, (c["n"] + 1) / e["W"]))
    n_seen = min(len(seen_c), int(a.seen_max_share / (1 - a.seen_max_share) * len(unseen_c)) if a.seen_max_share < 1
                 else len(seen_c))
    rng.shuffle(seen_c)
    why["answer seen: dropped by --seen-max-share"] += len(seen_c) - n_seen
    kept = unseen_c + seen_c[:n_seen]
    data = [d for d, _ in kept]
    use = [u for _, u in kept]
    picked_src = collections.Counter(d["src"] for d in data)

    n_uncut = round(len(data) * a.uncut_share / (1 - a.uncut_share))
    pool = [r for r in recs.values() if r["nat_box"] and len(r["P"]) + r["end"] + 64 <= a.max_seq]
    rng.shuffle(pool)
    for r in pool[:n_uncut]:
        tgt = r["gen"][:r["end"] + 1] + enc("\n\n" + FINAL_PREFIX + r["nat_box"] + "}.") + [im_end]
        data.append({"id": r["sid"], "kind": "uncut", "src": "natural", "W": None, "answer_seen": None,
                     "input_ids": r["P"], "target_ids": tgt})
    rng.shuffle(data)
    with (a.out / "sft.jsonl").open("w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    cut = [d for d in data if d["kind"] == "cut"]
    stats = {"usable_traces": len(recs), "cut_examples": len(examples),
             "examples_with_a_target": len(cut), "uncut_examples": len(data) - len(cut), "total": len(data),
             "picked_from": dict(picked_src), "candidates": dict(sorted(why.items())),
             "answer_seen_share": round(sum(d["answer_seen"] for d in cut) / max(1, len(cut)), 3),
             "W_median": statistics.median(d["W"] for d in cut) if cut else None,
             "budget_use_median": round(statistics.median(use), 2) if use else None,
             "target_tokens_median": statistics.median(len(d["target_ids"]) for d in cut) if cut else None,
             "seq_tokens_median": statistics.median(len(d["input_ids"]) + len(d["target_ids"]) for d in data)
             if data else None, "notice": NOTICES[a.notice]}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {a.out / 'sft.jsonl'}", flush=True)
    if _eng:                                       # shut vLLM down here, not at interpreter exit (that can hang)
        import gc
        _eng.clear()
        gc.collect()


if __name__ == "__main__":
    main()
