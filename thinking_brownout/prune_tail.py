#!/usr/bin/env python3
"""Hindsight targets by DELETION ONLY: ask the model which paragraphs of its own remaining
thinking were really needed, keep those, drop the rest, and check the result by replay.

For a cut point, the uncut trace tells us what the model went on to think and which answer
it gave. The model is shown the prefix, the numbered paragraphs that followed, and that
answer, and returns the numbers of the paragraphs to keep. No new text is written, so every
kept step is one the model really derived. The shortened tail is then replayed: prefix +
notice + tail + close, and the model must give the same final answer (or the correct one).
Accepted tails are appended to the runs file as condition "prune", where make_sft.py
competes them against the sampled continuations.

  python prune_tail.py runs/Qwen__Qwen3.8-27B.sft.jsonl traces/Qwen__Qwen3.8-27B.jsonl \\
      --model Qwen/Qwen3.8-27B --smoke
  python prune_tail.py runs/Qwen__Qwen3.8-27B.sft.jsonl traces/Qwen__Qwen3.8-27B.jsonl \\
      --model Qwen/Qwen3.8-27B
"""
import argparse
import asyncio
import json
import re

from cut_bench import prepare
from cutlib import (CLOSE, Server, clean, final_box, find_think_open, graded, messages_for,
                    notice_text, read_traces, same)

ASK = """You are shortening your own reasoning. No new reasoning may be written.

PROBLEM:
{problem}

REASONING ALREADY WRITTEN (it stays as it is):
{prefix}

WHAT CAME AFTER IT, as numbered paragraphs. It ended with the final answer {final}:
{rest}

The reasoning must now end as fast as possible. Choose the smallest set of the numbered
paragraphs that, read right after the reasoning already written, still reaches the final
answer {final} by valid steps. Drop repeated checks, second methods used only to confirm,
doubts about the wording that were then resolved, and drafts of the reply. Keep every step
the answer really depends on. If the reasoning already written contains the conclusion,
keep nothing.

Reply with only a JSON list of the paragraph numbers to keep, in order. Example: [2, 5, 6]"""


def split_units(rest, max_units=150):
    units = [u for u in re.split(r"\n\s*\n", rest) if u.strip()]
    if len(units) < 3:
        units = [u for u in rest.split("\n") if u.strip()]
    while len(units) > max_units:                       # merge neighbours until it fits
        units = [units[i] + "\n\n" + units[i + 1] if i + 1 < len(units) else units[i]
                 for i in range(0, len(units), 2)]
    return units


def parse_keep(reply, n):
    m = re.search(r"\[[\d,\s]*\]", reply or "")
    if not m:
        return None
    keep = sorted({int(x) for x in re.findall(r"\d+", m.group(0))})
    return [k for k in keep if 1 <= k <= n]


async def main_async(args):
    traces = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
    cuts, levels, have = {}, set(), set()
    for line in open(args.runs):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r["id"], r["sample"], r["u"])
        if r["kind"] == "prune":
            have.add(key)
        else:
            cuts.setdefault(key, r)
            levels.add(r["grace"])
    levels = [args.max_tail]
    todo = [(k, m) for k, m in cuts.items() if k not in have and k[:2] in traces
            and traces[k[:2]].get("finished") and final_box(traces[k[:2]].get("answer")) is not None
            and (args.include_wrong or m["uncut_correct"])
            and not (args.skip_concluded and m["prefix_has_final_box"])]
    if args.max_cuts:
        todo = todo[: args.max_cuts]
    print(f"{len(cuts)} cut points in the runs file, {len(todo)} to prune (tails over {levels[0]} tokens are dropped)")
    srv = Server(args.base_url, args.model, args.concurrency)
    think_open = await find_think_open(srv, messages_for(traces[todo[0][0][:2]], args.system), args.template_kwargs)
    stats = {"asked": 0, "no_list": 0, "empty": 0, "too_long": 0, "accepted": 0, "rejected": 0, "tok_before": 0, "tok_after": 0}
    lock = asyncio.Lock()
    fout = None if args.smoke else open(args.runs, "a")

    async def one(key, meta):
        t = traces[key[:2]]
        prefix, rest = t["thinking"][: meta["prefix_chars"]], t["thinking"][meta["prefix_chars"]:]
        if len(prefix) + len(rest) > args.max_chars:
            stats["too_long"] += 1
            return
        units = split_units(rest)
        final = final_box(t["answer"])
        numbered = "\n\n".join(f"[{i + 1}] {u.strip()}" for i, u in enumerate(units))
        reply = await srv.chat([{"role": "user", "content": ASK.format(problem=t["prompt"], prefix=prefix,
                                                                      final=final, rest=numbered)}],
                               600, args.judge_kwargs)
        stats["asked"] += 1
        keep = parse_keep(reply, len(units))
        if keep is None:
            stats["no_list"] += 1
            return
        if not keep:
            stats["empty"] += 1                          # same as the "hard" condition, already covered
            return
        tail = "\n\n".join(units[k - 1].strip() for k in keep)
        n_tail = len(await srv.tok(tail))
        if n_tail > args.max_tail:
            stats["too_long"] += 1
            return
        base_ids, think_ids = await prepare(srv, t, args, think_open)
        ids = (base_ids + think_ids[: key[2]] + await srv.tok(notice_text("notice", 0), cache=True)
               + await srv.tok(tail + "\n" + CLOSE + "\n\n"))
        recs = []
        for k in range(args.k):
            p = await srv.complete(ids, args.answer_max, 7000 + k)
            ans = clean(p["text"])
            recs.append(dict({x: meta[x] for x in ("id", "sample", "source", "gold", "L", "u", "frac", "prefix_chars",
                                                   "uncut_correct", "uncut_finished", "uncut_answer_tok",
                                                   "prefix_has_final_box")},
                             key=f"{key[0]}|{key[1]}|{key[2]}|prune|{k}", cond="prune", kind="prune", grace=n_tail, k=k,
                             think_after=tail, think_after_tok=n_tail, self_term=True, answer=ans,
                             answer_tok=p["n"], answer_finish=p["finish"], correct=graded(ans, t["gold"]),
                             kept=keep, n_units=len(units)))
        good = sum(r["correct"] or (final_box(r["answer"]) is not None and same(final_box(r["answer"]), final))
                   for r in recs)
        async with lock:
            stats["accepted" if good == args.k else "rejected"] += 1
            stats["tok_before"] += meta["L"] - key[2]
            stats["tok_after"] += n_tail
            if args.smoke:
                print(f"\n--- {key} | kept {len(keep)} of {len(units)} paragraphs: {keep} | "
                      f"{meta['L'] - key[2]} -> {n_tail} thinking tokens | replays with the same answer: {good}/{args.k}")
                print("tail:\n   ", tail[:700].replace("\n", "\n    "))
            else:
                for r in recs:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()

    await asyncio.gather(*[one(k, m) for k, m in (todo[:3] if args.smoke else todo)])
    if fout:
        fout.close()
    await srv.close()
    s = stats
    print(f"\nasked {s['asked']} | no list in the reply {s['no_list']} | kept nothing {s['empty']} | "
          f"too long {s['too_long']} | replay kept the answer {s['accepted']} | replay changed it {s['rejected']}")
    if s["tok_before"]:
        print(f"thinking after the cut, over the replayed cut points: {s['tok_before']} -> {s['tok_after']} tokens "
              f"({1 - s['tok_after'] / s['tok_before']:.0%} removed)")
    if not args.smoke:
        print(f"appended to {args.runs}. Next: python make_sft.py {args.runs} {args.traces}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("traces")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--k", type=int, default=2, help="replays per pruned tail; all must keep the answer")
    ap.add_argument("--max-tail", type=int, default=2048, help="drop pruned tails longer than this many tokens")
    ap.add_argument("--include-wrong", action="store_true",
                    help="also prune traces whose uncut answer was wrong (the target then keeps that same answer)")
    ap.add_argument("--skip-concluded", action="store_true", help="skip cut points that already had the conclusion")
    ap.add_argument("--max-cuts", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=90000, help="skip when prefix + rest is longer than this")
    ap.add_argument("--answer-max", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--template-kwargs", type=json.loads, default={"enable_thinking": True})
    ap.add_argument("--judge-kwargs", type=json.loads, default={"enable_thinking": False},
                    help="chat template settings for the pruning question (thinking off)")
    ap.add_argument("--system", default=None)
    ap.add_argument("--smoke", action="store_true", help="three cut points, print, save nothing")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
