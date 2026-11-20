#!/usr/bin/env python3
"""Hindsight targets by DELETION ONLY: ask the model which paragraphs of its own remaining
thinking were really needed, keep those, drop the rest, and check the result by replay.

For a cut point, the uncut trace tells us what the model went on to think and which answer
it gave. The model is shown the prefix, the numbered paragraphs that followed, and that
answer, and returns the numbers of the paragraphs to keep. No new text is written, so every
kept step is one the model really derived. The shortened tail is then replayed twice:
  plain    prefix + notice + tail + close: the model must give the right answer
  masked   the same with the final answer hidden inside the tail ("35" -> "?"): the kept STEPS
           must lead to the answer, so a tail that only restates the conclusion is rejected
The judge thinks before it chooses (thinking on, offline).
Accepted tails are appended to the runs file as condition "prune", where make_sft.py
competes them against the sampled continuations.

  python prune_tail.py runs/Qwen__Qwen3.8-27B.sft.jsonl traces/Qwen__Qwen3.8-27B.jsonl \\
      --model Qwen/Qwen3.8-27B --smoke            # 50 cut points, writes prune_smoke.md
  python prune_tail.py runs/Qwen__Qwen3.8-27B.sft.jsonl traces/Qwen__Qwen3.8-27B.jsonl \\
      --model Qwen/Qwen3.8-27B
"""
import argparse
import asyncio
import json
import random
import re
from collections import defaultdict

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
    m = re.findall(r"\[[\d,\s]*\]", reply or "")
    if not m:
        return None
    keep = sorted({int(x) for x in re.findall(r"\d+", m[-1])})      # the last list in the reply
    return [k for k in keep if 1 <= k <= n]


def _replace_boxed(text):
    """\boxed{...} -> \boxed{?} (nested braces handled). Returns (text, count)."""
    out, i, n = [], 0, 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out), n
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[i:j] + "\\boxed{?}")
        n += 1
        i = k


def mask_answer(text, final):
    """Hide the final answer in a kept tail, so a replay can only succeed if the kept STEPS lead
    to it (a tail that is just "so the answer is 35" fails). Returns (masked text, replacements)."""
    text, n = _replace_boxed(text)
    forms = {final.strip()}
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", final.strip()):
        bare = final.strip().replace(",", "")
        forms |= {bare, f"{int(float(bare)):,}" if float(bare).is_integer() else bare}
    if final.startswith("\\frac") or final.startswith("\\dfrac"):
        forms |= {final.replace("\\dfrac", "\\frac"), final.replace("\\frac", "\\dfrac")}
        m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", final.strip())
        if m:
            forms.add(f"{m.group(1)}/{m.group(2)}")
    for f in sorted((x for x in forms if x), key=len, reverse=True):
        pat = (r"(?<!\w)(?<!\d\.)" + re.escape(f) + r"(?!\w)(?!\.\d)") if re.fullmatch(r"[-\d.,/]+", f) else re.escape(f)
        text, c = re.subn(pat, "?", text)
        n += c
    return text, n


def _state(meta, hard_ok):
    if meta["prefix_has_final_box"] or hard_ok:
        return "has conclusion"
    return "mid-way"


async def main_async(args):
    traces = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
    cuts, have, hard = {}, set(), {}
    for line in open(args.runs):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r["id"], r["sample"], r["u"])
        if r["kind"] == "prune":
            have.add(key)
            continue
        cuts.setdefault(key, r)
        if r["kind"] == "hard":
            hard.setdefault(key, []).append(bool(r["correct"]))
    hard_ok = {k: bool(v) and all(v) for k, v in hard.items()}
    todo = [(k, m) for k, m in cuts.items() if (args.smoke or k not in have) and k[:2] in traces
            and traces[k[:2]].get("finished") and final_box(traces[k[:2]].get("answer")) is not None
            and (args.include_wrong or m["uncut_correct"])
            and (args.include_concluded or _state(m, hard_ok.get(k, False)) != "has conclusion")]
    rng = random.Random(0)
    if args.smoke:      # mostly mid-way cut points: that is where pruning matters
        mid = [x for x in todo if _state(x[1], hard_ok.get(x[0], False)) == "mid-way"]
        con = [x for x in todo if _state(x[1], hard_ok.get(x[0], False)) != "mid-way"]
        rng.shuffle(mid); rng.shuffle(con)
        n_mid = min(len(mid), int(args.smoke * 0.8))
        todo = mid[:n_mid] + con[: args.smoke - n_mid]
    elif args.max_cuts:
        todo = todo[: args.max_cuts]
    if not todo:
        raise SystemExit("nothing to prune (check --include-wrong / --skip-concluded, or the runs file)")
    print(f"{len(cuts)} cut points in the runs file, {len(todo)} to prune"
          + (" (smoke: nothing is saved to the runs file)" if args.smoke else ""))
    srv = Server(args.base_url, args.model, args.concurrency)
    think_open = await find_think_open(srv, messages_for(traces[todo[0][0][:2]], args.system), args.template_kwargs)
    notice = await srv.tok(notice_text("notice", 0), cache=True)
    S = defaultdict(lambda: defaultdict(int))
    lock = asyncio.Lock()
    examples = []
    fout = None if args.smoke else open(args.runs, "a")

    async def replay(ids, gold, final, seed0):
        res = []
        for k in range(args.k):
            p = await srv.complete(ids, args.answer_max, seed0 + k)
            ans = clean(p["text"])
            box = final_box(ans)
            res.append({"answer": ans, "n": p["n"], "finish": p["finish"], "correct": graded(ans, gold),
                        "ok": graded(ans, gold) or (box is not None and same(box, final))})
        return res

    async def one(key, meta):
        t = traces[key[:2]]
        st = _state(meta, hard_ok.get(key, False))
        c = S[st]
        c["cut points"] += 1
        prefix, rest = t["thinking"][: meta["prefix_chars"]], t["thinking"][meta["prefix_chars"]:]
        if len(prefix) + len(rest) > args.max_chars:
            c["too long to ask"] += 1
            return
        units = split_units(rest)
        final = final_box(t["answer"])
        numbered = "\n\n".join(f"[{i + 1}] {u.strip()}" for i, u in enumerate(units))
        r = await srv.chat_full([{"role": "user", "content": ASK.format(problem=t["prompt"], prefix=prefix,
                                                                       final=final, rest=numbered)}],
                                args.judge_max_tokens, args.judge_kwargs, seed=rng.randint(0, 10**6))
        c["judge tokens"] += r["n"]
        keep = parse_keep(r["content"], len(units))
        ex = {"key": key, "state": st, "meta": meta, "t": t, "units": units, "keep": keep, "final": final,
              "judge": r, "verdict": None}
        if keep is None:
            c["no list (incl. judge ran out of tokens)"] += 1
            ex["verdict"] = "no list"
        elif not keep:
            c["kept nothing"] += 1
            ex["verdict"] = "kept nothing (covered by the immediate close)"
        else:
            tail = "\n\n".join(units[k - 1].strip() for k in keep)
            n_tail = len(await srv.tok(tail))
            ex.update(tail=tail, n_tail=n_tail)
            if n_tail > args.max_tail:
                c["tail over --max-tail"] += 1
                ex["verdict"] = "tail too long"
            else:
                base_ids, think_ids = await prepare(srv, t, args, think_open)
                head = base_ids + think_ids[: key[2]] + notice
                plain = await replay(head + await srv.tok(tail + "\n" + CLOSE + "\n\n"), t["gold"], final, 7000)
                masked_tail, n_masked = mask_answer(tail, final)
                masked = await replay(head + await srv.tok(masked_tail + "\n" + CLOSE + "\n\n"), t["gold"], final, 8000)
                ok_plain, ok_masked = all(x["ok"] for x in plain), all(x["ok"] for x in masked)
                accept = ok_plain and ok_masked
                ex.update(plain=plain, masked=masked, n_masked=n_masked, masked_tail=masked_tail)
                c["replayed"] += 1
                c["replay passes"] += ok_plain
                c["masked replay passes"] += ok_masked
                c["accepted"] += accept
                c["answer was in the tail"] += n_masked > 0
                if accept:
                    c["tok before"] += meta["L"] - key[2]
                    c["tok after"] += n_tail
                ex["verdict"] = ("ACCEPT" if accept else "reject: replay failed" if not ok_plain
                                 else "reject: only the conclusion was kept (fails once the answer is hidden)")
                if fout and accept:
                    async with lock:
                        for k, x in enumerate(plain):
                            fout.write(json.dumps(dict(
                                {f: meta[f] for f in ("id", "sample", "source", "gold", "L", "u", "frac", "prefix_chars",
                                                      "uncut_correct", "uncut_finished", "uncut_answer_tok",
                                                      "prefix_has_final_box")},
                                key=f"{key[0]}|{key[1]}|{key[2]}|prune|{k}", cond="prune", kind="prune", grace=n_tail,
                                k=k, think_after=tail, think_after_tok=n_tail, self_term=True, answer=x["answer"],
                                answer_tok=x["n"], answer_finish=x["finish"], correct=x["correct"],
                                kept=keep, n_units=len(units), masked_pass=True), ensure_ascii=False) + "\n")
                        fout.flush()
        if args.smoke:
            examples.append(ex)

    await asyncio.gather(*[one(k, m) for k, m in todo])
    if fout:
        fout.close()
    await srv.close()

    rows = ["cut points", "too long to ask", "no list (incl. judge ran out of tokens)", "kept nothing",
            "tail over --max-tail", "replayed", "replay passes", "masked replay passes", "accepted", "answer was in the tail"]
    states = [x for x in ("mid-way", "has conclusion") if x in S]
    print(f"\n{'':<42}" + "".join(f"{x:>16}" for x in states))
    for row in rows:
        print(f"{row:<42}" + "".join(f"{S[x][row]:>16}" for x in states))
    for x in states:
        c = S[x]
        if c["tok before"]:
            print(f"{x}: accepted tails keep {c['tok after']} of {c['tok before']} thinking tokens after the cut "
                  f"({1 - c['tok after'] / c['tok before']:.0%} removed)")
        print(f"{x}: judge used {c['judge tokens'] / max(c['cut points'], 1):.0f} tokens per cut point")
    print("""
replay passes          prefix + [cut] + kept paragraphs: every replay gives the right answer
masked replay passes   same, with the final answer hidden in the kept paragraphs: the kept STEPS
                       must lead to it (a tail that only states the conclusion fails here)
accepted               both passed; only these become SFT candidates
mid-way                the immediate close failed at this cut point, so steps are really missing""")
    if args.smoke:
        write_samples(examples, args.smoke_out)
        print(f"\nexamples to read: {args.smoke_out}")
    else:
        print(f"appended to {args.runs}. Next: python make_sft.py {args.runs} {args.traces}")


def _clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n // 2] + "\n[...]\n" + s[-n // 2:]


def write_samples(examples, path):
    order = {"ACCEPT": 0}
    examples.sort(key=lambda e: (e["state"] != "mid-way", order.get((e["verdict"] or "")[:6], 1), str(e["key"])))
    out = ["# prune_tail smoke: examples to read\n",
           "Check for each ACCEPT: are the kept paragraphs real steps toward the answer, "
           "or only a restated conclusion? Does the kept tail read as a sensible finish?\n"]
    for n, e in enumerate(examples, 1):
        m, t = e["meta"], e["t"]
        out.append(f"\n---\n\n## {n}. {e['verdict']}\n")
        out.append(f"- trace `{e['key'][0]}` sample {e['key'][1]} | cut at {e['key'][2]} of {m['L']} thinking tokens "
                   f"({m['frac']:.0%}) | state: **{e['state']}** | final answer `{e['final']}` (gold `{t['gold']}`)")
        if e.get("keep") is not None:
            out.append(f"- kept {len(e['keep'])} of {len(e['units'])} paragraphs: {e['keep']}"
                       + (f" | tail {e['n_tail']} tokens vs {m['L'] - e['key'][2]} originally" if e.get("n_tail") else ""))
        if e.get("plain"):
            fb = lambda x: f"`{final_box(x['answer'])}` {'✓' if x['ok'] else '✗'}"
            out.append(f"- replay: {', '.join(fb(x) for x in e['plain'])} | answer hidden "
                       f"({e['n_masked']} places): {', '.join(fb(x) for x in e['masked'])}")
        out.append(f"- judge: {e['judge']['n']} tokens, finish `{e['judge']['finish']}`\n")
        out.append("**Problem**\n\n```\n" + _clip(t["prompt"], 700) + "\n```\n")
        out.append("**End of the thinking before the cut**\n\n```\n" + _clip(t["thinking"][: m["prefix_chars"]][-900:], 900) + "\n```\n")
        out.append("**Paragraphs after the cut** (KEEP / drop)\n")
        keep = set(e.get("keep") or [])
        for i, u in enumerate(e["units"], 1):
            out.append(f"- [{i}] **{'KEEP' if i in keep else 'drop'}** · " + _clip(u, 360).replace("\n", " ⏎ "))
        if e["judge"]["reasoning"]:
            out.append("\n<details><summary>judge thinking</summary>\n\n```\n" + _clip(e["judge"]["reasoning"], 2500) + "\n```\n</details>\n")
        out.append("\n**Judge reply:** `" + _clip(e["judge"]["content"], 300).replace("\n", " ") + "`\n")
    open(path, "w").write("\n".join(out))


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
    ap.add_argument("--include-concluded", action="store_true",
                    help="also prune cut points that already had the conclusion (by default they are skipped: "
                         "the immediate close already gives them the shortest possible target, 0 thinking)")
    ap.add_argument("--max-cuts", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=90000, help="skip when prefix + rest is longer than this")
    ap.add_argument("--answer-max", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--template-kwargs", type=json.loads, default={"enable_thinking": True})
    ap.add_argument("--judge-kwargs", type=json.loads, default={"enable_thinking": True},
                    help="chat template settings for the pruning question (thinking ON by default: this is offline)")
    ap.add_argument("--judge-max-tokens", type=int, default=12000, help="the judge's own thinking + reply")
    ap.add_argument("--system", default=None)
    ap.add_argument("--smoke", type=int, nargs="?", const=50, default=0,
                    help="prune N cut points (default 50, mostly mid-way), save nothing, write examples to read")
    ap.add_argument("--smoke-out", default="prune_smoke.md")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
