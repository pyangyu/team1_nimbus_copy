"""Label fidelity bench: does a probe label predict what a real cut does to the answer?

The gateway's needed-budget probe closes the thinking at a bucket edge, appends the early-stop line and
asks for the final answer only (48 tokens). The serving path is different: a silent cut lets the agent
write a full answer (1024 tokens) after the early-stop line, and an announced cut inserts a notice,
lets the model think N more tokens, then writes the full answer. This bench runs all of them on the same
thinking traces and grades them, so the probe label can be checked against the real outcomes.

  python -m math_agent.labelbench --base-url http://<vllm>:8000/v1 --n 32 --seed 0 --out runs/labelbench

For each sampled problem (same sampling as loadgen, so the same problems as a sweep with that seed):
  1. plan (no thinking), then the solve prompt exactly as the agent renders it
  2. uncut thinking with the budget, greedy, recording where each bucket edge falls in the text
  3. at the full thinking and at every edge below its length:
       probe:      thinking[:e] + early stop + </think> + "The final answer is \\boxed{" (48 tokens)
       silent:     thinking[:e] + early stop + </think> + full answer
       announced:  thinking[:e] + notice(N) + up to N more thinking + </think> + full answer
  4. grade every answer against gold
Writes rows.jsonl (one row per problem and edge) and summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

from . import prompts
from .common import add_common_args, examples_from
from .grading import extract, grade
from .llm import EARLY_STOP, FINAL_PREFIX, NOTICES, render_chat

EDGES = (32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
GATEWAY_NOTICE = "\n\n<budget_cut left={N}>\n"
NOTICE_TEXTS = {"gateway": GATEWAY_NOTICE, "count": NOTICES["count"]}


class Client:
    def __init__(self, base_url: str, api_key: str, model: str | None, concurrency: int):
        self.base = base_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=900.0, headers={"Authorization": f"Bearer {api_key}"},
                                      limits=httpx.Limits(max_connections=200))
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.completion_tokens = 0

    async def resolve(self) -> None:
        if not self.model:
            r = await self.http.get(f"{self.base}/models")
            r.raise_for_status()
            self.model = r.json()["data"][0]["id"]

    async def complete(self, prompt: str, max_tokens: int, stop: list[str], marks_at: tuple[int, ...] = ()
                       ) -> tuple[str, int, str | None, dict[int, int]]:
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stop": stop,
                "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                "skip_special_tokens": False}
        parts: list[str] = []
        n, nchars, finish, marks = 0, 0, None, {}
        async with self.sem:
            self.calls += 1
            async with self.http.stream("POST", f"{self.base}/completions", json=body) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"{r.status_code}: {(await r.aread())[:300]!r}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    for ch in ev.get("choices") or []:
                        if ch.get("text"):
                            parts.append(ch["text"])
                            nchars += len(ch["text"])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                    if ev.get("usage"):
                        n = ev["usage"].get("completion_tokens", n)
                    for e in marks_at:
                        if e <= n and e not in marks:
                            marks[e] = nchars
        self.completion_tokens += n
        return "".join(parts), n, finish, marks


async def one(cl: Client, ex, budget: int, notices: dict[str, str], ns: list[int], answer_tokens: int,
              final_tokens: int) -> list[dict]:
    plan_prefix = render_chat(prompts.plan_messages(ex.question)) + "<think>\n\n</think>\n\n"
    plan, _, _, _ = await cl.complete(plan_prefix, 128, ["<|im_end|>"])
    solve = render_chat(prompts.solve_messages(ex.question, plan, [])) + "<think>\n"
    thinking, used, finish, marks = await cl.complete(solve, budget, ["</think>"], EDGES)
    natural = finish == "stop"

    async def probe(th: str) -> str | None:
        text, _, _, _ = await cl.complete(solve + th + EARLY_STOP + "</think>\n\n" + FINAL_PREFIX, final_tokens,
                                          ["<|im_end|>"])
        return extract(FINAL_PREFIX + text, ex.dataset)

    async def answer(th: str, exhausted: bool) -> tuple[str | None, int]:
        text, n, _, _ = await cl.complete(solve + th + (EARLY_STOP if exhausted else "") + "</think>\n\n",
                                          answer_tokens, ["<|im_end|>"])
        return extract(text, ex.dataset), n

    ref, (uncut_pred, uncut_tokens) = await asyncio.gather(probe(thinking), answer(thinking, not natural))
    base = {"id": ex.id, "level": ex.level, "gold": ex.gold, "used": used, "natural": natural,
            "ref": ref, "uncut_pred": uncut_pred, "uncut_correct": grade(uncut_pred, ex.gold, ex.dataset),
            "uncut_answer_tokens": uncut_tokens, "ref_correct": grade(ref, ex.gold, ex.dataset)}
    rows = []
    for e in [e for e in EDGES if e in marks and e < used]:
        th = thinking[: marks[e]]
        row = {**base, "edge": e, "frac": round(e / used, 3)}
        p, (sp, st) = await asyncio.gather(probe(th), answer(th, True))
        row.update({"probe": p, "probe_match": p == ref, "silent_pred": sp, "silent_answer_tokens": st,
                    "silent_correct": grade(sp, ex.gold, ex.dataset)})
        for name, notice in notices.items():
            for n_more in ns:
                more, n2, fin2, _ = await cl.complete(solve + th + notice.format(N=n_more), n_more, ["</think>"])
                ap, at = await answer(th + notice.format(N=n_more) + more, fin2 == "length")
                key = f"{name}{n_more}"
                row.update({f"{key}_pred": ap, f"{key}_correct": grade(ap, ex.gold, ex.dataset),
                            f"{key}_used_more": n2, f"{key}_closed": fin2 == "stop", f"{key}_answer_tokens": at})
        rows.append(row)
    return rows


def summarize(rows: list[dict], paths: list[str]) -> dict:
    out = {"rows": len(rows), "problems": len({r["id"] for r in rows})}
    groups = {"frac<=0.25": lambda r: r["frac"] <= 0.25, "0.25<frac<=0.5": lambda r: 0.25 < r["frac"] <= 0.5,
              "0.5<frac<=0.75": lambda r: 0.5 < r["frac"] <= 0.75, "frac>0.75": lambda r: r["frac"] > 0.75,
              "all": lambda r: True}
    for gname, sel in groups.items():
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        s = {"n": len(g), "uncut_correct": sum(r["uncut_correct"] for r in g) / len(g),
             "probe_match": sum(r["probe_match"] for r in g) / len(g)}
        for p in paths:
            s[f"{p}_correct"] = sum(r[f"{p}_correct"] for r in g) / len(g)
            # among problems the uncut path got right: does the probe label predict survival on this path?
            ok = [r for r in g if r["uncut_correct"]]
            m = [r for r in ok if r["probe_match"]]
            mm = [r for r in ok if not r["probe_match"]]
            s[f"{p}_kept_given_match"] = round(sum(r[f"{p}_correct"] for r in m) / len(m), 3) if m else None
            s[f"{p}_kept_given_mismatch"] = round(sum(r[f"{p}_correct"] for r in mm) / len(mm), 3) if mm else None
            s[f"{p}_n_match"], s[f"{p}_n_mismatch"] = len(m), len(mm)
        out[gname] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    return out


async def main(args) -> None:
    rng = random.Random(args.seed)
    pool = examples_from(args)
    chosen = [rng.choice(pool) for _ in range(args.n)]
    seen, examples = set(), []
    for ex in chosen:
        if ex.id not in seen:
            seen.add(ex.id)
            examples.append(ex)
    cl = Client(args.base_url, args.api_key, args.model, args.concurrency)
    await cl.resolve()
    notices = {k: NOTICE_TEXTS[k] for k in args.notices.split(",") if k}
    ns = [int(x) for x in args.more.split(",") if x]
    out = Path(args.out or f"runs/labelbench-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    done = 0

    async def run(ex):
        nonlocal done
        try:
            r = await one(cl, ex, args.budget, notices, ns, args.answer_tokens, args.final_tokens)
        except Exception as e:  # keep going; the row count tells
            r = [{"id": ex.id, "error": f"{type(e).__name__}: {e}"}]
        rows.extend(r)
        done += 1
        print(f"{done:>3}/{len(examples)} {ex.id} rows={len(r)} calls={cl.calls} tokens={cl.completion_tokens}",
              flush=True)

    await asyncio.gather(*(run(ex) for ex in examples))
    await cl.http.aclose()
    good = [r for r in rows if "error" not in r]
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    paths = ["silent"] + [f"{k}{n}" for k in notices for n in ns]
    summary = {**summarize(good, paths), "errors": len(rows) - len(good), "calls": cl.calls,
               "completion_tokens": cl.completion_tokens, "wall_s": round(time.time() - t0, 1),
               "budget": args.budget, "notices": list(notices), "more": ns, "model": cl.model}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", out)


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n", type=int, default=32, help="draws from the dataset, as loadgen does (duplicates dropped)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--notices", default="gateway,count", help="announced-cut notices to test")
    p.add_argument("--more", default="64,256", help="thinking tokens allowed after the notice")
    p.add_argument("--answer-tokens", type=int, default=1024)
    p.add_argument("--final-tokens", type=int, default=48)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
