"""Settled-detector bench: can a judge tell, from the thinking so far, that a request can be cut for free?

Three judges are scored on the same thinking traces at the same cut points, against what a real silent cut
does to the answer:
  jev:   Open-Jev reads the prompt head and the tail of the thinking so far (prefill only, its own GPU) and
         answers "is the final answer settled?" (a score question) and "same answer if stopped now?" (noul)
  deer:  the serving model's own confidence in a trial answer (DEER-style): close the thinking, prefill the
         final-answer prefix, and take the mean token probability of the answer it writes (one vLLM call)
  probe: the gateway's needed-budget label: does that trial answer match the answer at the full thinking
The ground truth per row is a silent cut at that point followed by a full answer (1024 tokens), graded.

  python -m math_agent.settledbench --base-url http://<vllm>:8000/v1 --jev-url http://127.0.0.1:8791/v1/systemone \
      --n 32 --seed 0 --out runs/settledbench

For each sampled problem (same draws as a sweep with that seed): plan, the solve prompt as the agent renders it,
the uncut thinking (greedy, budget), then at every cut point (bucket edges below the length plus fixed fractions
of the length) the three judges and the silent cut. Writes rows.jsonl and summary.json. The summary ranks the
judges (AUC against "the cut kept the answer right") and simulates an early-exit policy per judge: exit at the
first cut point whose signal clears a threshold, and report accuracy against thinking tokens, with the oracle
(earliest cut point that keeps the answer right) as the ceiling.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

import httpx

from . import prompts
from .common import add_common_args, examples_from
from .grading import extract, grade
from .llm import EARLY_STOP, FINAL_PREFIX, render_chat

EDGES = (256, 512, 1024, 2048, 4096, 8192)
FRACS = (0.25, 0.5, 0.75, 0.9)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


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

    async def stream(self, prompt: str, max_tokens: int, stop: list[str]) -> tuple[str, int, str | None, list[tuple[int, int]]]:
        """Greedy streamed completion: text, tokens, finish reason, and (tokens, chars) after every event."""
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stop": stop,
                "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                "skip_special_tokens": False}
        parts: list[str] = []
        n, nchars, finish, trail = 0, 0, None, []
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
                    trail.append((n, nchars))
        self.completion_tokens += n
        return "".join(parts), n, finish, trail

    async def trial(self, prompt: str, max_tokens: int) -> tuple[str, float | None]:
        """A final-answer-only completion with token logprobs: text and the DEER confidence (mean token
        probability over the answer tokens up to the closing brace)."""
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                "stop": ["<|im_end|>"], "stream": False, "logprobs": 1, "skip_special_tokens": False}
        async with self.sem:
            self.calls += 1
            r = await self.http.post(f"{self.base}/completions", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]!r}")
        ch = r.json()["choices"][0]
        text = ch.get("text") or ""
        lp = (ch.get("logprobs") or {})
        toks, lps = lp.get("tokens") or [], lp.get("token_logprobs") or []
        self.completion_tokens += len(toks)
        return text, deer_confidence(toks, lps)


def deer_confidence(tokens: list[str], logprobs: list[float | None]) -> float | None:
    """Mean probability of the answer tokens, stopping after the token that closes the boxed answer."""
    ps = []
    for t, lp in zip(tokens, logprobs):
        if lp is None:
            continue
        ps.append(math.exp(lp))
        if "}" in t:
            break
    return sum(ps) / len(ps) if ps else None


def chars_at(trail: list[tuple[int, int]], tokens: int) -> int | None:
    """Characters of thinking streamed when the token count first reached `tokens`."""
    for n, c in trail:
        if n >= tokens:
            return c
    return None


def cut_points(used: int, edges=EDGES, fracs=FRACS, min_gap: float = 0.08) -> list[int]:
    """Bucket edges below the length plus fixed fractions of it, deduplicated when within min_gap of the length."""
    cands = sorted({e for e in edges if e < used} | {int(used * f) for f in fracs})
    out: list[int] = []
    for c in cands:
        if c < 32 or c >= used:
            continue
        if out and (c - out[-1]) < min_gap * used:
            continue
        out.append(c)
    return out


async def one(cl: Client, jev, ex, budget: int, answer_tokens: int, final_tokens: int) -> list[dict]:
    plan_prefix = render_chat(prompts.plan_messages(ex.question)) + "<think>\n\n</think>\n\n"
    plan, _, _, _ = await cl.stream(plan_prefix, 128, ["<|im_end|>"])
    solve = render_chat(prompts.solve_messages(ex.question, plan, [])) + "<think>\n"
    thinking, used, finish, trail = await cl.stream(solve, budget, ["</think>"])
    natural = finish == "stop"

    async def probe(th: str) -> tuple[str | None, float | None]:
        text, conf = await cl.trial(solve + th + EARLY_STOP + "</think>\n\n" + FINAL_PREFIX, final_tokens)
        return extract(FINAL_PREFIX + text, ex.dataset), conf

    async def answer(th: str, exhausted: bool) -> tuple[str | None, int]:
        text, n, _, _ = await cl.stream(solve + th + (EARLY_STOP if exhausted else "") + "</think>\n\n",
                                        answer_tokens, ["<|im_end|>"])
        return extract(text, ex.dataset), n

    async def judge(th: str, n: int) -> dict | None:
        if jev is None:
            return None
        for _ in range(20):  # the judge skips when busy; wait for a slot
            out = await jev.ask_settled(solve, th, n)
            if out is not None or jev.last_error:
                return out
            await asyncio.sleep(0.05)
        return None

    (ref, ref_conf), (uncut_pred, uncut_tokens), full_judge = await asyncio.gather(
        probe(thinking), answer(thinking, not natural), judge(thinking, used))
    base = {"id": ex.id, "level": ex.level, "dataset": ex.dataset, "gold": ex.gold, "used": used, "natural": natural,
            "ref": ref, "ref_correct": grade(ref, ex.gold, ex.dataset), "uncut_pred": uncut_pred,
            "uncut_correct": grade(uncut_pred, ex.gold, ex.dataset), "uncut_answer_tokens": uncut_tokens,
            "deer_full": ref_conf, "jev_full": full_judge}
    rows = []
    for c in cut_points(used):
        at = chars_at(trail, c)
        if at is None:
            continue
        th = thinking[:at]
        row = {**base, "cut": c, "frac": round(c / used, 3)}
        (p, conf), (sp, st), j = await asyncio.gather(probe(th), answer(th, True), judge(th, c))
        row.update({"probe": p, "probe_match": p == ref, "deer": conf, "silent_pred": sp,
                    "silent_answer_tokens": st, "silent_correct": grade(sp, ex.gold, ex.dataset),
                    "silent_same": sp == uncut_pred,
                    "jev_settled": None if j is None else j.get("settled"),
                    "jev_done": None if j is None else j.get("done"),
                    "jev_level": None if j is None else j.get("level"),
                    "jev_ms": None if j is None else j.get("ms")})
        rows.append(row)
    return rows


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based AUC (probability a positive outranks a negative; ties count half)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


SIGNALS = ("jev_settled", "jev_done", "deer", "probe_match")


def signal(row: dict, name: str) -> float | None:
    v = row.get(name)
    if v is None:
        return None
    return float(v)


def early_exit(rows: list[dict], name: str, tau: float) -> dict:
    """Per problem: exit at the first cut point whose signal clears tau, else run uncut."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    correct, tokens, exits, n = 0, 0, 0, 0
    for pid, rs in by.items():
        rs = sorted(rs, key=lambda r: r["cut"])
        n += 1
        hit = next((r for r in rs if (signal(r, name) or 0.0) >= tau), None)
        if hit is not None:
            exits += 1
            correct += hit["silent_correct"]
            tokens += hit["cut"]
        else:
            correct += rs[0]["uncut_correct"]
            tokens += rs[0]["used"]
    return {"accuracy": round(correct / n, 4), "mean_thinking": round(tokens / n, 1), "exited": exits, "n": n}


def oracle_exit(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    correct, tokens, exits = 0, 0, 0
    for rs in by.values():
        rs = sorted(rs, key=lambda r: r["cut"])
        hit = next((r for r in rs if r["silent_correct"]), None)
        if hit is not None:
            exits += 1
            correct += 1
            tokens += hit["cut"]
        else:
            correct += rs[0]["uncut_correct"]
            tokens += rs[0]["used"]
    return {"accuracy": round(correct / len(by), 4), "mean_thinking": round(tokens / len(by), 1), "exited": exits,
            "n": len(by)}


def summarize(rows: list[dict]) -> dict:
    out: dict = {"rows": len(rows), "problems": len({r["id"] for r in rows})}
    if not rows:
        return out
    ids = {r["id"]: r for r in rows}
    out["uncut_accuracy"] = round(sum(r["uncut_correct"] for r in ids.values()) / len(ids), 4)
    out["mean_used"] = round(sum(r["used"] for r in ids.values()) / len(ids), 1)
    out["natural"] = sum(r["natural"] for r in ids.values())
    ok = [r for r in rows if r["uncut_correct"]]
    out["rows_uncut_correct"] = len(ok)
    out["silent_kept_right"] = round(sum(r["silent_correct"] for r in ok) / len(ok), 4) if ok else None
    out["auc"] = {}
    for name in SIGNALS:
        have = [r for r in ok if signal(r, name) is not None]
        out["auc"][name] = {"n": len(have),
                            "auc": None if not have else auc([signal(r, name) for r in have],
                                                             [bool(r["silent_correct"]) for r in have])}
    depth = {"frac<=0.25": lambda r: r["frac"] <= 0.25, "0.25<frac<=0.5": lambda r: 0.25 < r["frac"] <= 0.5,
             "0.5<frac<=0.75": lambda r: 0.5 < r["frac"] <= 0.75, "frac>0.75": lambda r: r["frac"] > 0.75}
    out["by_depth"] = {}
    for g, sel in depth.items():
        gs = [r for r in ok if sel(r)]
        if not gs:
            continue
        d = {"n": len(gs), "silent_kept_right": round(sum(r["silent_correct"] for r in gs) / len(gs), 3)}
        for name in SIGNALS:
            have = [r for r in gs if signal(r, name) is not None]
            d[f"{name}_mean"] = round(sum(signal(r, name) for r in have) / len(have), 3) if have else None
        out["by_depth"][g] = d
    out["early_exit"] = {"none": {"accuracy": out["uncut_accuracy"], "mean_thinking": out["mean_used"], "exited": 0,
                                  "n": len(ids)},
                         "oracle": oracle_exit(rows)}
    for name in SIGNALS:
        for tau in THRESHOLDS:
            out["early_exit"][f"{name}>={tau}"] = early_exit(rows, name, tau)
    jm = [r["jev_ms"] for r in rows if r.get("jev_ms") is not None]
    out["jev_ms_mean"] = round(sum(jm) / len(jm), 1) if jm else None
    out["jev_missing"] = sum(1 for r in rows if r.get("jev_settled") is None)
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
    jev = None
    if args.jev_url:
        from budget_classifier import Buckets, JevPrior
        jev = JevPrior(Buckets(), args.jev_url, timeout=args.jev_timeout, max_inflight=args.jev_concurrency)
    out = Path(args.out or f"runs/settledbench-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    done = 0

    async def run(ex):
        nonlocal done
        try:
            r = await one(cl, jev, ex, args.budget, args.answer_tokens, args.final_tokens)
        except Exception as e:  # keep going; the row count tells
            r = [{"id": ex.id, "error": f"{type(e).__name__}: {e}"}]
        rows.extend(r)
        done += 1
        print(f"{done:>3}/{len(examples)} {ex.id} rows={len(r)} calls={cl.calls} tokens={cl.completion_tokens}",
              flush=True)

    await asyncio.gather(*(run(ex) for ex in examples))
    await cl.http.aclose()
    if jev is not None:
        await jev.aclose()
    good = [r for r in rows if "error" not in r]
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {**summarize(good), "errors": len(rows) - len(good), "calls": cl.calls,
               "completion_tokens": cl.completion_tokens, "wall_s": round(time.time() - t0, 1),
               "budget": args.budget, "dataset": args.dataset, "seed": args.seed, "model": cl.model,
               "jev": None if jev is None else {"calls": jev.calls, "failures": jev.failures,
                                                "last_error": jev.last_error}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", out)


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n", type=int, default=32, help="draws from the dataset, as loadgen does (duplicates dropped)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--jev-url", default=None, help="Open-Jev endpoint(s); omit to skip the Jev judge")
    p.add_argument("--jev-timeout", type=float, default=10.0)
    p.add_argument("--jev-concurrency", type=int, default=8)
    p.add_argument("--answer-tokens", type=int, default=1024)
    p.add_argument("--final-tokens", type=int, default=48)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
