#!/usr/bin/env python3
"""Thinking-length classifier, final version: the served model rates its own request.

For each request we ask the SAME model that will serve it one question, with thinking off:

    "Rate how much reasoning this problem needs on a 0-9 scale: 0 = the answer is immediate;
     1 = one short calculation; ... 9 = a very long solution with many dead ends.
     Do not solve it. Reply with the digit only."   (all ten levels are described)

The output is restricted to the ten digit tokens 0-9 (allowed_token_ids), so it costs exactly
one token and every digit gets a real probability. The score is the expected level:

    score = sum_k  k * P(k)        (0 = immediate ... 9 = very long)

Requests are ranked by score, highest first. No training and no head.
The rating request starts with exactly the text of the real request, so vLLM's prefix cache
shares the prompt's KV between the rating and the real generation.

Use it from the controller:

    scorer = EffortScorer("http://localhost:8000", "Qwen/Qwen3.8-27B")
    s, probs = await scorer.score(problem_text)          # one request
    order = rank({req_id: s, ...})                        # highest score first

Check it on collected traces (scores every problem once, cached, then reports how well
the score orders the real thinking lengths):

    python classifier.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B --smoke
    python classifier.py traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B
"""
import argparse
import asyncio
import json
import math
import os

import numpy as np
import pandas as pd

# All ten levels are described (describing only 0/3/6/9 made the model pick 3 or 6 almost every time).
EFFORT_Q = ("Rate how much reasoning this problem needs on a 0-9 scale: "
            "0 = the answer is immediate; 1 = one short calculation; 2 = two or three simple steps; "
            "3 = a few steps; 4 = several steps; 5 = many steps; 6 = a long multi-step solution; "
            "7 = a long solution with case work; 8 = a very long solution that needs search or a construction; "
            "9 = a very long solution with many dead ends. "
            "Do not solve it. Reply with the digit only.")
SUFFIX = "\n\nPut your final answer in \\boxed{}."     # the real request's own text; must match collect_traces.py
EDGES = [32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
LEVELS = np.arange(10)


# ---------------------------------------------------------------- the scorer

class EffortScorer:
    def __init__(self, base_url, model, no_think_kwargs=None, suffix=SUFFIX, concurrency=64):
        import httpx
        self.client = httpx.AsyncClient(base_url=base_url, timeout=None)
        self.model = model
        self.kwargs = no_think_kwargs or {"enable_thinking": False}
        self.suffix = suffix
        self.sem = asyncio.Semaphore(concurrency)
        self.digit_ids = None

    async def _post(self, path, body):
        for attempt in range(4):
            try:
                r = await self.client.post(path, json=body)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                resp = getattr(e, "response", None)
                if resp is not None and 400 <= resp.status_code < 500:
                    raise SystemExit(f"{path} answered {resp.status_code}:\n{resp.text[:1500]}\nSend me this output.")
                if attempt == 3:
                    raise
                await asyncio.sleep(1 + attempt)

    async def _digits(self):
        """Token id of each digit 0-9, asked from the server's tokenizer."""
        if self.digit_ids is None:
            ids = []
            for d in "0123456789":
                toks = (await self._post("/tokenize", {"model": self.model, "prompt": d,
                                                       "add_special_tokens": False}))["tokens"]
                if len(toks) != 1:
                    raise SystemExit(f"digit {d!r} is {len(toks)} tokens in this tokenizer: {toks}")
                ids.append(int(toks[0]))
            self.digit_ids = ids
        return self.digit_ids

    def messages(self, problem):
        # Same start as the real request (problem + suffix), then the question: prefix-cache friendly.
        return [{"role": "user", "content": problem + self.suffix + "\n\n---\n" + EFFORT_Q}]

    async def score(self, problem):
        """Returns (score, P(0..9)). score is the expected level, 0-9."""
        ids = await self._digits()
        body = {"model": self.model, "messages": self.messages(problem),
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "allowed_token_ids": ids,                       # the only possible outputs are 0-9
                "chat_template_kwargs": self.kwargs}            # thinking off
        async with self.sem:
            data = await self._post("/v1/chat/completions", body)
        lp = ((data["choices"][0].get("logprobs") or {}).get("content") or [None])[0]
        p = np.zeros(10)
        if lp:
            for c in lp.get("top_logprobs") or [lp]:
                s = c["token"].strip()
                if len(s) == 1 and s in "0123456789":
                    p[int(s)] += math.exp(c["logprob"])
        if p.sum() <= 0:                                        # should not happen with allowed_token_ids
            text = data["choices"][0]["message"].get("content") or ""
            if text.strip()[:1] in list("0123456789"):
                p[int(text.strip()[0])] = 1.0
            else:
                raise RuntimeError(f"no digit in the reply: {text!r}")
        p = p / p.sum()
        return float(p @ LEVELS), p

    async def close(self):
        await self.client.aclose()


def rank(scores):
    """{request_id: score} -> request ids, highest score first (cut these first under overload)."""
    return sorted(scores, key=lambda k: -scores[k])


# ---------------------------------------------------------------- check on collected traces

def tier(tokens):
    return np.minimum(np.searchsorted(EDGES, tokens, side="left"), len(EDGES) - 1)


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:            # half-written last line of a run in progress
                continue
            if r.get("finished") and r.get("think_tokens") is not None:
                L = r["think_tokens"]
            elif r.get("finish_reason") == "length":
                L = EDGES[-1]
            else:
                continue
            rows.append({"id": r["id"], "prompt": r["prompt"], "suffix": r.get("suffix"), "L": max(int(L), 1)})
    return pd.DataFrame(rows)


async def score_all(df, args):
    scorer = EffortScorer(args.base_url, args.model, json.loads(args.no_think_kwargs), concurrency=args.concurrency)
    probs = df.drop_duplicates("prompt")
    if args.smoke:
        problem = probs.iloc[0]["prompt"]
        s, p = await scorer.score(problem)
        print("token ids of 0..9:", scorer.digit_ids)
        print("problem:", problem[:200].replace("\n", " "))
        print("P(0..9):", np.round(p, 4))
        print(f"score (expected level): {s:.2f}")
        await scorer.close()
        return None
    out, done = {}, 0

    async def one(prompt):
        nonlocal done
        s, p = await scorer.score(prompt)
        out[prompt] = {"score": s, "p": [round(float(x), 6) for x in p]}
        done += 1
        if done % 200 == 0:
            print(f"  scored {done}/{len(probs)}", flush=True)

    await asyncio.gather(*[one(p) for p in probs["prompt"]])
    await scorer.close()
    return out


def report(df, scores, test_frac=0.2, seed=0):
    df = df.copy()
    df["score"] = df["prompt"].map(lambda p: scores[p]["score"])
    df["tier"] = tier(df["L"].values)
    ids = sorted(df["id"].unique())
    np.random.default_rng(seed).shuffle(ids)
    test = set(ids[: int(len(ids) * test_frac)])            # same split as the earlier predictor runs
    te = df["id"].isin(test).values

    g = df.groupby("id")["L"]                               # best possible: the other samples of the same problem
    n, tot = g.transform("count"), np.log2(df["L"]).groupby(df["id"]).transform("sum")
    others = (tot - np.log2(df["L"])) / (n - 1).clip(lower=1)
    multi = (n > 1).values

    print(f"\n{len(df)} traces, {df['prompt'].nunique()} problems")
    y = df["tier"]                                          # same target as the earlier predictor tables
    print(f"{'Spearman with the real thinking tier':<40}{'all problems':>14}{'test split':>12}")
    print(f"{'  score (no training)':<40}{spearman(df['score'], y):>14.3f}{spearman(df['score'][te], y[te]):>12.3f}")
    print(f"{'  best possible (same-problem resample)':<40}{spearman(others[multi], y[multi]):>14.3f}"
          f"{spearman(others[multi & te], y[multi & te]):>12.3f}")
    print("(the test split is the 20% of problems used in the earlier predictor tables, for comparison)")

    q = pd.qcut(df["score"].rank(method="first"), 5, labels=False)
    total = df["L"].sum()
    print(f"\n{'score group':<16}{'score range':>14}{'traces':>8}{'median think':>14}{'p90 think':>11}{'share of tokens':>17}")
    for k in range(5):
        m = q == k
        print(f"{['lowest 20%', '20-40%', '40-60%', '60-80%', 'highest 20%'][k]:<16}"
              f"{df['score'][m].min():>7.2f}-{df['score'][m].max():<6.2f}{m.sum():>8}"
              f"{df['L'][m].median():>14.0f}{df['L'][m].quantile(.9):>11.0f}{df['L'][m].sum() / total:>17.0%}")

    longest = df["L"] >= df["L"].quantile(0.9)
    flagged = df["score"] >= df["score"].quantile(0.8)
    print(f"\nthe highest-scoring 20% of requests hold {(longest & flagged).sum() / longest.sum():.0%} "
          f"of the 10% longest requests")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--no-think-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out-dir", default="predictor")
    ap.add_argument("--smoke", action="store_true", help="score one problem, print it, exit")
    args = ap.parse_args()

    df = load(args.traces)
    if df.empty:
        raise SystemExit("no finished traces in the file")
    name = os.path.basename(args.traces).replace(".jsonl", "")
    cache = os.path.join(args.out_dir, f"{name}.levels.{args.model.replace('/', '__')}.scores.json")
    if os.path.exists(cache) and not args.smoke:
        scores = json.load(open(cache))
        print(f"read cached scores from {cache}")
    else:
        scores = asyncio.run(score_all(df, args))
        if scores is None:
            return
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(scores, open(cache, "w"))
        print(f"scores saved to {cache}")
    report(df, scores)


if __name__ == "__main__":
    main()
