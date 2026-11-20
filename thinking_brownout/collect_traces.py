#!/usr/bin/env python3
"""Collect UNCUT thinking traces from a vLLM OpenAI-compatible server.

One run = one model. Start `vllm serve <model>` first, then run this script.
Output: traces/<model>.jsonl, one line per (problem, sample). Safe to re-run:
finished (problem, sample) pairs are skipped.

  python collect_traces.py --model Qwen/Qwen3.5-9B --smoke      # 1 request, print raw reply
  python collect_traces.py --model Qwen/Qwen3.5-9B --limit 20   # small trial
  python collect_traces.py --model Qwen/Qwen3.5-9B              # full run
"""
import argparse
import asyncio
import json
import os
import random
import re
import time

import httpx

# Per-model request settings. Sampling params are NOT set here on purpose:
# vLLM then uses each model's own generation_config.json defaults.
MODEL_CFG = {
    # thinking on; reasoning_effort left at the model default ("xhigh")
    "Qwen/Qwen3.8-27B": {"chat_template_kwargs": {"enable_thinking": True}},
    "Qwen/Qwen3.5-9B": {"chat_template_kwargs": {"enable_thinking": True}},
    "Qwen/Qwen3.5-4B": {"chat_template_kwargs": {"enable_thinking": True}},
    # Muse Glimmer sets its effort with a line in the SYSTEM prompt (low / medium / high / xhigh).
    # "high" is the level its own tooling uses by default. Needs --reasoning-parser muse_glimmer.
    "meta-models/Muse-Glimmer-30B": {"system": "Reasoning strength: high"},
}

SUFFIX = "\n\nPut your final answer in \\boxed{}."


# ---------------------------------------------------------------- prompts
def load_prompts(args):
    """Return a list of {id, source, level, prompt, gold}."""
    if args.prompts_file:
        with open(args.prompts_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    from datasets import load_dataset

    rng = random.Random(0)  # fixed: every model gets the same problems
    items = []

    def take(rows, n):
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:n]

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [(i, r) for i, r in enumerate(ds)]
    for i, r in take(rows, args.n_gsm8k):
        items.append({"id": f"gsm8k-{i}", "source": "gsm8k", "level": None,
                      "prompt": r["question"], "gold": r["answer"].split("####")[-1].strip()})

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [(i, r) for i, r in enumerate(ds)]
    for i, r in take(rows, args.n_math500):
        items.append({"id": f"math500-{i}", "source": "math500", "level": r.get("level"),
                      "prompt": r["problem"], "gold": str(r["answer"])})

    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    idx = take(range(len(ds)), args.n_deepscaler)
    for i in idx:
        r = ds[i]
        items.append({"id": f"deepscaler-{i}", "source": "deepscaler", "level": None,
                      "prompt": r["problem"], "gold": str(r["answer"])})
    return items


# ---------------------------------------------------------------- scoring (crude)
def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{")
    depth, j = 1, i
    while j < len(text) and depth > 0:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def norm(s):
    s = str(s).strip().rstrip(".")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\$", ""), ("$", ""), ("^\\circ", ""), ("\\%", ""),
                 ("%", ""), (" ", "")]:
        s = s.replace(a, b)
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def is_correct(answer_text, gold):
    """Last \\boxed{} vs gold, with the shared math-aware grader (analyze_boxed.same)."""
    pred = last_boxed(answer_text or "")
    if pred is None:
        return False
    try:
        from analyze_boxed import same
        return same(pred, gold)
    except ImportError:
        pass
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


# ---------------------------------------------------------------- one request
def split_thinking(msg, think_open, think_close):
    """Return (thinking, answer). Uses the server's reasoning parser if it ran,
    otherwise splits the raw text on the thinking end tag."""
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        return reasoning, content
    if think_close in content:
        thinking, answer = content.split(think_close, 1)
        return thinking.replace(think_open, "", 1).strip(), answer.strip()
    if think_open in content:  # thinking started but never closed (hit max_tokens)
        return content.replace(think_open, "", 1).strip(), ""
    return "", content


async def count_tokens(client, model, text):
    if not text:
        return 0
    try:
        r = await client.post("/tokenize", json={"model": model, "prompt": text,
                                                 "add_special_tokens": False})
        r.raise_for_status()
        return r.json()["count"]
    except Exception:
        return None


async def chat(client, args, cfg, item, sample):
    messages = [{"role": "user", "content": item["prompt"] + args.suffix}]
    if cfg.get("system"):
        messages.insert(0, {"role": "system", "content": cfg["system"]})
    body = {"model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "seed": args.seed + sample,
            **{k: v for k, v in cfg.items() if k != "system"}}
    for attempt in range(3):
        try:
            r = await client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                print(f"[fail] {item['id']} s{sample}: {type(e).__name__} {e}")
                return None
            await asyncio.sleep(5 * (attempt + 1))


async def run_one(client, sem, args, cfg, item, sample, out, lock, stats):
    async with sem:
        t0 = time.time()
        resp = await chat(client, args, cfg, item, sample)
        if resp is None:
            return
        choice = resp["choices"][0]
        thinking, answer = split_thinking(choice["message"], args.think_open, args.think_close)
        usage = resp.get("usage") or {}
        rec = {
            "id": item["id"], "source": item["source"], "level": item.get("level"),
            "sample": sample, "model": args.model, "cfg": cfg, "suffix": args.suffix,
            "prompt": item["prompt"], "gold": item["gold"],
            "thinking": thinking, "answer": answer,
            "think_tokens": await count_tokens(client, args.model, thinking),
            "answer_tokens": await count_tokens(client, args.model, answer),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "latency_s": round(time.time() - t0, 2),
        }
        # finished = thinking ended by itself and an answer followed (a usable label)
        rec["finished"] = bool(thinking) and bool(answer) and rec["finish_reason"] == "stop"
        rec["correct"] = is_correct(answer, item["gold"])
        async with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            stats["n"] += 1
            if stats["n"] % 50 == 0:
                rate = stats["n"] / (time.time() - stats["t0"])
                left = (stats["total"] - stats["n"]) / max(rate, 1e-9) / 60
                print(f"{stats['n']}/{stats['total']} done, ~{left:.0f} min left")


# ---------------------------------------------------------------- main
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out-dir", default="traces")
    ap.add_argument("--samples", type=int, default=4, help="samples per problem")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-gsm8k", type=int, default=300)
    ap.add_argument("--n-math500", type=int, default=500)
    ap.add_argument("--n-deepscaler", type=int, default=500)
    ap.add_argument("--skip-ids-from", nargs="+", default=None,
                    help="a trace file; problems whose id is already in it are skipped (use it to collect "
                         "only NEW problems, e.g. more DeepScaleR for training data)")
    ap.add_argument("--prompts-file", help="optional JSONL with id, source, prompt, gold")
    ap.add_argument("--limit", type=int, help="only the first N problems (trial run)")
    ap.add_argument("--think-open", default="<think>")
    ap.add_argument("--think-close", default="</think>")
    ap.add_argument("--suffix", default=SUFFIX,
                    help="text added after every problem (control experiments: change the answer-format instruction)")
    ap.add_argument("--tag", default="", help="added to the output file name, e.g. --tag plainbox")
    ap.add_argument("--only-source", help="only problems from this dataset, e.g. gsm8k")
    ap.add_argument("--system", help="override the system prompt for this model (e.g. 'Reasoning strength: xhigh')")
    ap.add_argument("--smoke", action="store_true", help="send 1 request, print raw reply, exit")
    args = ap.parse_args()

    cfg = dict(MODEL_CFG.get(args.model, {}))
    if args.system:
        cfg["system"] = args.system
    args.suffix = args.suffix.replace("\\n", "\n")   # allow \n typed on the command line
    items = load_prompts(args)
    if args.only_source:
        items = [it for it in items if it["source"] == args.only_source]
    if args.skip_ids_from:
        seen = set()
        for path in args.skip_ids_from:
            with open(path) as f:
                for line in f:
                    try:
                        seen.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        before = len(items)
        items = [it for it in items if it["id"] not in seen]
        print(f"--skip-ids-from: {before - len(items)} problems already collected, {len(items)} new")
    if args.limit:
        random.Random(1).shuffle(items)  # keep a mix of sources in a trial run
        items = items[:args.limit]

    limits = httpx.Limits(max_connections=args.concurrency + 16)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=None, limits=limits) as client:
        if args.smoke:
            resp = await chat(client, args, cfg, items[0], 0)
            msg = resp["choices"][0]["message"]
            print("message keys:", [k for k, v in msg.items() if v])
            print("raw content (first 600 chars):\n", (msg.get("content") or "")[:600])
            thinking, answer = split_thinking(msg, args.think_open, args.think_close)
            print("\nparsed thinking tokens:", await count_tokens(client, args.model, thinking))
            print("parsed answer (first 300 chars):", answer[:300])
            print("usage:", resp.get("usage"), "finish:", resp["choices"][0].get("finish_reason"))
            if not thinking:
                print("\n!! No thinking found. Check the reasoning parser / --think-close tag / MODEL_CFG.")
            return

        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, args.model.replace("/", "__") + ("." + args.tag if args.tag else "") + ".jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:      # half-written last line from a killed run
                        continue
                    done.add((r["id"], r["sample"]))
        todo = [(it, s) for it in items for s in range(args.samples) if (it["id"], s) not in done]
        random.Random(2).shuffle(todo)  # mix short and long requests in the batch
        print(f"{len(items)} problems x {args.samples} samples; {len(done)} already done, {len(todo)} to run -> {path}")

        sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
        stats = {"n": 0, "total": len(todo), "t0": time.time()}
        with open(path, "a") as out:
            await asyncio.gather(*[run_one(client, sem, args, cfg, it, s, out, lock, stats)
                                   for it, s in todo])
        print(f"done: wrote {stats['n']} traces in {(time.time() - stats['t0']) / 60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
