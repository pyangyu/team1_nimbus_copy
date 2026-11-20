#!/usr/bin/env python3
"""Shared code for the cut experiments (cut_bench.py, cut_report.py, make_sft.py).

How a cut is replayed against vLLM, with no training:
  1. build the real request (chat template + the problem), as token ids
  2. append the first u tokens of the thinking the model produced in the uncut trace
  3. append a notice (or close the thinking right away)
  4. let the model continue from there with /v1/completions
Needs analyze_boxed.py in the same folder (for grading).
"""
import asyncio
import json
import re

import numpy as np

from analyze_boxed import boxed_all, same

SUFFIX = "\n\nPut your final answer in \\boxed{}."      # must match collect_traces.py
CAP = 32768
CLOSE = "</think>"
NOTICE = ("\n\n[Server notice: the system is overloaded and your thinking budget has been cut. "
          "If you already have a conclusion, stop exploring and stop verifying. "
          "Finish your reasoning in as few tokens as possible, then give the final answer.{budget}]\n\n")
BUDGET = " You have at most {n} tokens of thinking left."

# hesitation phrases: re-checking, trying other methods, re-reading the problem, second-guessing
DITHER = re.compile(r"\b(?:wait|hmm+|alternatively|actually|on second thought|double[- ]?check(?:ing)?|"
                    r"let me (?:re-?check|verify|re-?examine|reconsider|check again|think again|re-?think|"
                    r"make sure|confirm)|but let me|let's (?:double[- ]?check|verify|re-?check|confirm)|"
                    r"(?:check|verify) (?:with|by|using) (?:an? )?(?:alternative|another|different)|"
                    r"alternative (?:method|approach|check|interpretation|way)|another (?:method|approach|way)|"
                    r"search (?:my )?memory|let'?s not (?:jump|rush)|could it be|is it possible that|"
                    r"let'?s (?:explore|consider) (?:another|other|alternative)|other interpretation|"
                    # second-guessing what the problem means (passed the old list: an 812-token target
                    # that debated "total or separately?" had 0 hits and 11 with these)
                    r"maybe|perhaps|ambigu\w*|could (?:also )?mean|might mean|interpret\w*|"
                    r"(?:another|different) reading|or (?:maybe|perhaps|separately|instead)|"
                    r"which (?:one )?is (?:expected|intended)|i lean|need (?:to )?be careful|"
                    r"but if (?:they|the)|if they (?:want|meant|mean|expect))\b",
                    re.IGNORECASE)


def dither_count(text):
    return len(DITHER.findall(text or ""))


def dither_rate(text, n_tok):
    """Hesitation phrases per 1,000 tokens."""
    return dither_count(text) * 1000 / max(n_tok, 1)


def graded(answer, gold):
    """Same grader for cut and uncut answers: the last \\boxed{} must match the gold answer."""
    boxes = boxed_all(answer or "")
    return bool(boxes) and same(boxes[-1][1], gold)


def final_box(answer):
    boxes = boxed_all(answer or "")
    return boxes[-1][1] if boxes else None


def read_traces(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("finished") and r.get("think_tokens") is not None:
                r["L"] = max(int(r["think_tokens"]), 1)
            elif r.get("finish_reason") == "length":
                r["L"] = CAP
            else:
                continue
            rows.append(r)
    return rows


def test_problem_ids(rows, seed=0, test_frac=0.2):
    """Same problem split as judge_predictor.py / train_predictor.py."""
    ids = sorted({r["id"] for r in rows})
    np.random.default_rng(seed).shuffle(ids)
    return set(ids[: int(len(ids) * test_frac)])


def parse_conds(spec):
    """'hard,notice:512,budget:512' -> [(name, kind, grace)].
    hard         close the thinking at the cut, no notice
    notice:G     notice text, then up to G more thinking tokens, then a forced close
    budget:G     same, and the notice tells the model the number G"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part == "hard":
            out.append(("hard", "hard", 0))
        else:
            kind, grace = part.split(":")
            if kind not in ("notice", "budget"):
                raise SystemExit(f"unknown condition {part!r}")
            out.append((part, kind, int(grace)))
    return out


def notice_text(kind, grace):
    return NOTICE.format(budget=BUDGET.format(n=grace) if kind == "budget" else "")


def cut_points(L, fracs, absolutes):
    pts = {int(L * f) for f in fracs} | {a for a in absolutes if L > a * 1.1}
    return sorted(u for u in pts if 64 <= u < L - 64)


def clean(text):
    """Drop chat special tokens such as <|im_end|> from generated text."""
    return re.sub(r"<\|[^|>\n]*\|>", "", text or "").strip()


class Server:
    def __init__(self, base_url, model, concurrency):
        import httpx
        self.c = httpx.AsyncClient(base_url=base_url, timeout=None)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self._tok_cache = {}

    async def _post(self, path, body):
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.c.post(path, json=body)
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    resp = getattr(e, "response", None)
                    if resp is not None and 400 <= resp.status_code < 500:
                        raise SystemExit(f"{path} answered {resp.status_code}:\n{resp.text[:1500]}\nSend me this output.")
                    if attempt == 3:
                        raise
                    await asyncio.sleep(1 + attempt)

    async def tok(self, text, cache=False):
        if not text:
            return []
        if cache and text in self._tok_cache:
            return self._tok_cache[text]
        ids = (await self._post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False}))["tokens"]
        if cache:
            self._tok_cache[text] = ids
        return ids

    async def tok_chat(self, messages, template_kwargs):
        return (await self._post("/tokenize", {"model": self.model, "messages": messages,
                                               "add_generation_prompt": True,
                                               "chat_template_kwargs": template_kwargs}))["tokens"]

    async def detok(self, ids):
        if not ids:
            return ""
        return (await self._post("/detokenize", {"model": self.model, "tokens": ids}))["prompt"]

    async def complete(self, ids, max_tokens, seed, stop=None):
        body = {"model": self.model, "prompt": ids, "max_tokens": max_tokens, "seed": seed,
                "skip_special_tokens": False}        # keep </think> visible in the text
        if stop:
            body["stop"] = stop
        d = await self._post("/v1/completions", body)
        ch = d["choices"][0]
        return {"text": ch.get("text") or "", "n": int((d.get("usage") or {}).get("completion_tokens") or 0),
                "finish": ch.get("finish_reason"), "stop_reason": ch.get("stop_reason")}

    async def chat(self, messages, max_tokens, template_kwargs):
        """Plain chat call (used by prune_tail.py to ask which paragraphs to keep)."""
        d = await self._post("/v1/chat/completions", {"model": self.model, "messages": messages,
                                                      "max_tokens": max_tokens, "temperature": 0,
                                                      "chat_template_kwargs": template_kwargs})
        return d["choices"][0]["message"].get("content") or ""

    async def chat_full(self, messages, max_tokens, template_kwargs, seed=0, temperature=None):
        """Chat call that also returns the thinking (when the server's reasoning parser splits it)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": template_kwargs}
        if temperature is not None:
            body["temperature"] = temperature
        d = await self._post("/v1/chat/completions", body)
        ch = d["choices"][0]
        msg = ch["message"]
        return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "finish": ch.get("finish_reason"), "n": int((d.get("usage") or {}).get("completion_tokens") or 0)}

    async def close(self):
        await self.c.aclose()


def messages_for(r, system=None):
    suffix = r.get("suffix")
    msgs = [{"role": "user", "content": r["prompt"] + (SUFFIX if suffix is None else suffix)}]
    return ([{"role": "system", "content": system}] if system else []) + msgs


async def find_think_open(srv, messages, template_kwargs, verbose=False):
    """Does the chat template already open the thinking block, or does the model write <think> itself?
    Returns the text we must put before the thinking prefix."""
    gen = await srv.tok_chat(messages, template_kwargs)
    tail = await srv.detok(gen[-12:])
    if "<think>" in tail:
        if verbose:
            print("the chat template already ends with <think>; nothing to add")
        return ""
    probe = await srv.complete(gen, 6, 0)
    if verbose:
        print("end of the templated prompt:", repr(tail))
        print("first tokens the model writes:", repr(probe["text"]))
    if probe["text"].lstrip().startswith("<think>"):
        return "<think>\n"
    print("!! The model did not start with <think>. Check that thinking is on (--template-kwargs). "
          "Send me the two lines above.")
    return ""
