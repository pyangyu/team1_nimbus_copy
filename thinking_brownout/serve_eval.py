"""Serving experiment for Thinking Brownout: replay a burst of requests against vLLM under one policy
and record, for every request, when it arrived, when it got its first token (TTFT), when it finished,
how much it thought, whether it was cut or rejected, and whether its answer is right.

    python serve_eval.py traces/Qwen__Qwen3.8-27B.eval.jsonl --policy nocut  --out serve/nocut.jsonl
    python serve_eval.py traces/Qwen__Qwen3.8-27B.eval.jsonl --policy tb     --out serve/tb.jsonl --adapter cut
    python serve_report.py serve/*.jsonl --ref serve/nocut.jsonl

Policies (all see the same arrivals: same times, same problems, request i is the same in every run):
  nocut   FCFS queue, every request thinks in full.
  reject  SLO-aware admission: a request whose predicted TTFT is over the SLO is turned away.
  budget  admission-time budget: thinking is capped when the request starts, lower when the queue is long.
  exit    confidence-based early exit (Dynasor-style probes); needs fewer agreeing probes when the queue is long.
  hard    Thinking Brownout selection, but a cut forces </think> at once (no adapter).
  tb      Thinking Brownout: the cut writes [cut] and the activated LoRA takes over, in place.

Every cut happens INSIDE vLLM, in place (tb_vllm.py): the client sends "<rid> tb|hard" over UDP and the
request keeps running; nothing is aborted and nothing is prefilled again. A tb request is sent to the
adapter from the start, but the adapter is masked until the marker, so before a cut it is the base model.
vLLM must run with tb_vllm loaded (see tb_vllm.py) on the same machine as this script.

The gateway owns the batch: at most --slots requests run at once and the rest wait in a FIFO queue, so a
"slot" in the paper is a slot here. Start vLLM with --max-num-seqs at least --slots (+ room for probes).
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
import socket
import statistics
import sys
import time
from collections import defaultdict, deque

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutlib import CAP, Server, find_think_open, graded, messages_for, read_traces  # noqa: E402

INVOKE = "\n\n[cut]\n"                     # must match cut_alora.INVOKE
CLOSE = "\n</think>\n\n"
EFFORT_Q = ("Rate how much reasoning this problem needs on a 0-9 scale: "
            "0 = the answer is immediate; 1 = one short calculation; 2 = two or three simple steps; "
            "3 = a few steps; 4 = several steps; 5 = many steps; 6 = a long multi-step solution; "
            "7 = a long solution with case work; 8 = a very long solution that needs search or a construction; "
            "9 = a very long solution with many dead ends. "
            "Do not solve it. Reply with the digit only.")            # must match classifier.EFFORT_Q
SUFFIX = "\n\nPut your final answer in \\boxed{}."
PROBE = "\n\n... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
# prior median thinking length per rounded self-rating, replaced online by finished uncut requests
LHAT0 = {0: 150, 1: 250, 2: 450, 3: 1200, 4: 2500, 5: 5000, 6: 9000, 7: 15000, 8: 22000, 9: 28000}


# ------------------------------------------------------------------ arrivals

def make_arrivals(args, rng):
    """Arrival times (s). Either BurstGPT timestamps scaled to --rate, or Poisson at --rate with bursts."""
    if args.burstgpt:
        ts = []
        with open(args.burstgpt) as f:
            head = f.readline().strip().split(",")
            col = next(i for i, h in enumerate(head) if h.lower().startswith("timestamp"))
            for line in f:
                try:
                    ts.append(float(line.split(",")[col]))
                except (ValueError, IndexError):
                    pass
        ts = sorted(t - args.burstgpt_start for t in ts if args.burstgpt_start <= t < args.burstgpt_start + args.duration * args.burstgpt_speed)
        ts = [t / args.burstgpt_speed for t in ts]                   # compress the trace in time
        keep = args.rate * args.duration / max(len(ts), 1)           # thin (or keep all) to hit the mean rate
        return [t for t in ts if rng.random() < keep] if keep < 1 else ts
    bursts = []
    for b in filter(None, args.bursts.split(",")):
        s, d, m = (float(x) for x in b.split(":"))
        bursts.append((s, s + d, m))
    peak = max([m for *_, m in bursts] + [1.0]) * args.rate
    t, out = 0.0, []
    while True:                                                      # thinning of a Poisson process
        t += rng.expovariate(peak)
        if t >= args.duration:
            return out
        lam = args.rate * next((m for s, e, m in bursts if s <= t < e), 1.0)
        if rng.random() < lam / peak:
            out.append(t)


# ------------------------------------------------------------------ requests

class Req:
    def __init__(self, rid, row, arrival):
        self.rid, self.row, self.arrival = rid, row, arrival
        self.score = None
        self.admit = self.first = self.finish = None
        self.rejected = False
        self.gen_ids = []                 # tokens of the main (uncut) generation
        self.think_done = False
        self.u = 0                        # thinking tokens written before any cut
        self.after = 0                    # thinking tokens written after the cut
        self.answer = ""
        self.answer_tok = 0
        self.cut = None                   # None, "hard", "tb", "budget", "exit"
        self.cut_u = None
        self.cut_t = None
        self.task = None
        self.cut_requested = None
        self.marker_seen = False
        self.budget = None
        self.probes = []
        self.t_tok = deque(maxlen=64)     # recent token times, for the decode-speed estimate

    def record(self, policy, slo):
        ttft = None if self.first is None else self.first - self.arrival
        return {"rid": self.rid, "id": self.row["id"], "policy": policy, "gold": self.row["gold"],
                "arrival": round(self.arrival, 3), "score": self.score, "rejected": self.rejected,
                "admit": None if self.admit is None else round(self.admit, 3),
                "ttft": None if ttft is None else round(ttft, 3),
                "finish": None if self.finish is None else round(self.finish, 3),
                "e2e": None if self.finish is None else round(self.finish - self.arrival, 3),
                "slo_ok": (ttft is not None and ttft <= slo),
                "think_tokens": self.u + self.after, "u": self.u, "after_cut": self.after,
                "answer_tokens": self.answer_tok, "cut": self.cut, "cut_u": self.cut_u,
                "cut_at": None if self.cut_t is None else round(self.cut_t, 3),
                "budget": self.budget, "answer": self.answer,
                "correct": (not self.rejected) and graded(self.answer, self.row["gold"])}


class Gateway:
    def __init__(self, args, srv, client, think_open):
        self.a, self.srv, self.c, self.think_open = args, srv, client, think_open
        self.waiting = deque()
        self.running = {}
        self.free = args.slots
        self.events = {}
        self.t0 = None
        self.lhat_obs = defaultdict(lambda: deque(maxlen=200))
        self.v = args.v0                  # tokens/s per running sequence (EWMA)
        self.digit_ids = None
        self.n_cut = 0
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tag = f"{args.policy}{random.Random().randrange(10**6):06d}"   # unique tb_rid prefix per run
        self.n_invoke = 5

    def now(self):
        return time.monotonic() - self.t0

    # ---------------- estimates
    def lhat(self, s):
        k = int(min(9, max(0, round(s if s is not None else 4))))
        obs = self.lhat_obs[k]
        return statistics.median(obs) if len(obs) >= 5 else LHAT0[k]

    def remaining(self, r):
        if r.cut in ("hard", "budget", "exit"):
            return self.a.answer_est
        if r.cut == "tb":
            return max(self.a.c_after - r.after, 0) + self.a.answer_est
        if r.think_done:
            return max(self.a.answer_est - r.answer_tok, 20)
        left = self.lhat(r.score) - r.u
        if r.budget is not None:
            left = min(left, r.budget - r.u)
        return max(left, 50) + self.a.answer_est

    def predicted_ttft(self, extra_pos=0):
        """Predicted TTFT of a request that would join the queue now."""
        fin = sorted(self.remaining(r) / max(self.v, 1e-3) for r in self.running.values())
        m = len(self.waiting) + 1 + extra_pos
        if self.free >= m:
            return 0.0
        need = m - self.free
        if not fin:
            return 0.0
        # the need-th slot to free up; past one round, each slot serves another typical request
        rounds, i = divmod(need - 1, len(fin))
        typical = (statistics.median(self.lhat(r.score) for r in self.running.values()) + self.a.answer_est) / max(self.v, 1e-3)
        return fin[i] + rounds * typical

    # ---------------- vLLM calls
    async def score(self, r):
        if self.digit_ids is None:
            ids = []
            for d in "0123456789":
                ids.append(int((await self.srv.tok(d))[0]))
            self.digit_ids = ids
        body = {"model": self.a.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "allowed_token_ids": self.digit_ids, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": r.row["prompt"] + SUFFIX + "\n\n---\n" + EFFORT_Q}]}
        d = (await self.c.post("/v1/chat/completions", json=body)).json()
        lp = ((d["choices"][0].get("logprobs") or {}).get("content") or [None])[0] or {}
        p = [0.0] * 10
        for c in lp.get("top_logprobs") or []:
            s = c["token"].strip()
            if len(s) == 1 and s.isdigit():
                p[int(s)] += math.exp(c["logprob"])
        z = sum(p) or 1.0
        return sum(k * x for k, x in enumerate(p)) / z

    def send_cut(self, r, mode):
        """Tell the engine (tb_vllm.TBCut) to cut request r in place: mode 'tb' or 'hard'."""
        r.cut, r.cut_u, r.cut_t = r.cut_requested, r.u, self.now()
        self.udp.sendto(f"{self.tag}-{r.rid} {mode}".encode(), ("127.0.0.1", self.a.tb_port))

    async def stream(self, r, prompt_ids, max_tokens, model, on_tok):
        body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "stream": True,
                "temperature": self.a.temperature, "top_p": self.a.top_p, "seed": self.a.seed + r.rid,
                "logprobs": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False,
                "vllm_xargs": {"tb_rid": f"{self.tag}-{r.rid}"}}
        async with self.c.stream("POST", "/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code}: {(await resp.aread())[:500]!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[6:])["choices"]
                if not ch:
                    continue
                toks = (ch[0].get("logprobs") or {}).get("tokens") or []
                ids = [int(t.split(":", 1)[1]) for t in toks if t.startswith("token_id:")]
                on_tok(ids, ch[0].get("text") or "")

    # ---------------- request life cycle
    async def lifecycle(self, r, out, lock):
        a = self.a
        if a.policy in ("tb", "hard", "reject"):
            try:
                r.score = await self.score(r)
            except Exception:
                r.score = None
        if a.policy == "reject" and self.predicted_ttft() > a.slo:
            r.rejected = True
            await self.write(r, out, lock)
            return
        # wait for a slot (FIFO)
        if self.free > 0 and not self.waiting:
            self.free -= 1
        else:
            ev = asyncio.Event()
            self.waiting.append(r)
            self.events[r.rid] = ev
            await ev.wait()
        r.admit = self.now()
        self.running[r.rid] = r
        try:
            await self.run(r)
        except Exception as e:                       # keep the experiment going; the request counts as wrong
            print(f"  request {r.rid} failed: {e!r}", flush=True)
        finally:
            r.finish = self.now()
            self.running.pop(r.rid, None)
            if r.cut is None and r.think_done and not r.rejected:
                self.lhat_obs[int(min(9, max(0, round(r.score if r.score is not None else 4))))].append(r.u)
            if self.waiting:
                nxt = self.waiting.popleft()
                self.events.pop(nxt.rid).set()
            else:
                self.free += 1
        await self.write(r, out, lock)

    async def run(self, r):
        """One request, one stream, start to end. A cut is a message to the engine; the stream goes on:
        tb  -> the engine writes [cut], the adapter takes over from there, closes and answers;
        hard / budget / exit -> the engine writes </think> and the model answers."""
        a = self.a
        msgs = messages_for(r.row)
        base = await self.srv.tok_chat(msgs, {"enable_thinking": True})
        base = base + (await self.srv.tok(self.think_open, cache=True) if self.think_open else [])
        if a.policy == "budget":                     # admission budget, CALM-MAS style: shrink with the queue
            q = len(self.waiting)
            r.budget = int(max(a.budget_min, a.budget_max / (1 + q / max(a.slots, 1))))
        tail = [""]
        marker = INVOKE.strip()                      # "[cut]"

        def on_tok(ids, t):
            now = self.now()
            if r.first is None:
                r.first = now
            r.gen_ids += ids
            for _ in ids:
                r.t_tok.append(now)
            if r.think_done:
                r.answer += t
                r.answer_tok += len(ids)
                return
            tail[0] = (tail[0] + t)[-32:]
            if "</think>" in tail[0]:
                r.think_done = True
                r.answer = tail[0].split("</think>", 1)[1]
                return
            if r.cut == "tb" and not r.marker_seen and marker in tail[0]:
                r.marker_seen = True
                r.u = max(r.u - (self.n_invoke - len(ids)), 0)   # the marker is not thinking
                r.after = 0
                tail[0] = ""
                return
            if r.marker_seen:
                r.after += len(ids)
            else:
                r.u += len(ids)
            if r.budget is not None and r.cut is None and r.u >= r.budget:
                r.cut_requested = "budget"
                self.send_cut(r, "hard")

        model = a.adapter if a.policy == "tb" else a.model   # tb: adapter attached, masked until the marker
        r.task = asyncio.ensure_future(self.stream(r, base, CAP - len(base) - 8, model, on_tok))
        prober = asyncio.ensure_future(self.prober(r, base)) if a.policy == "exit" else None
        try:
            await r.task
        finally:
            if prober:
                prober.cancel()
        if r.cut == "tb" and not r.marker_seen:      # the thinking ended before the cut landed
            r.cut = r.cut_u = None
        if not r.think_done:                         # ran to the cap without closing: no answer
            r.answer = ""

    async def prober(self, r, base):
        """Dynasor-style: every --probe-every thinking tokens, ask for the answer now; exit when the
        last `need` probes agree (need drops from 3 to 2 while requests are waiting)."""
        a = self.a
        probe_ids = await self.srv.tok(PROBE, cache=True)
        last = 0
        while not r.think_done and r.cut is None:
            await asyncio.sleep(0.2)
            if r.u - last < a.probe_every:
                continue
            last = r.u
            d = (await self.c.post("/v1/completions", json={
                "model": a.model, "prompt": base + list(r.gen_ids) + probe_ids, "max_tokens": 20,
                "temperature": 0, "stop": ["}"]})).json()
            ans = (d["choices"][0].get("text") or "").strip()
            r.probes.append(ans)
            need = 2 if self.waiting else 3
            if ans and len(r.probes) >= need and len(set(r.probes[-need:])) == 1 and not r.think_done and r.cut is None:
                r.cut_requested = "exit"
                self.send_cut(r, "hard")
                return

    # ---------------- the Thinking Brownout controller (Algorithm 1)
    async def controller(self):
        a = self.a
        while True:
            await asyncio.sleep(a.interval)
            speeds = [len(r.t_tok) / max(r.t_tok[-1] - r.t_tok[0], 1e-3)
                      for r in self.running.values() if len(r.t_tok) >= 16]
            if speeds:
                self.v = 0.8 * self.v + 0.2 * statistics.median(speeds)
            if a.policy not in ("tb", "hard") or not self.waiting:
                continue
            now = self.now()
            fin = [self.remaining(r) / self.v for r in self.running.values()]
            k = 0
            for m, w in enumerate(self.waiting, start=1):
                slack = a.slo - (now - w.arrival)
                k = max(k, m - self.free - sum(1 for f in fin if f <= slack))
            if k <= 0:
                continue
            smin = a.slo - (now - self.waiting[0].arrival)
            elig = [r for r in self.running.values()
                    if r.cut is None and not r.think_done
                    and r.u >= a.u_min and self.remaining(r) / self.v > smin]
            sel = a.select
            if sel == "random":
                random.shuffle(elig)
            elif sel == "written":                   # most thinking written so far first
                elig.sort(key=lambda r: r.u, reverse=True)
            elif sel == "low":                       # lowest self-rating first
                elig.sort(key=lambda r: (round(r.score if r.score is not None else 4), -r.u))
            elif sel == "oracle":                    # most thinking still to go, from the request's own trace
                elig.sort(key=lambda r: r.row["L"] - r.u, reverse=True)
            else:                                    # high: Thinking Brownout, highest self-rating first
                elig.sort(key=lambda r: (round(r.score if r.score is not None else 4), r.u), reverse=True)
            for r in elig[:k]:
                r.cut_requested = a.policy
                self.n_cut += 1
                self.send_cut(r, "tb" if a.policy == "tb" else "hard")

    async def write(self, r, out, lock):
        async with lock:
            out.write(json.dumps(r.record(self.a.policy, self.a.slo), ensure_ascii=False) + "\n")
            out.flush()


async def main_async(args):
    rng = random.Random(args.seed)
    rows, seen = [], set()
    for r in read_traces(args.problems):
        if r["id"] not in seen:
            seen.add(r["id"])
            rows.append({"id": r["id"], "prompt": r["prompt"], "gold": r["gold"], "suffix": r.get("suffix"),
                         "L": r["L"]})
    rng.shuffle(rows)
    arrivals = make_arrivals(args, rng)
    print(f"{len(rows)} problems, {len(arrivals)} arrivals over {args.duration:.0f} s "
          f"(mean {len(arrivals) / args.duration:.2f}/s), policy {args.policy}, {args.slots} slots, SLO {args.slo} s")
    srv = Server(args.base_url, args.model, 1000)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    client = httpx.AsyncClient(base_url=args.base_url, timeout=None, limits=limits)
    think_open = await find_think_open(srv, messages_for(rows[0]), {"enable_thinking": True}, verbose=True)
    gw = Gateway(args, srv, client, think_open)
    gw.n_invoke = len(await srv.tok(INVOKE))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lock = asyncio.Lock()
    with open(args.out, "w") as out:
        out.write(json.dumps({"meta": True, "policy": args.policy, "select": args.select, "adapter": args.adapter, "slots": args.slots, "slo": args.slo,
                              "rate": args.rate, "bursts": args.bursts, "duration": args.duration,
                              "burstgpt": args.burstgpt, "n": len(arrivals), "seed": args.seed}) + "\n")
        gw.t0 = time.monotonic()
        ctl = asyncio.ensure_future(gw.controller())
        tasks = []
        for i, t in enumerate(arrivals):
            delay = t - gw.now()
            if delay > 0:
                await asyncio.sleep(delay)
            r = Req(i, rows[i % len(rows)], t)
            tasks.append(asyncio.ensure_future(gw.lifecycle(r, out, lock)))
            if i % 50 == 0:
                print(f"  t={gw.now():6.1f}s  arrived {i + 1}  running {len(gw.running)}  waiting {len(gw.waiting)}"
                      f"  cut {gw.n_cut}  v={gw.v:.1f} tok/s/seq", flush=True)
        while not all(t.done() for t in tasks):
            await asyncio.sleep(5)
            print(f"  t={gw.now():6.1f}s  done {sum(t.done() for t in tasks)}/{len(tasks)}  running {len(gw.running)}"
                  f"  waiting {len(gw.waiting)}  cut {gw.n_cut}", flush=True)
        ctl.cancel()
    await client.aclose()
    await srv.close()
    print(f"done -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", help="trace file whose problems (prompt, gold) are served, e.g. the eval traces")
    ap.add_argument("--policy", required=True, choices=["nocut", "reject", "budget", "exit", "hard", "tb"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--adapter", default="cut", help="LoRA module name of the cut adapter in vLLM (policy tb)")
    # load
    ap.add_argument("--rate", type=float, default=0.5, help="base arrival rate, requests/s")
    ap.add_argument("--bursts", default="30:10:4.5,90:10:4.5", help="start:duration:multiplier,... (s)")
    ap.add_argument("--duration", type=float, default=150.0, help="arrivals stop after this many seconds")
    ap.add_argument("--burstgpt", default=None, help="BurstGPT csv: replay its timestamps instead of --bursts")
    ap.add_argument("--burstgpt-start", type=float, default=0.0)
    ap.add_argument("--burstgpt-speed", type=float, default=1.0, help="compress the trace in time by this factor")
    # serving
    ap.add_argument("--slots", type=int, default=32, help="max requests running at once (the batch)")
    ap.add_argument("--slo", type=float, default=10.0, help="TTFT SLO, seconds")
    ap.add_argument("--interval", type=float, default=0.5, help="controller step, seconds")
    ap.add_argument("--select", default="high", choices=["high", "low", "written", "random", "oracle"],
                    help="which running requests to cut (hard / tb); high = Thinking Brownout")
    ap.add_argument("--u-min", type=int, default=256, help="never cut before this many thinking tokens")
    ap.add_argument("--tb-port", type=int, default=5599, help="UDP port of tb_vllm's cut listener (TB_PORT)")
    ap.add_argument("--c-after", type=int, default=150, help="expected thinking after [cut], for the TTFT model")
    ap.add_argument("--answer-est", type=int, default=330, help="expected answer length, for the TTFT model")
    ap.add_argument("--answer-max", type=int, default=2048)
    ap.add_argument("--v0", type=float, default=30.0, help="initial per-sequence decode speed, tokens/s")
    ap.add_argument("--budget-max", type=int, default=16384, help="policy budget")
    ap.add_argument("--budget-min", type=int, default=512, help="policy budget")
    ap.add_argument("--probe-every", type=int, default=256, help="policy exit")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
