"""The settled judge: Jev re-asked mid-decode about a request's thinking so far.

A prompt-level prediction says how much thinking a request will need; it cannot say whether this request,
right now, has already got there. The judge can. Every `every` tokens past `start`, the gateway hands Jev
the prompt head and the tail of the thinking streamed so far, and Jev answers whether the final answer is
settled. The answer lands on the request as `settled` (a probability) and the controller's `settled` cut
mode uses it: a settled request's remaining thinking costs (1 - settled) of what the prompt-level
distribution says, so it is the first to be cut, down to what it has already used.

Jev is prefill-only on its own GPU, so a judgment costs about a tenth of a second and never queues behind
the overloaded serving model, unlike a trial answer from the model itself (DEER), which is one more request
at the back of the same queue.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from budget_classifier.tailfeat import FeatureJudge

from .controller import Active

SIGNALS = ("settled", "done", "mean")
BACKENDS = ("jev", "feat")


@dataclass
class Judgment:
    a: Active
    prompt: str
    thinking: str
    used: int
    t0: float = field(default_factory=time.monotonic)
    logprobs: list | None = None


class JudgeScheduler:
    """Latest-wins queue: a request waiting to be judged is judged on its newest thinking, once."""

    def __init__(self, gw, *, concurrency: int = 4, signal: str = "settled", backend: str = "jev",
                 feat: FeatureJudge | None = None):
        if signal not in SIGNALS:
            raise ValueError(f"judge signal must be one of {SIGNALS}")
        if backend not in BACKENDS:
            raise ValueError(f"judge backend must be one of {BACKENDS}")
        self.gw, self.concurrency, self.signal = gw, concurrency, signal
        # feat: the trained surface-feature judge (budget_classifier.tailfeat), scored inline on the gateway
        # CPU from the streamed thinking and its token logprobs, no model call
        self.backend, self.feat = backend, feat
        self.jobs: dict[str, Judgment] = {}
        self.workers = 0
        self.counts = {"submitted": 0, "replaced": 0, "asked": 0, "answered": 0, "failed": 0, "after_finish": 0,
                       "wait_s": 0.0, "call_ms": 0.0, "settled_sum": 0.0}

    @property
    def enabled(self) -> bool:
        if self.backend == "feat":
            return self.feat is not None
        return self.gw.clf.jev is not None

    @property
    def wants_logprobs(self) -> bool:
        return self.backend == "feat" and self.feat is not None

    def submit(self, a: Active, prompt: str, thinking: str, used: int, logprobs: list | None = None) -> bool:
        if not self.enabled:
            return False
        self.counts["submitted"] += 1
        if self.backend == "feat":
            t0 = time.perf_counter()
            v = self.feat.score(thinking, logprobs or [], used)
            self.counts["asked"] += 1
            self._record(a, used, {"settled": v, "done": None, "level": None,
                                   "ms": round((time.perf_counter() - t0) * 1000, 3)}, v)
            return True
        if a.rid in self.jobs:
            self.counts["replaced"] += 1
        self.jobs[a.rid] = Judgment(a, prompt, thinking, used)
        if self.workers < self.concurrency:
            self.workers += 1
            asyncio.get_running_loop().create_task(self._work())
        return True

    def _next(self) -> Judgment | None:
        for rid in [r for r in self.jobs if r not in self.gw.controller.active]:
            del self.jobs[rid]
            self.counts["after_finish"] += 1
        if not self.jobs:
            return None
        # the request with the most thinking first: it is the one most likely to be settled, so the one a cut
        # is most likely to be free on
        job = max(self.jobs.values(), key=lambda j: (j.used, -j.t0))
        del self.jobs[job.a.rid]
        return job

    def value(self, out: dict) -> float | None:
        s, d = out.get("settled"), out.get("done")
        if self.signal == "settled":
            return s
        if self.signal == "done":
            return d
        if s is None or d is None:
            return s if d is None else d
        return (s + d) / 2

    async def _work(self) -> None:
        try:
            while (job := self._next()) is not None:
                await self._judge(job)
        finally:
            self.workers -= 1

    async def _judge(self, job: Judgment) -> None:
        a = job.a
        self.counts["asked"] += 1
        self.counts["wait_s"] += time.monotonic() - job.t0
        out = await self.gw.clf.jev.ask_settled(job.prompt, job.thinking, job.used)
        if out is None:
            self.counts["failed"] += 1
            return
        if a.rid not in self.gw.controller.active:
            self.counts["after_finish"] += 1
            return
        v = self.value(out)
        if v is None:
            self.counts["failed"] += 1
            return
        self._record(a, job.used, out, v)

    def _record(self, a: Active, used: int, out: dict, v: float) -> None:
        self.counts["answered"] += 1
        self.counts["call_ms"] += out.get("ms", 0.0)
        self.counts["settled_sum"] += v
        a.settled, a.judged_at = float(v), used
        a.judgments += 1
        if self.gw.log is not None:
            row = {"judge": True, "rid": a.rid, "used": used, "backend": self.backend, "settled": out.get("settled"),
                   "done": out.get("done"), "level": out.get("level"), "value": round(v, 4), "ms": out.get("ms"),
                   "limit": a.limit, "cut_from": a.cut_from, "p": round(self.gw.monitor.p, 3),
                   "policy": self.gw.controller.policy, "mode": self.gw.controller.mode, "ts": time.time()}
            self.gw.log.write(json.dumps(row) + "\n")
            self.gw.log.flush()

    def stats(self) -> dict:
        n = self.counts["answered"]
        return {**self.counts, "enabled": self.enabled, "signal": self.signal, "backend": self.backend,
                "queued": len(self.jobs),
                "workers": self.workers, "wait_s": round(self.counts["wait_s"], 3),
                "mean_call_ms": round(self.counts["call_ms"] / n, 1) if n else None,
                "mean_settled": round(self.counts["settled_sum"] / n, 4) if n else None}
