"""Needed-budget labels.

A finished request tells us how long the model thought, not how much of that it needed. The prober
replays the request's own thinking cut at the bucket edges below its length, closes the thinking there,
and asks for the final answer only. The label is the lowest edge whose answer still matches the answer
at the full thinking. Probes walk down from the top edge and stop at the first mismatch.

Each probe is one short vLLM call (the prefix is in the prefix cache, the completion is a few dozen
tokens). The label feeds the classifier, Jev's example memory and the oracle exactly where the length
label used to.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from .answers import EARLY_STOP, FINAL_PREFIX, final_answer


@dataclass
class ProbeJob:
    rid: str
    prompt: str
    thinking: str
    marks: dict[int, int]
    used: int
    natural: bool
    model: str | None
    key: str
    headers: dict[str, str]
    t0: float = field(default_factory=time.monotonic)


class Prober:
    def __init__(self, gw, *, concurrency: int = 2, max_edges: int = 3, final_tokens: int = 48,
                 max_queue: int = 1000, enabled: bool = True):
        self.gw = gw
        self.concurrency, self.max_edges, self.final_tokens, self.max_queue = concurrency, max_edges, final_tokens, max_queue
        self.enabled = enabled
        self.queue: list[ProbeJob] = []
        self.workers = 0
        self.in_flight = 0
        self.counts = {"submitted": 0, "labeled": 0, "probes": 0, "errors": 0, "dropped": 0, "no_reference": 0,
                       "needed_below_used": 0, "wait_s": 0.0}
        self.by_edge: dict[int, dict[str, int]] = {}

    @staticmethod
    def needed_tokens(edge: int) -> int:
        """The answer survived a cut at `edge`, so the tightest bucket is the one whose upper edge is `edge`."""
        return max(edge - 1, 0)

    def submit(self, job: ProbeJob) -> bool:
        if not self.enabled:
            return False
        if len(self.queue) >= self.max_queue:
            self.counts["dropped"] += 1
            return False
        self.queue.append(job)
        self.counts["submitted"] += 1
        if self.workers < self.concurrency:
            self.workers += 1
            asyncio.get_running_loop().create_task(self._work())
        return True

    async def _work(self) -> None:
        try:
            while self.queue:
                job = self.queue.pop(0)
                self.counts["wait_s"] += time.monotonic() - job.t0
                self.in_flight += 1
                try:
                    await self._label(job)
                except Exception:
                    self.counts["errors"] += 1
                    self._observe(job, job.used, job.natural, -1)
                finally:
                    self.in_flight -= 1
        finally:
            self.workers -= 1

    async def _answer(self, job: ProbeJob, thinking: str) -> str | None:
        body = {"model": job.model, "prompt": job.prompt + thinking + EARLY_STOP + "</think>\n\n" + FINAL_PREFIX,
                "max_tokens": self.final_tokens, "temperature": 0.0, "stream": False, "stop": ["<|im_end|>"]}
        self.counts["probes"] += 1
        r = await self.gw.http.post(f"{self.gw.upstream}/completions", json=body, headers=job.headers)
        r.raise_for_status()
        text = r.json()["choices"][0].get("text") or ""
        return final_answer(FINAL_PREFIX + text)

    async def _label(self, job: ProbeJob) -> None:
        ref = await self._answer(job, job.thinking)
        if ref is None:
            self.counts["no_reference"] += 1
            self._observe(job, job.used, job.natural)
            return
        edges = sorted((e for e in job.marks if 0 < e < job.used), reverse=True)[: self.max_edges]
        needed, probes = job.used, 1
        for e in edges:
            probes += 1
            got = await self._answer(job, job.thinking[: job.marks[e]])
            stat = self.by_edge.setdefault(e, {"probed": 0, "agreed": 0})
            stat["probed"] += 1
            if got != ref:
                break
            stat["agreed"] += 1
            needed = self.needed_tokens(e)
        if needed < job.used:
            self.counts["needed_below_used"] += 1
        # a request that hit its own budget without a matching lower edge needed at least what it used
        self._observe(job, needed, job.natural or needed < job.used, probes)

    def _observe(self, job: ProbeJob, needed: int, finished: bool, probes: int = 0) -> None:
        self.counts["labeled"] += 1
        if finished:
            self.gw.oracle[job.key] = needed
        obs = self.gw.clf.observe(job.rid, needed, finished=finished, model=job.model)
        if self.gw.log is not None:
            row = {"probe": True, "rid": job.rid, "used": job.used, "needed": needed, "natural": job.natural,
                   "finished": finished, "probes": probes, "label": obs.get("label"), "bucket": obs.get("bucket"),
                   "policy": self.gw.controller.policy, "predictor": self.gw.controller.predictor,
                   "mode": self.gw.controller.mode, "ts": time.time()}
            self.gw.log.write(json.dumps(row) + "\n")
            self.gw.log.flush()

    def stats(self) -> dict:
        n = self.counts["labeled"]
        return {**self.counts, "enabled": self.enabled, "queued": len(self.queue), "in_flight": self.in_flight,
                "wait_s": round(self.counts["wait_s"], 3),
                "probes_per_label": round(self.counts["probes"] / n, 3) if n else None,
                "by_edge": {str(e): v for e, v in sorted(self.by_edge.items())}}
