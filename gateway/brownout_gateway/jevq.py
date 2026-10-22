from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .controller import JEV_PREDICTORS, Active
from .priority import from_answer, question


@dataclass
class Job:
    a: Active
    cls: str
    step: str
    deadline_ms: int | None
    done: asyncio.Future
    t0: float = field(default_factory=time.monotonic)


class JevScheduler:
    def __init__(self, clf, controller, *, concurrency: int = 1):
        self.clf, self.controller = clf, controller
        self.concurrency = concurrency
        self.jobs: dict[str, Job] = {}
        self.workers = 0
        self.counts = {"submitted": 0, "asked": 0, "answered": 0, "before_first_token": 0, "during_decode": 0,
                       "after_finish": 0, "dropped_finished": 0, "tier_lowered": 0, "tier_raised": 0,
                       "priority_changed": 0, "wait_s": 0.0}

    @property
    def enabled(self) -> bool:
        return self.clf.jev is not None

    def submit(self, a: Active, cls: str, step: str, deadline_ms: int | None) -> asyncio.Future | None:
        if not self.enabled:
            return None
        loop = asyncio.get_running_loop()
        job = Job(a, cls, step, deadline_ms, loop.create_future())
        self.jobs[a.rid] = job
        self.counts["submitted"] += 1
        if self.workers < self.concurrency:
            self.workers += 1
            loop.create_task(self._work())
        return job.done

    def _next(self) -> Job | None:
        for rid in [r for r in self.jobs if r not in self.controller.active]:
            self.jobs.pop(rid).done.set_result(False)
            self.counts["dropped_finished"] += 1
        if not self.jobs:
            return None
        job = min(self.jobs.values(), key=lambda j: (j.a.used, j.t0))
        del self.jobs[job.a.rid]
        return job

    async def _work(self) -> None:
        try:
            while (job := self._next()) is not None:
                try:
                    await self._ask(job)
                finally:
                    if not job.done.done():
                        job.done.set_result(job.a.jev_at is not None)
        finally:
            self.workers -= 1

    async def _ask(self, job: Job) -> None:
        a = job.a
        self.counts["asked"] += 1
        self.counts["wait_s"] += time.monotonic() - job.t0
        pred, extra = await self.clf.ask_jev(a.rid, {"priority": question(job.cls, job.step, job.deadline_ms)})
        if a.rid not in self.controller.active:
            self.counts["after_finish"] += 1
            return
        level = from_answer(extra.get("priority"))
        if pred is None and level is None:
            return
        self.counts["answered"] += 1
        self.counts["before_first_token" if a.used == 0 else "during_decode"] += 1
        a.jev_at = a.used
        if pred is not None:
            a.jev_tier = pred.label
        if self.controller.predictor not in JEV_PREDICTORS:
            return
        if pred is not None and pred.budget != a.tier_budget:
            self.counts["tier_lowered" if pred.budget < a.tier_budget else "tier_raised"] += 1
            a.tier, a.tier_budget = pred.label, pred.budget
        if level is not None:
            if level != a.priority:
                self.counts["priority_changed"] += 1
            a.set_priority(level, "jev")

    def stats(self) -> dict:
        asked = self.counts["asked"]
        return {**self.counts, "wait_s": round(self.counts["wait_s"], 3),
                "mean_wait_s": round(self.counts["wait_s"] / asked, 3) if asked else None,
                "queued": len(self.jobs), "workers": self.workers}
