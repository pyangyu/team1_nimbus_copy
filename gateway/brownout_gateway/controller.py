from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .pressure import PressureMonitor
from .priority import WEIGHT

POLICIES = ("none", "reject", "defer", "uniform", "brownout")
CUTTING = ("uniform", "brownout")
NOTICES = ("silent", "announced")
PREDICTORS = ("jev", "jev-serial", "online", "oracle")
JEV_PREDICTORS = ("jev", "jev-serial")


@dataclass
class Active:
    rid: str
    requested: int
    tier_budget: int
    priority: str
    limit: int
    priority_source: str = "heuristic"
    used: int = 0
    cut_from: int | None = None
    tier: str = ""
    jev_at: int | None = None
    jev_tier: str | None = None
    replica: int = 0

    def set_priority(self, level: str, source: str) -> None:
        self.priority, self.priority_source = level, source


class Controller:
    def __init__(self, monitor: PressureMonitor, *, policy: str = "brownout", notice: str = "silent",
                 predictor: str = "jev", min_left: int = 64, max_frac: float = 0.9, active_at: float = 0.05,
                 renotice_min: int = 64, renotice_frac: float = 0.1, interval: float = 0.25):
        self.monitor = monitor
        self.monitors = [monitor]
        self.set_policy(policy, notice, predictor)
        self.min_left, self.max_frac, self.active_at, self.interval = min_left, max_frac, active_at, interval
        self.renotice_min, self.renotice_frac = renotice_min, renotice_frac
        self.active: dict[str, Active] = {}
        self.cuts = 0

    def set_policy(self, policy: str, notice: str | None = None, predictor: str | None = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if notice is not None and notice not in NOTICES:
            raise ValueError(f"notice must be one of {NOTICES}")
        if predictor is not None and predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {PREDICTORS}")
        self.policy = policy
        if notice is not None:
            self.notice = notice
        if predictor is not None:
            self.predictor = predictor

    @property
    def cutting(self) -> bool:
        return any(self.cutting_at(k) for k in range(len(self.monitors)))

    def cutting_at(self, replica: int) -> bool:
        return self.policy in CUTTING and self.monitors[replica].p >= self.active_at

    def p(self, a: Active) -> float:
        return self.monitors[a.replica].p

    def base(self, a: Active) -> int:
        return a.requested if self.policy == "uniform" else min(a.requested, a.tier_budget)

    def weight(self, a: Active) -> float:
        return WEIGHT[a.priority] if self.policy == "brownout" else 1.0

    def target(self, a: Active) -> int:
        frac = 1 - min(self.max_frac, self.p(a) * self.weight(a))
        return max(int(self.base(a) * frac), a.used + self.min_left)

    def renotice(self, left: int) -> int:
        return max(self.renotice_min, int(self.renotice_frac * left))

    def start(self, a: Active) -> Active:
        if self.cutting_at(a.replica):
            a.limit = min(a.requested, max(int(self.base(a) * (1 - min(self.max_frac, self.p(a) * self.weight(a)))),
                                           self.min_left))
        else:
            a.limit = a.requested
        self.active[a.rid] = a
        return a

    def finish(self, a: Active) -> None:
        self.active.pop(a.rid, None)

    def tick(self) -> None:
        if not self.cutting:
            return
        for a in list(self.active.values()):
            if not self.cutting_at(a.replica):
                continue
            t = self.target(a)
            if t < a.limit:
                if a.cut_from is None:
                    a.cut_from = a.limit
                    self.cuts += 1
                a.limit = t

    async def run(self) -> None:
        while True:
            self.tick()
            await asyncio.sleep(self.interval)

    def stats(self) -> dict:
        return {"policy": self.policy, "notice": self.notice, "predictor": self.predictor, "cutting": self.cutting,
                "in_flight": len(self.active), "cuts": self.cuts}
