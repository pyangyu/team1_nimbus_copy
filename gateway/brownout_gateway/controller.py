from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .pressure import PressureMonitor
from .priority import WEIGHT

POLICIES = ("none", "reject", "defer", "uniform", "brownout")
CUTTING = ("uniform", "brownout")
NOTICES = ("silent", "announced")


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

    def set_priority(self, level: str, source: str) -> None:
        self.priority, self.priority_source = level, source


class Controller:
    def __init__(self, monitor: PressureMonitor, *, policy: str = "brownout", notice: str = "silent",
                 min_left: int = 64, max_frac: float = 0.9, active_at: float = 0.05, interval: float = 0.25):
        self.monitor = monitor
        self.set_policy(policy, notice)
        self.min_left, self.max_frac, self.active_at, self.interval = min_left, max_frac, active_at, interval
        self.active: dict[str, Active] = {}
        self.cuts = 0

    def set_policy(self, policy: str, notice: str | None = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if notice is not None and notice not in NOTICES:
            raise ValueError(f"notice must be one of {NOTICES}")
        self.policy = policy
        if notice is not None:
            self.notice = notice

    @property
    def cutting(self) -> bool:
        return self.policy in CUTTING and self.monitor.p >= self.active_at

    def base(self, a: Active) -> int:
        return a.requested if self.policy == "uniform" else min(a.requested, a.tier_budget)

    def weight(self, a: Active) -> float:
        return WEIGHT[a.priority] if self.policy == "brownout" else 1.0

    def target(self, a: Active) -> int:
        frac = 1 - min(self.max_frac, self.monitor.p * self.weight(a))
        return max(int(self.base(a) * frac), a.used + self.min_left)

    def start(self, a: Active) -> Active:
        if self.cutting:
            a.limit = min(a.requested, max(int(self.base(a) * (1 - min(self.max_frac, self.monitor.p * self.weight(a)))),
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
        return {"policy": self.policy, "notice": self.notice, "cutting": self.cutting,
                "in_flight": len(self.active), "cuts": self.cuts}
