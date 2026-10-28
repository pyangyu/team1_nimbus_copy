from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass

from budget_classifier import Buckets

from .pressure import PressureMonitor
from .priority import WEIGHT

POLICIES = ("none", "reject", "defer", "uniform", "brownout")
CUTTING = ("uniform", "brownout")
NOTICES = ("silent", "announced")
PREDICTORS = ("jev", "jev-serial", "online", "oracle")
JEV_PREDICTORS = ("jev", "jev-serial")
MODES = ("proportional", "ranked")


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
    probs: list[float] | None = None
    probs_source: str = "none"

    def set_priority(self, level: str, source: str) -> None:
        self.priority, self.priority_source = level, source

    def set_probs(self, probs: list[float] | None, source: str) -> None:
        if probs:
            self.probs, self.probs_source = list(probs), source


class Controller:
    """Lowers the thinking limits of running requests while the queue backs up.

    proportional: every request's limit becomes requested x (1 - p x weight), floored at used + min_left.
    ranked: the same total number of tokens is taken away, but from the requests where it is expected to
    cost least. Each request carries a distribution over the thinking budget it needs (from the predictor);
    the tokens in bucket k cost p_k x (fraction of the bucket removed), and the cheapest slices go first.
    """

    def __init__(self, monitor: PressureMonitor, *, policy: str = "brownout", notice: str = "silent",
                 predictor: str = "jev", mode: str = "proportional", buckets: Buckets | None = None,
                 min_left: int = 64, max_frac: float = 0.9, active_at: float = 0.05, interval: float = 0.25):
        self.monitor = monitor
        self.buckets = buckets or Buckets()
        self.set_policy(policy, notice, predictor, mode)
        self.min_left, self.max_frac, self.active_at, self.interval = min_left, max_frac, active_at, interval
        self.active: dict[str, Active] = {}
        self.cuts = 0
        self.ranked = {"cuts_with_probs": 0, "cuts_flat": 0, "ticks": 0, "shed_tokens": 0}

    def set_policy(self, policy: str, notice: str | None = None, predictor: str | None = None,
                   mode: str | None = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if notice is not None and notice not in NOTICES:
            raise ValueError(f"notice must be one of {NOTICES}")
        if predictor is not None and predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {PREDICTORS}")
        if mode is not None and mode not in MODES:
            raise ValueError(f"cut mode must be one of {MODES}")
        self.policy = policy
        if notice is not None:
            self.notice = notice
        if predictor is not None:
            self.predictor = predictor
        if mode is not None:
            self.mode = mode

    @property
    def cutting(self) -> bool:
        return self.policy in CUTTING and self.monitor.p >= self.active_at

    @property
    def ranking(self) -> bool:
        return self.policy == "brownout" and self.mode == "ranked"

    def base(self, a: Active) -> int:
        return a.requested if self.policy == "uniform" or self.ranking else min(a.requested, a.tier_budget)

    def weight(self, a: Active) -> float:
        return WEIGHT[a.priority] if self.policy == "brownout" else 1.0

    def frac(self, a: Active) -> float:
        return min(self.max_frac, self.monitor.p * self.weight(a))

    def target(self, a: Active) -> int:
        return max(int(self.base(a) * (1 - self.frac(a))), a.used + self.min_left)

    def start(self, a: Active) -> Active:
        if self.cutting and not self.ranking:
            a.limit = min(a.requested, max(int(self.base(a) * (1 - self.frac(a))), self.min_left))
        else:
            a.limit = a.requested
        self.active[a.rid] = a
        return a

    def finish(self, a: Active) -> None:
        self.active.pop(a.rid, None)

    # ranked mode

    def shed_target(self) -> float:
        """Tokens the proportional rule would take from the requested budgets at this pressure."""
        return sum(a.requested * self.frac(a) for a in self.active.values())

    def conditional(self, a: Active) -> list[float]:
        """The request's distribution over needed-budget buckets, given what it has already used.

        Buckets the request has already passed get no mass. Mass above the client's own budget is folded
        into the bucket that holds that budget: a request that needs more than it asked for still loses
        from every token taken below its budget.
        """
        n, edges = self.buckets.n, self.buckets.edges
        alive = [edges[i + 1] > a.used for i in range(n)]
        if not any(alive):
            return [0.0] * n
        p = list(a.probs) if a.probs is not None and len(a.probs) == n else None
        if p is not None:
            top = self.buckets.index(max(a.requested - 1, 0))
            p[top] += sum(p[top + 1:])
            for i in range(top + 1, n):
                p[i] = 0.0
        mass = sum(p[i] for i in range(n) if alive[i]) if p is not None else 0.0
        if p is None or mass <= 1e-9:
            share = 1.0 / sum(alive)
            return [share if alive[i] else 0.0 for i in range(n)]
        return [p[i] / mass if alive[i] else 0.0 for i in range(n)]

    def _slice(self, a: Active, cond: list[float], k: int, cur: int) -> tuple[float, int, int, int] | None:
        """The topmost removable slice at or below bucket k, as (density, k, lo, hi), or None."""
        edges, floor = self.buckets.edges, a.used + self.min_left
        while k >= 0:
            lo, hi = max(edges[k], floor), min(edges[k + 1], cur)
            if edges[k] >= cur:
                k -= 1
                continue
            if hi <= lo:
                return None
            width = edges[k + 1] - edges[k]
            return cond[k] / width / self.weight(a), k, lo, hi
        return None

    def rank_targets(self) -> dict[str, int]:
        acts = list(self.active.values())
        need = self.shed_target() - sum(a.requested - a.limit for a in acts)
        if need <= 0:
            return {}
        heap: list[tuple[float, int, str, int, int, int]] = []
        cond: dict[str, list[float]] = {}
        seq = 0
        for a in acts:
            cond[a.rid] = self.conditional(a)
            s = self._slice(a, cond[a.rid], self.buckets.index(max(a.limit - 1, 0)), a.limit)
            if s is not None:
                heapq.heappush(heap, (s[0], seq, a.rid, s[1], s[2], s[3]))
                seq += 1
        targets: dict[str, int] = {}
        while need > 0 and heap:
            _, _, rid, k, lo, hi = heapq.heappop(heap)
            take = min(hi - lo, need)
            targets[rid] = int(hi - take)
            need -= take
            if take >= hi - lo and k > 0:
                a = self.active[rid]
                s = self._slice(a, cond[rid], k - 1, lo)
                if s is not None:
                    heapq.heappush(heap, (s[0], seq, rid, s[1], s[2], s[3]))
                    seq += 1
        return targets

    def tick(self) -> None:
        if not self.cutting:
            return
        if self.ranking:
            self.ranked["ticks"] += 1
            targets = self.rank_targets()
        else:
            targets = {a.rid: self.target(a) for a in self.active.values()}
        for rid, t in targets.items():
            a = self.active.get(rid)
            if a is None or t >= a.limit:
                continue
            if a.cut_from is None:
                a.cut_from = a.limit
                self.cuts += 1
                if self.ranking:
                    self.ranked["cuts_with_probs" if a.probs is not None else "cuts_flat"] += 1
            if self.ranking:
                self.ranked["shed_tokens"] += a.limit - t
            a.limit = t

    async def run(self) -> None:
        while True:
            self.tick()
            await asyncio.sleep(self.interval)

    def stats(self) -> dict:
        return {"policy": self.policy, "notice": self.notice, "predictor": self.predictor, "mode": self.mode,
                "cutting": self.cutting, "in_flight": len(self.active), "cuts": self.cuts, "ranked": dict(self.ranked)}
