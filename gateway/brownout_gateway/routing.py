from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from .pressure import PressureMonitor


@dataclass
class Replica:
    url: str
    monitor: PressureMonitor = field(default_factory=PressureMonitor)
    routed: int = 0
    requests: int = 0
    think: int = 0
    mark: int = -1
    fresh: int = 0

    def load(self) -> float:
        return self.monitor.waiting + self.monitor.running + (self.fresh if self.mark == self.monitor.samples else 0)

    def stats(self) -> dict:
        return {"url": self.url, "routed": self.routed, "requests": self.requests, "think": self.think,
                "load": self.load(), "pressure": self.monitor.stats()}


class Router:
    def __init__(self, replicas: list[Replica], *, max_sticky: int = 100_000):
        if not replicas:
            raise ValueError("at least one upstream replica is needed")
        self.replicas, self.max_sticky = replicas, max_sticky
        self.next = 0
        self.sticky: OrderedDict[str, int] = OrderedDict()

    def pick(self) -> int:
        n = len(self.replicas)
        loads = [r.load() for r in self.replicas]
        low = min(loads)
        k = min((i for i in range(n) if loads[i] == low), key=lambda i: (i - self.next) % n)
        self.next = (k + 1) % n
        r = self.replicas[k]
        if r.mark != r.monitor.samples:
            r.mark, r.fresh = r.monitor.samples, 0
        r.fresh += 1
        r.routed += 1
        return k

    def route(self, key: str | None, new: bool) -> int:
        if key is not None and not new and key in self.sticky:
            self.sticky.move_to_end(key)
            return self.sticky[key]
        k = self.pick()
        if key is not None:
            self.sticky[key] = k
            self.sticky.move_to_end(key)
            while len(self.sticky) > self.max_sticky:
                self.sticky.popitem(last=False)
        return k

    def pressure_stats(self) -> dict:
        ms = [r.monitor.stats() for r in self.replicas]
        if len(ms) == 1:
            return ms[0]
        return {"p": max(m["p"] for m in ms), "waiting": sum(m["waiting"] for m in ms),
                "running": sum(m["running"] for m in ms), "samples": sum(m["samples"] for m in ms),
                "errors": sum(m["errors"] for m in ms),
                "p_hist_deciles": [sum(h) for h in zip(*(m["p_hist_deciles"] for m in ms))]}

    def stats(self) -> list[dict]:
        return [r.stats() for r in self.replicas]
