from __future__ import annotations

import asyncio
import re
import time
from collections import deque

import httpx

METRIC = re.compile(r"^(vllm:num_requests_(?:waiting|running))(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)
GENERATED = re.compile(r"^vllm:generation_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)


def parse_metrics(text: str) -> tuple[float, float]:
    vals = {"vllm:num_requests_waiting": 0.0, "vllm:num_requests_running": 0.0}
    for name, v in METRIC.findall(text):
        vals[name] += float(v)
    return vals["vllm:num_requests_waiting"], vals["vllm:num_requests_running"]


def parse_generated(text: str) -> float | None:
    vals = GENERATED.findall(text)
    return sum(float(v) for v in vals) if vals else None


def pressure(waiting: float, running: float) -> float:
    return min(1.0, waiting / max(running, 1.0))


def slowdown(tpot: float | None, slo: float | None) -> float:
    if not slo or tpot is None:
        return 0.0
    return min(1.0, max(0.0, (tpot - slo) / slo))


class PressureMonitor:
    def __init__(self, metrics_url: str | None = None, *, interval: float = 0.5,
                 http: httpx.AsyncClient | None = None, history: int = 50_000, clock=time.time,
                 tpot_slo: float | None = None, rate_window: float = 5.0):
        self.metrics_url, self.interval, self.http, self.clock = metrics_url, interval, http, clock
        self.tpot_slo, self.rate_window = tpot_slo, rate_window
        self.waiting = self.running = 0.0
        self.tpot: float | None = None
        self.generated: deque[tuple[float, float, float]] = deque()
        self.p = 0.0
        self.errors = 0
        self.samples = 0
        self.hist = [0] * 10
        self.series: deque[tuple[float, float, float, float, float | None]] = deque(maxlen=history)

    def set(self, waiting: float, running: float, tpot: float | None = None) -> None:
        self.waiting, self.running, self.tpot = float(waiting), float(running), tpot
        self.p = max(pressure(self.waiting, self.running), slowdown(tpot, self.tpot_slo))
        self.samples += 1
        self.hist[min(int(self.p * 10), 9)] += 1
        self.series.append((round(self.clock(), 3), round(self.p, 4), self.waiting, self.running,
                            None if tpot is None else round(tpot * 1000, 1)))

    def decode_time(self, t: float, generated: float, running: float) -> float | None:
        if self.generated and generated < self.generated[-1][1]:
            self.generated.clear()
        self.generated.append((t, generated, running))
        while len(self.generated) > 2 and t - self.generated[1][0] >= self.rate_window:
            self.generated.popleft()
        t0, g0, _ = self.generated[0]
        if t <= t0 or generated <= g0:
            return None
        busy = sum(x[2] for x in self.generated) / len(self.generated)
        return busy / ((generated - g0) / (t - t0)) if busy >= 1 else None

    async def poll_once(self) -> None:
        try:
            r = await self.http.get(self.metrics_url, timeout=2.0)
            r.raise_for_status()
            waiting, running = parse_metrics(r.text)
            generated = parse_generated(r.text)
            self.set(waiting, running, None if generated is None else self.decode_time(self.clock(), generated, running))
        except Exception:
            self.errors += 1

    async def run(self) -> None:
        if not self.metrics_url or self.http is None:
            return
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval)

    def since(self, t: float) -> list[tuple[float, float, float, float, float | None]]:
        return [x for x in self.series if x[0] >= t]

    def stats(self) -> dict:
        return {"p": round(self.p, 3), "waiting": self.waiting, "running": self.running,
                "tpot_ms": None if self.tpot is None else round(self.tpot * 1000, 1),
                "tpot_slo_ms": None if self.tpot_slo is None else round(self.tpot_slo * 1000, 1),
                "samples": self.samples, "errors": self.errors, "p_hist_deciles": self.hist}
