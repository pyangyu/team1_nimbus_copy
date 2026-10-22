from __future__ import annotations

import asyncio
import re

import httpx

METRIC = re.compile(r"^(vllm:num_requests_(?:waiting|running))(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)


def parse_metrics(text: str) -> tuple[float, float]:
    vals = {"vllm:num_requests_waiting": 0.0, "vllm:num_requests_running": 0.0}
    for name, v in METRIC.findall(text):
        vals[name] += float(v)
    return vals["vllm:num_requests_waiting"], vals["vllm:num_requests_running"]


def pressure(waiting: float, running: float) -> float:
    return min(1.0, waiting / max(running, 1.0))


class PressureMonitor:
    def __init__(self, metrics_url: str | None = None, *, interval: float = 0.5,
                 http: httpx.AsyncClient | None = None):
        self.metrics_url, self.interval, self.http = metrics_url, interval, http
        self.waiting = self.running = 0.0
        self.p = 0.0
        self.errors = 0
        self.samples = 0
        self.hist = [0] * 10

    def set(self, waiting: float, running: float) -> None:
        self.waiting, self.running = float(waiting), float(running)
        self.p = pressure(self.waiting, self.running)
        self.samples += 1
        self.hist[min(int(self.p * 10), 9)] += 1

    async def poll_once(self) -> None:
        try:
            r = await self.http.get(self.metrics_url, timeout=2.0)
            r.raise_for_status()
            self.set(*parse_metrics(r.text))
        except Exception:
            self.errors += 1

    async def run(self) -> None:
        if not self.metrics_url or self.http is None:
            return
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval)

    def stats(self) -> dict:
        return {"p": round(self.p, 3), "waiting": self.waiting, "running": self.running,
                "samples": self.samples, "errors": self.errors, "p_hist_deciles": self.hist}
