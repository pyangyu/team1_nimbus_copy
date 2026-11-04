"""Arrival times replayed from the BurstGPT trace (HPMLL/BurstGPT, Azure OpenAI traffic at a campus).

`burstgpt_counts.json` holds per-second request counts for a bursty stretch of BurstGPT_1.csv (trace hours
226 onward, where the mean rate climbs to several requests per second and the variance-to-mean ratio of the
per-second counts reaches 5 to 9; the earlier days are near-Poisson). The load generator takes a window of n
consecutive arrivals from a seeded offset, spreads each second's arrivals uniformly within it, and rescales
time so the window's mean rate equals the requested rate. The clustering of the real trace is kept; only the
clock speed changes.

Measured on this window: the gaps between 32 to 64 consecutive arrivals have a coefficient of variation of
about 1.0 on average (Poisson-like) and 1.3 to 1.4 at the 90th percentile. The trace's burstiness is in how
its rate moves over minutes and hours (the hourly rate swings from 0.3/s to 9/s here), not in the spacing of
a few dozen arrivals, so a replayed window gives real timing rather than a harsher burst than Poisson.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_counts(path: str | None = None) -> list[int]:
    p = Path(path) if path else Path(__file__).with_name("burstgpt_counts.json")
    return json.loads(p.read_text(encoding="utf-8"))["counts"]


def window(counts: list[int], n: int, rng: random.Random) -> list[float]:
    """Raw arrival times (trace seconds, jittered within the second) of n consecutive requests from a random offset."""
    total = sum(counts)
    if n > total:
        raise ValueError(f"the trace window holds {total} arrivals, fewer than {n}")
    start = rng.randrange(0, total - n + 1)  # the index of the first arrival among all arrivals in the window
    times: list[float] = []
    seen = 0
    for sec, c in enumerate(counts):
        if seen + c <= start:
            seen += c
            continue
        for _ in range(c):
            if seen >= start:
                times.append(sec + rng.random())
            seen += 1
            if len(times) == n:
                return sorted(times)
    return sorted(times)


def arrivals(n: int, rate: float, seed: int, path: str | None = None) -> list[float]:
    """Offsets in seconds from the first arrival for n workflows at a mean of `rate` per second."""
    rng = random.Random(seed)
    if n <= 1:
        return [0.0] * n
    t = window(load_counts(path), n, rng)
    span = t[-1] - t[0]
    target = (n - 1) / rate
    scale = target / span if span > 0 else 0.0
    return [(x - t[0]) * scale for x in t]


def burstiness(offsets: list[float]) -> float | None:
    """Coefficient of variation of the gaps: 1 for Poisson, higher for bursts."""
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    if len(gaps) < 2:
        return None
    m = sum(gaps) / len(gaps)
    if m <= 0:
        return None
    var = sum((g - m) ** 2 for g in gaps) / len(gaps)
    return var ** 0.5 / m
