"""Trace-driven arrivals: replay a window of the BurstGPT trace (Azure ChatGPT and GPT-4 requests, KDD 25) with
MATH-500 problems woven into it.

The window [start_h, start_h + window_h) of the trace is compressed into duration_s and thinned at random to
load * C * duration_s arrivals, so the trace's burst shape is kept at a mean rate of load x C. Each kept request
becomes a MATH problem by rank: requests are ordered by their response length in the trace and split into MATH
levels in the pool's level proportions, so the trace's long answers become the hard problems. Phases come from the
whole (unthinned) window in bin_s bins: a bin whose rate is above C is a burst.

The default window (day 21, 16:30-18:30, conversation log, 551 requests) has two bursts at 1.6x its mean, a lull
at 0.3x, a sharp burst at 2.1x and recovery. Its bursts carry the same response lengths as the rest of the window,
unlike windows whose bursts are scripted loops of 1-4 token replies, which would turn every burst into easy problems.
"""
from __future__ import annotations

import bisect
import csv
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx

from .datasets import Example
from .workload import Segment

TRACES = ("burstgpt",)
URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"
LOGS = {"conversation": ("Conversation log",), "api": ("API log",), "all": ("Conversation log", "API log")}
DEFAULT_LOG = "conversation"
DEFAULT_START_H = 520.5
DEFAULT_WINDOW_H = 2.0
DEFAULT_S = 1800.0
DEFAULT_LOAD = 0.9
DEFAULT_BIN_S = 120.0


@dataclass(frozen=True)
class Request:
    t: float
    model: str
    request_tokens: int
    response_tokens: int
    log: str


@dataclass(frozen=True)
class Replay:
    arrivals: list[tuple[float, int]]
    requests: list[Request]
    examples: list[Example]
    profile: list[Segment]
    window_n: int


def data_dir() -> Path:
    d = Path(os.environ.get("LIFELINE_DATA", Path(__file__).resolve().parents[1] / "data")) / "burstgpt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _csv() -> Path:
    path = data_dir() / "BurstGPT_1.csv"
    if not path.exists():
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        tmp.rename(path)
    return path


@lru_cache(maxsize=None)
def load(log: str = DEFAULT_LOG) -> list[Request]:
    if log not in LOGS:
        raise ValueError(f"unknown trace log {log!r}, expected one of {tuple(LOGS)}")
    out = []
    with open(_csv(), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Log Type"] in LOGS[log] and int(row["Response tokens"]) > 0:
                out.append(Request(float(row["Timestamp"]), row["Model"], int(row["Request tokens"]),
                                   int(row["Response tokens"]), row["Log Type"]))
    out.sort(key=lambda r: r.t)
    return out


def level_quotas(pool: list[Example], n: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for e in pool:
        counts[e.level] = counts.get(e.level, 0) + 1
    exact = {lv: n * c / len(pool) for lv, c in counts.items()}
    quotas = {lv: int(x) for lv, x in exact.items()}
    for lv in sorted(exact, key=lambda lv: (quotas[lv] - exact[lv], lv))[:n - sum(quotas.values())]:
        quotas[lv] += 1
    return quotas


def weave(kept: list[Request], pool: list[Example], rng: random.Random) -> list[Example]:
    quotas = level_quotas(pool, len(kept))
    order = sorted(range(len(kept)), key=lambda i: (kept[i].response_tokens, i))
    out: list[Example | None] = [None] * len(kept)
    pos = 0
    for lv in sorted(quotas):
        bag = [e for e in pool if e.level == lv]
        draws: list[Example] = []
        while len(draws) < quotas[lv]:
            draws += rng.sample(bag, len(bag))
        for e in draws[:quotas[lv]]:
            out[order[pos]] = e
            pos += 1
    return out


def replay(reqs: list[Request], pool: list[Example], capacity: float, *, start_h: float = DEFAULT_START_H,
           window_h: float = DEFAULT_WINDOW_H, duration_s: float = DEFAULT_S, load: float = DEFAULT_LOAD,
           bin_s: float = DEFAULT_BIN_S, seed: int | str = 0) -> Replay:
    if not pool:
        raise ValueError("empty problem pool")
    lo, span = start_h * 3600, window_h * 3600
    ts = [r.t for r in reqs]
    win = reqs[bisect.bisect_left(ts, lo):bisect.bisect_left(ts, lo + span)]
    n = round(load * capacity * duration_s)
    if n < 1:
        raise ValueError(f"load {load} x C {capacity} x {duration_s} s gives no arrivals")
    if n > len(win):
        raise ValueError(f"trace window has {len(win)} requests but the run needs {n}: "
                         "lower the load or widen the window")
    rng = random.Random(f"{seed}-trace")
    scale = duration_s / span
    kept = sorted(((win[i].t - lo + rng.random()) * scale, win[i]) for i in rng.sample(range(len(win)), n))
    nbins = math.ceil(duration_s / bin_s)
    counts = [0] * nbins
    for r in win:
        counts[min(int((r.t - lo) * scale // bin_s), nbins - 1)] += 1
    profile = []
    for k, c in enumerate(counts):
        d = min(bin_s, duration_s - k * bin_s)
        profile.append(Segment(d, round(c / len(win) * duration_s / d * load, 3)))
    arrivals = [(off, min(int(off // bin_s), nbins - 1)) for off, _ in kept]
    requests = [r for _, r in kept]
    return Replay(arrivals, requests, weave(requests, pool, rng), profile, len(win))


def spec(profile: list[Segment]) -> str:
    return ",".join(f"{s.duration_s:g}:{s.rate:g}" for s in profile)
