"""Workload specs, per-problem budgets and rate profiles for the load generator.

A workload is a comma list of `dataset[:levels][=weight]`, for example `math500:1-5`
or `gsm8k=0.5,math500:4-5=0.5`. Levels are one MATH level or a range `lo-hi`.

A profile is a comma list of `duration_s:rate`, the rate in units of capacity C,
for example `180:0.7,120:1.5`. Arrivals are Poisson within each segment.

A split keeps half of a pool, by a hash of the problem id, so two runs can use disjoint problems:
the spike suite warms the length predictor on split b and measures on split a.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .datasets import Example
from .datasets import math as mathsets

DEFAULT_WORKLOAD = "math500:1-5"
DEFAULT_PROFILE = "180:0.7,120:1.2,180:0.7,180:1.5,240:0.7,120:2.0,180:0.7,240:1.2,300:0.7"
LEVEL_BUDGETS = {1: (2048, 4096, 150.0), 2: (2048, 4096, 150.0), 3: (4096, 6144, 300.0),
                 4: (6144, 6144, 450.0), 5: (6144, 6144, 450.0)}
GSM8K_BUDGET = (1024, 2048, 90.0)
HARD_BUDGET = (6144, 6144, 450.0)
SPLITS = ("a", "b")


@dataclass(frozen=True)
class Component:
    dataset: str
    levels: tuple[int, ...] | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class Segment:
    duration_s: float
    rate: float

    @property
    def phase(self) -> str:
        return "burst" if self.rate > 1.0 else "baseline"


def parse_levels(text: str) -> tuple[int, ...]:
    lo, _, hi = text.partition("-")
    lo_i, hi_i = int(lo), int(hi or lo)
    if not 1 <= lo_i <= hi_i <= 5:
        raise ValueError(f"bad MATH levels {text!r}, expected 1-5 or a range like 2-4")
    return tuple(range(lo_i, hi_i + 1))


def parse_workload(spec: str) -> list[Component]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition("=")
        ds, _, levels = name.partition(":")
        if ds not in mathsets.DATASETS:
            raise ValueError(f"unknown dataset {ds!r} in workload {spec!r}, expected one of {mathsets.DATASETS}")
        if levels and ds != "math500":
            raise ValueError(f"levels apply to math500 only, got {part!r}")
        w = float(weight) if weight else 1.0
        if w <= 0:
            raise ValueError(f"weight must be positive in {part!r}")
        out.append(Component(ds, parse_levels(levels) if levels else None, w))
    if not out:
        raise ValueError("empty workload spec")
    return out


def pools(components: list[Component]) -> list[list[Example]]:
    out = []
    for c in components:
        pool = mathsets.load(c.dataset, list(c.levels) if c.levels else None)
        if not pool:
            raise ValueError(f"no problems for {c}")
        out.append(pool)
    return out


def split(pool: list[Example], part: str) -> list[Example]:
    if part not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {part!r}")
    return [e for e in pool if SPLITS[hashlib.sha256(e.id.encode()).digest()[0] & 1] == part]


class Sampler:
    def __init__(self, components: list[Component], pools_: list[list[Example]], seed: int | str):
        self.components, self.pools = components, pools_
        self.weights = [c.weight for c in components]
        self.rng = random.Random(seed)

    def draw(self) -> Example:
        if len(self.pools) == 1:
            return self.rng.choice(self.pools[0])
        k = self.rng.choices(range(len(self.pools)), weights=self.weights)[0]
        return self.rng.choice(self.pools[k])


def band(ex: Example) -> str:
    if ex.dataset == "gsm8k":
        return "gsm8k"
    if ex.level:
        return f"L{ex.level}"
    return "aime"


def budget_for(ex: Example) -> tuple[int, int, float]:
    if ex.dataset == "gsm8k":
        return GSM8K_BUDGET
    return LEVEL_BUDGETS.get(ex.level, HARD_BUDGET)


def parse_profile(spec: str) -> list[Segment]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        d, sep, r = part.partition(":")
        if not sep:
            raise ValueError(f"profile segment {part!r} must be duration_s:rate")
        seg = Segment(float(d), float(r))
        if seg.duration_s <= 0 or seg.rate < 0:
            raise ValueError(f"bad profile segment {part!r}")
        out.append(seg)
    if not out:
        raise ValueError("empty profile")
    return out


def total_s(profile: list[Segment]) -> float:
    return sum(s.duration_s for s in profile)


def arrivals(profile: list[Segment], capacity: float, seed: int | str) -> list[tuple[float, int]]:
    rng = random.Random(f"{seed}-arrivals")
    out, start = [], 0.0
    for k, s in enumerate(profile):
        rate, end, t = s.rate * capacity, start + s.duration_s, start
        while rate > 0:
            t += rng.expovariate(rate)
            if t >= end:
                break
            out.append((t, k))
        start = end
    return out


def segment_at(profile: list[Segment], offset: float) -> int | None:
    if offset < 0:
        return None
    start = 0.0
    for k, s in enumerate(profile):
        start += s.duration_s
        if offset < start:
            return k
    return None


def phase_at(profile: list[Segment], offset: float) -> str:
    k = segment_at(profile, offset)
    return "tail" if k is None else profile[k].phase
