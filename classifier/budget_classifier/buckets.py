from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

DEFAULT_EDGES = (0, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _fmt(n: int) -> str:
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


@dataclass(frozen=True)
class Buckets:
    edges: tuple[int, ...] = DEFAULT_EDGES

    def __post_init__(self) -> None:
        e = tuple(int(x) for x in self.edges)
        object.__setattr__(self, "edges", e)
        if not 2 <= len(e) - 1 <= 10:
            raise ValueError("need between 2 and 10 buckets (Jev score questions take at most 10 levels)")
        if e[0] != 0 or any(b <= a for a, b in zip(e, e[1:])):
            raise ValueError("edges must start at 0 and strictly increase")

    @property
    def n(self) -> int:
        return len(self.edges) - 1

    def index(self, tokens: int) -> int:
        return min(max(bisect_right(self.edges, max(int(tokens), 0)) - 1, 0), self.n - 1)

    def budget(self, i: int) -> int:
        return self.edges[i + 1]

    def label(self, i: int) -> str:
        lo, hi = self.edges[i], self.edges[i + 1]
        return f"{_fmt(lo)}+" if i == self.n - 1 else f"{_fmt(lo)}-{_fmt(hi)}"

    def labels(self) -> list[str]:
        return [self.label(i) for i in range(self.n)]
