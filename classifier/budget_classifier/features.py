from __future__ import annotations

import math
import re
import zlib
from collections import Counter

TOKEN = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
BIAS = 0


def _h(s: str, dim: int) -> int:
    return 1 + zlib.crc32(s.encode()) % (dim - 1)


def featurize(text: str, dim: int = 1 << 18) -> dict[int, float]:
    toks = ["<num>" if t.isdigit() else t for t in TOKEN.findall(text.lower())]
    c: Counter[int] = Counter()
    for t in toks:
        c[_h("u:" + t, dim)] += 1
    for a, b in zip(toks, toks[1:]):
        c[_h(f"b:{a} {b}", dim)] += 1
    c[_h(f"len:{int(math.log2(len(text) + 1))}", dim)] += 4
    c[_h(f"words:{int(math.log2(len(toks) + 1))}", dim)] += 4
    c[_h(f"digits:{min(sum(t == '<num>' for t in toks), 20) // 4}", dim)] += 2
    x = {i: math.log1p(v) for i, v in c.items()}
    norm = math.sqrt(sum(v * v for v in x.values())) or 1.0
    x = {i: v / norm for i, v in x.items()}
    x[BIAS] = 1.0
    return x


def dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(i, 0.0) for i, v in a.items() if i != BIAS)
