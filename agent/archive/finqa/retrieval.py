"""BM25 over one filing's chunks. Local, no LLM call."""
from __future__ import annotations

import math
import re
from collections import Counter

from .datasets import Chunk

_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set("the of and a an in to for on by is are was were be as at or from with that this "
            "what which how did does do its it their".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.tf for w in t)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int, exclude: set[str] = frozenset()) -> list[Chunk]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            if self.chunks[i].id in exclude:
                continue
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


class Retriever:
    """Caches one BM25 index per filing."""

    def __init__(self, corpus: dict[str, list[Chunk]]):
        self.corpus = corpus
        self._idx: dict[str, BM25] = {}

    def search(self, doc: str, query: str, k: int) -> list[Chunk]:
        if doc not in self._idx:
            self._idx[doc] = BM25(self.corpus.get(doc, []))
        return self._idx[doc].search(query, k)
