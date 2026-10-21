from __future__ import annotations

import numpy as np


class OnlineModel:
    def __init__(self, n: int, dim: int = 1 << 18, lr: float = 0.5, spread: float = 0.5):
        self.n, self.dim, self.lr, self.spread = n, dim, lr, spread
        self.W = np.zeros((n, dim), dtype=np.float32)
        self.G = np.zeros((n, dim), dtype=np.float32)
        self.updates = 0

    def _xv(self, x: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        return np.fromiter(x.keys(), dtype=np.int64, count=len(x)), np.fromiter(x.values(), dtype=np.float32, count=len(x))

    def proba(self, x: dict[int, float]) -> np.ndarray:
        idx, val = self._xv(x)
        z = self.W[:, idx] @ val
        z = np.exp(z - z.max())
        return z / z.sum()

    def target(self, k: int) -> np.ndarray:
        t = np.exp(-np.abs(np.arange(self.n) - k) / self.spread)
        return t / t.sum()

    def update_exact(self, x: dict[int, float], k: int) -> None:
        self._step(x, self.proba(x) - self.target(k))

    def update_at_least(self, x: dict[int, float], k: int) -> None:
        if k <= 0:
            return
        p = self.proba(x)
        above = np.arange(self.n) >= k
        s = max(float(p[above].sum()), 1e-9)
        self._step(x, p - np.where(above, p / s, 0.0))

    def _step(self, x: dict[int, float], gz: np.ndarray) -> None:
        idx, val = self._xv(x)
        g = np.outer(gz.astype(np.float32), val)
        self.G[:, idx] += g * g
        self.W[:, idx] -= self.lr * g / (np.sqrt(self.G[:, idx]) + 1e-6)
        self.updates += 1

    def reset_optimizer(self) -> None:
        self.G.fill(0.0)
