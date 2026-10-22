from __future__ import annotations

import json
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .buckets import Buckets
from .features import featurize
from .jev import JevPrior
from .memory import ExampleMemory
from .online import OnlineModel


@dataclass
class Prediction:
    request_id: str
    bucket: int
    label: str
    budget: int
    confidence: float
    source: str
    shadow: bool
    model: str | None
    probs: list[float]
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Pending:
    x: dict[int, float]
    prompt: str
    bucket: int
    model: str | None


@dataclass
class VersionStats:
    labels: int = 0
    censored: int = 0
    window: deque = field(default_factory=lambda: deque(maxlen=200))

    def summary(self) -> dict:
        w = list(self.window)
        n = len(w)
        return {
            "labels": self.labels,
            "censored": self.censored,
            "window": n,
            "exact": sum(p == t for p, t in w) / n if n else None,
            "within_one": sum(abs(p - t) <= 1 for p, t in w) / n if n else None,
            "under": sum(p < t for p, t in w) / n if n else None,
            "mean_abs_error": sum(abs(p - t) for p, t in w) / n if n else None,
        }


class BudgetClassifier:
    def __init__(self, buckets: Buckets | None = None, *, dim: int = 1 << 18, lr: float = 0.5,
                 spread: float = 0.5, jev: JevPrior | None = None, warmup: int = 200,
                 jev_until: float = 0.95, examples: int = 4, quantile: float = 0.5,
                 censored: str = "update", active_at: float = 0.05, window: int = 200,
                 max_pending: int = 100_000, reset_on_upgrade: bool = True):
        if censored not in ("update", "skip"):
            raise ValueError("censored must be 'update' or 'skip'")
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.buckets = buckets or Buckets()
        self.dim = dim
        self.online = OnlineModel(self.buckets.n, dim=dim, lr=lr, spread=spread)
        self.memory = ExampleMemory()
        self.jev = jev
        self.warmup, self.jev_until, self.examples = warmup, jev_until, examples
        self.quantile, self.censored, self.active_at = quantile, censored, active_at
        self.window, self.max_pending = window, max_pending
        self.reset_on_upgrade = reset_on_upgrade
        self.pending: OrderedDict[str, _Pending] = OrderedDict()
        self.versions: dict[str | None, VersionStats] = {}
        self.current_model: str | None = None
        self.upgrades = 0
        self.pressure = 0.0

    @property
    def shadow(self) -> bool:
        return self.pressure < self.active_at

    def set_pressure(self, p: float) -> None:
        self.pressure = max(0.0, float(p))

    def _version(self, model: str | None) -> VersionStats:
        if model not in self.versions:
            self.versions[model] = VersionStats(window=deque(maxlen=self.window))
        return self.versions[model]

    def _switch(self, model: str | None) -> None:
        if model is None or model == self.current_model:
            return
        if self.current_model is not None:
            self.upgrades += 1
            if self.reset_on_upgrade:
                self.online.reset_optimizer()
        self.current_model = model
        self._version(model)

    def _pick(self, p: np.ndarray) -> int:
        return int(min(np.searchsorted(np.cumsum(p), self.quantile - 1e-9), self.buckets.n - 1))

    async def predict(self, request_id: str, prompt: str, model: str | None = None,
                      extra_questions: dict | None = None) -> Prediction:
        self._switch(model)
        model = self.current_model
        x = featurize(prompt, self.dim)
        p = self.online.proba(x)
        n = self._version(model).labels
        w = n / (n + self.warmup) if self.warmup > 0 else 1.0
        source = "online" if n else "cold"
        extra: dict = {}
        if self.jev is not None and w < self.jev_until:
            pj, extra = await self.jev.ask_with(prompt, self.memory.similar(x, model, self.examples), extra_questions)
            if pj is not None:
                p = w * p + (1 - w) * pj
                source = "blend" if n else "jev"
        k = self._pick(p)
        self.pending[request_id] = _Pending(x, prompt, k, model)
        self.pending.move_to_end(request_id)
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
        return Prediction(request_id, k, self.buckets.label(k), self.buckets.budget(k), round(float(p[k]), 4),
                          source, self.shadow, model, [round(float(v), 4) for v in p], extra)

    def observe(self, request_id: str, thinking_tokens: int, *, finished: bool = True,
                model: str | None = None, prompt: str | None = None) -> dict:
        pend = self.pending.pop(request_id, None)
        if pend is None and prompt is None:
            return {"request_id": request_id, "used": False, "reason": "unknown request and no prompt"}
        x = pend.x if pend else featurize(prompt, self.dim)
        text = pend.prompt if pend else prompt
        model = model or (pend.model if pend else None) or self.current_model
        self._switch(model)
        k = self.buckets.index(thinking_tokens)
        stats = self._version(model)
        if finished:
            self.online.update_exact(x, k)
            self.memory.add(x, text, int(thinking_tokens), k, model)
            stats.labels += 1
            if pend is not None:
                stats.window.append((pend.bucket, k))
            kind = "exact"
        else:
            stats.censored += 1
            if self.censored == "skip":
                return {"request_id": request_id, "used": False, "reason": "censored", "bucket": k}
            self.online.update_at_least(x, k)
            kind = "at_least"
        return {"request_id": request_id, "used": True, "label": kind, "bucket": k,
                "predicted": pend.bucket if pend else None}

    def stats(self) -> dict:
        return {
            "buckets": self.buckets.labels(),
            "current_model": self.current_model,
            "upgrades": self.upgrades,
            "pressure": self.pressure,
            "shadow": self.shadow,
            "pending": len(self.pending),
            "updates": self.online.updates,
            "jev": None if self.jev is None else {"calls": self.jev.calls, "failures": self.jev.failures,
                                                    "skipped_busy": self.jev.skipped_busy,
                                                    "last_error": self.jev.last_error},
            "versions": {str(m): v.summary() for m, v in self.versions.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"edges": list(self.buckets.edges), "dim": self.dim, "current_model": self.current_model,
                "upgrades": self.upgrades, "updates": self.online.updates,
                "versions": {json.dumps(m): {"labels": v.labels, "censored": v.censored}
                             for m, v in self.versions.items()}}
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, W=self.online.W, G=self.online.G, meta=json.dumps(meta))
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        with np.load(Path(path)) as z:
            meta = json.loads(str(z["meta"]))
            if tuple(meta["edges"]) != self.buckets.edges or meta["dim"] != self.dim:
                raise ValueError("saved state has different buckets or feature size")
            self.online.W[:] = z["W"]
            self.online.G[:] = z["G"]
        self.online.updates = meta["updates"]
        self.current_model, self.upgrades = meta["current_model"], meta["upgrades"]
        self.versions = {}
        for m, v in meta["versions"].items():
            s = self._version(json.loads(m))
            s.labels, s.censored = v["labels"], v["censored"]
