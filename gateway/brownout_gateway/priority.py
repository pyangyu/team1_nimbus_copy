from __future__ import annotations

import asyncio
import time

LEVELS = ("low", "normal", "high")
WEIGHT = {"low": 1.5, "normal": 1.0, "high": 0.5}
CRITERIA = [
    "low: cut this request first; it has slack or is a late retry",
    "normal: cut this request like any other",
    "high: protect this request; a user is waiting on it or its deadline is close",
]


def heuristic(cls: str, step: str, deadline_ms: int | None, now: float | None = None,
              tight_s: float = 20.0) -> str:
    now = time.time() if now is None else now
    if cls == "interactive":
        return "high"
    if deadline_ms is not None and deadline_ms / 1000 - now < tight_s:
        return "high"
    rnd = step.rpartition(".")[2]
    if rnd.isdigit() and int(rnd) >= 1:
        return "low"
    return "normal"


def question(cls: str, step: str, deadline_ms: int | None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    slack = None if deadline_ms is None else round(deadline_ms / 1000 - now, 1)
    return {
        "type": "score",
        "instructions": (
            "The server is overloaded and must shorten some requests' thinking. How strongly should "
            f"this request be protected from cuts? Traffic class: {cls}. Workflow step: {step}. "
            f"Seconds left before its deadline: {slack if slack is not None else 'none'}."
        ),
        "criteria": CRITERIA,
    }


def from_answer(answer: dict | None) -> str | None:
    try:
        probs = answer["probabilities"]
        return LEVELS[max(range(3), key=lambda i: float(probs[str(i)]))]
    except Exception:
        return None


class PriorityRefiner:
    def __init__(self, jev=None, *, max_inflight: int = 1):
        self.jev = jev
        self.max_inflight = max_inflight
        self.inflight = 0
        self.asked = self.answered = self.dropped = 0

    def refine(self, target, cls: str, step: str, deadline_ms: int | None, prompt: str) -> None:
        if self.jev is None:
            return
        if self.inflight >= self.max_inflight:
            self.dropped += 1
            return
        self.inflight += 1
        asyncio.get_running_loop().create_task(self._ask(target, cls, step, deadline_ms, prompt))

    async def _ask(self, target, cls, step, deadline_ms, prompt) -> None:
        self.asked += 1
        try:
            answers = await self.jev.post({"request": {"prompt_head": prompt[:1500]}},
                                          {"priority": question(cls, step, deadline_ms)})
        finally:
            self.inflight -= 1
        level = from_answer((answers or {}).get("priority"))
        if level is not None:
            self.answered += 1
            target.set_priority(level, "jev")

    def stats(self) -> dict:
        return {"asked": self.asked, "answered": self.answered, "dropped_busy": self.dropped}
