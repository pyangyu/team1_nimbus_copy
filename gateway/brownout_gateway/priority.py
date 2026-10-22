from __future__ import annotations

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
