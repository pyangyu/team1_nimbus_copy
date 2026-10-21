from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


class Jsonl:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(workflows: list[dict]) -> dict:
    n = len(workflows)
    if not n:
        return {"n": 0}
    lat = [w["latency_s"] for w in workflows]
    dl = [w for w in workflows if w.get("met_deadline") is not None]
    out = {
        "n": n,
        "accuracy": mean(w["correct"] for w in workflows),
        "accuracy_lenient": mean(w["correct_lenient"] for w in workflows),
        "errors": sum(1 for w in workflows if w.get("error")),
        "latency_p50_s": pct(lat, 50),
        "latency_p90_s": pct(lat, 90),
        "latency_p99_s": pct(lat, 99),
        "rounds_mean": mean(w["rounds"] for w in workflows),
        "llm_calls_mean": mean(w["llm_calls"] for w in workflows),
        "prompt_tokens_mean": mean(w["prompt_tokens"] for w in workflows),
        "thinking_tokens_mean": mean(w["thinking_tokens"] for w in workflows),
        "answer_tokens_mean": mean(w["answer_tokens"] for w in workflows),
    }
    if dl:
        out["deadline_met"] = mean(w["met_deadline"] for w in dl)
        out["correct_and_on_time"] = mean(w["met_deadline"] and w["correct"] for w in dl)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}
