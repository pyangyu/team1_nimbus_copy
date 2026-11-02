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
    fin = [w for w in workflows if not w.get("cancelled")]
    lat = [w["latency_s"] for w in fin]
    dl = [w for w in workflows if w.get("met_deadline") is not None]
    out = {
        "n": n,
        "accuracy": mean(w["correct"] for w in workflows),
        "errors": sum(1 for w in workflows if w.get("error") and not w.get("rejected") and not w.get("cancelled")),
        "rejected": sum(1 for w in workflows if w.get("rejected")),
        "cancelled": n - len(fin),
        "latency_p50_s": pct(lat, 50),
        "latency_p90_s": pct(lat, 90),
        "latency_p95_s": pct(lat, 95),
        "latency_p99_s": pct(lat, 99),
    }
    if fin:
        out.update({
            "rounds_mean": mean(w["rounds"] for w in fin),
            "llm_calls_mean": mean(w["llm_calls"] for w in fin),
            "prompt_tokens_mean": mean(w["prompt_tokens"] for w in fin),
            "thinking_tokens_mean": mean(w["thinking_tokens"] for w in fin),
            "answer_tokens_mean": mean(w["answer_tokens"] for w in fin),
        })
    if dl:
        out["deadline_met"] = mean(w["met_deadline"] for w in dl)
        out["correct_and_on_time"] = mean(w["met_deadline"] and w["correct"] for w in dl)
        adm = [w for w in dl if not w.get("rejected")]
        if adm and len(adm) < len(dl):
            out["admitted_correct_and_on_time"] = mean(w["met_deadline"] and w["correct"] for w in adm)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


BREAKDOWN = ("n", "accuracy", "deadline_met", "correct_and_on_time", "latency_p50_s", "latency_p95_s",
             "thinking_tokens_mean", "cancelled")


def breakdown(workflows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for w in workflows:
        groups.setdefault(str(w.get(key)), []).append(w)
    out = {}
    for k in sorted(groups):
        s = summarize(groups[k])
        out[k] = {c: s[c] for c in BREAKDOWN if c in s}
    return out


def pressure_summary(samples: list, phase_of=None) -> dict:
    if not samples:
        return {"p_samples": 0}
    n = len(samples)
    hist = [0] * 10
    for s in samples:
        hist[min(int(s[1] * 10), 9)] += 1
    out = {"p_samples": n, "p_mean": round(mean(s[1] for s in samples), 4),
           "p_at_cap": round(sum(s[1] >= 1.0 for s in samples) / n, 4),
           "p_zero": round(sum(s[1] == 0 for s in samples) / n, 4),
           "queue_frac": round(sum(s[2] > 0 for s in samples) / n, 4),
           "p_hist_deciles": [round(h / n, 4) for h in hist]}
    tpot = [s[4] for s in samples if len(s) > 4 and s[4] is not None]
    if tpot:
        out["tpot_ms_mean"] = round(mean(tpot), 1)
    if phase_of is not None:
        groups: dict[str, list] = {}
        for s in samples:
            groups.setdefault(phase_of(s[0]), []).append(s)
        out["by_phase"] = {k: {"p_samples": len(g), "p_mean": round(mean(s[1] for s in g), 4),
                               "p_at_cap": round(sum(s[1] >= 1.0 for s in g) / len(g), 4)}
                           for k, g in sorted(groups.items())}
    return out
