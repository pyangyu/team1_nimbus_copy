"""Surface features of a reasoning trace so far, and a logistic-regression judge over them.

Everything here is stdlib and runs in microseconds on the gateway's CPU, from the token stream alone:
token count, backtracking and verification phrase rates in the tail window (the same 1500 characters Jev
reads), candidate answers stated so far (boxed or "the answer is") and whether the latest one keeps
repeating, symbol densities, and the serving model's own token logprobs over the last window.

`FeatureJudge` scores "will a silent cut here keep the final answer?" with weights fitted offline on the
settled bench (agent/math_agent/settledfeat.py --fit); the shipped weights.json is that fit.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from importlib import resources
from pathlib import Path

TAIL_CHARS = 1500
LP_WINDOW = 128
BACKTRACK = re.compile(r"\b(wait|but|actually|hmm|however|mistake|wrong|re-?check|double-?check|hold on|"
                       r"alternatively|let me (?:re|try again|think again)|no,|oops|not right|incorrect)\b", re.I)
VERIFY = re.compile(r"\b(so the answer|the answer is|therefore|thus|hence|final answer|verif(?:y|ied)|confirm|"
                    r"that checks out|checks out|yes,|correct|indeed|matches)\b", re.I)
CANDIDATE = re.compile(r"\\boxed\{([^{}\n]{1,40})\}|(?:final answer|the answer)(?: is|:)\s*([^\n.,;]{1,40})", re.I)
MATH_SYMBOLS = set("=+-*/^\\()[]{}<>|")

FEATURES = (
    "log_tokens", "backtrack_per_k", "verify_per_k", "question_per_k", "newline_per_k",
    "has_candidate", "n_candidates", "last_repeats", "cands_in_tail", "last_two_agree", "frac_since_new",
    "digit_density", "symbol_density",
    "lp_mean_tail", "lp_std_tail", "lp_trend", "lp_surprise_tail", "lp_mean_all",
)
TEXT_ONLY = tuple(f for f in FEATURES if not f.startswith("lp_"))
LP_ONLY = ("log_tokens",) + tuple(f for f in FEATURES if f.startswith("lp_"))


def candidates(text: str) -> list[tuple[int, str]]:
    """(position, normalized answer) for every candidate answer stated in the text, in order."""
    out = []
    for m in CANDIDATE.finditer(text):
        a = (m.group(1) or m.group(2) or "").strip().strip("$ .").lower()
        a = re.sub(r"\s+", "", a)
        if a:
            out.append((m.start(), a))
    return out


def per_k(n: int, chars: int) -> float:
    return 1000.0 * n / max(chars, 1)


def features(prefix: str, logprobs: list[float | None], tokens: int) -> dict[str, float]:
    tail = prefix[-TAIL_CHARS:]
    f: dict[str, float] = {"log_tokens": math.log(max(tokens, 1))}
    f["backtrack_per_k"] = per_k(len(BACKTRACK.findall(tail)), len(tail))
    f["verify_per_k"] = per_k(len(VERIFY.findall(tail)), len(tail))
    f["question_per_k"] = per_k(tail.count("?"), len(tail))
    f["newline_per_k"] = per_k(tail.count("\n"), len(tail))
    cands = candidates(prefix)
    f["has_candidate"] = float(bool(cands))
    f["n_candidates"] = float(len({a for _, a in cands}))
    f["last_repeats"] = float(sum(1 for _, a in cands if a == cands[-1][1])) if cands else 0.0
    f["cands_in_tail"] = float(sum(1 for p, _ in cands if p >= len(prefix) - len(tail)))
    f["last_two_agree"] = float(len(cands) >= 2 and cands[-1][1] == cands[-2][1])
    first_pos: dict[str, int] = {}
    for p, a in cands:
        first_pos.setdefault(a, p)
    last_new = max(first_pos.values()) if first_pos else 0
    f["frac_since_new"] = (len(prefix) - last_new) / max(len(prefix), 1) if cands else 0.0
    f["digit_density"] = sum(c.isdigit() for c in tail) / max(len(tail), 1)
    f["symbol_density"] = sum(c in MATH_SYMBOLS for c in tail) / max(len(tail), 1)
    lps = [x for x in logprobs[:tokens] if x is not None]
    win = lps[-LP_WINDOW:]
    f["lp_mean_tail"] = statistics.fmean(win) if win else 0.0
    f["lp_std_tail"] = statistics.pstdev(win) if len(win) > 1 else 0.0
    half = len(win) // 2
    f["lp_trend"] = (statistics.fmean(win[half:]) - statistics.fmean(win[:half])) if half >= 8 else 0.0
    f["lp_surprise_tail"] = sum(1 for x in win if x < -2.0) / max(len(win), 1)
    f["lp_mean_all"] = statistics.fmean(lps) if lps else 0.0
    return f


class FeatureJudge:
    """Standardized logistic regression over `features`: P(a silent cut now keeps the final answer)."""

    def __init__(self, weights: dict):
        self.names = tuple(weights["features"])
        self.mean, self.scale = list(weights["mean"]), list(weights["scale"])
        self.coef, self.intercept = list(weights["coef"]), float(weights["intercept"])
        self.meta = {k: v for k, v in weights.items() if k not in ("features", "mean", "scale", "coef", "intercept")}
        if not (len(self.names) == len(self.mean) == len(self.scale) == len(self.coef)):
            raise ValueError("weights: features, mean, scale and coef must have the same length")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FeatureJudge":
        if path is None:
            text = resources.files(__package__).joinpath("weights.json").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        return cls(json.loads(text))

    def score_features(self, f: dict[str, float]) -> float:
        z = self.intercept
        for n, m, s, c in zip(self.names, self.mean, self.scale, self.coef):
            z += c * ((f.get(n, 0.0) - m) / (s or 1.0))
        return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))

    def score(self, prefix: str, logprobs: list[float | None], tokens: int) -> float:
        return self.score_features(features(prefix, logprobs, tokens))
