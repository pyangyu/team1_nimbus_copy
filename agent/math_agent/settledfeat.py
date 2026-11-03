"""Cheap settled judge: a trained classifier over surface features of the thinking so far, scored on the
settled-bench rows that Jev, DEER and the probe label were scored on.

The question it answers: does an LLM judge (Jev reading the trace tail) see anything a few microseconds of
feature counting plus logistic regression cannot? Every feature here is computable in the gateway from the
token stream alone (no extra model call): token count, backtracking and verification phrase rates in Jev's
1500-char tail window, candidate answers found so far (boxed or "the answer is") and whether the latest one
keeps repeating, symbol densities, and the serving model's own token logprobs over the last window.

  python -m math_agent.settledfeat runs/settledbench-math500-s0-n96-traces [--out summary-feat.json]

Reads rows.jsonl + traces.jsonl from the bench directory. Labels and row filter are the bench's: rows whose
uncut answer was right, label = the silent cut at that point kept it right. Scoring is out-of-fold with
problems grouped (no problem's cut points split across train and test), repeated over several fold draws,
so the number is comparable to the bench's AUC column for jev_done / deer / probe_match on the same rows.
Also simulates the bench's early-exit policy on the out-of-fold score.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

from .settledbench import SIGNALS, THRESHOLDS, auc, chars_at, early_exit, oracle_exit

TAIL_CHARS = 1500      # Jev's window (budget_classifier.jev.settled_state tail_chars)
LP_WINDOW = 128        # logprob window at the tail
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


def load(bench: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = [json.loads(l) for l in (bench / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    traces = {}
    tp = bench / "traces.jsonl"
    if tp.exists():
        for l in tp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                t = json.loads(l)
                traces[t["id"]] = t
    return rows, traces


def build(rows: list[dict], traces: dict[str, dict]) -> list[dict]:
    """The scored rows (uncut answer right, trace present) with their feature dict attached."""
    out = []
    for r in rows:
        if "error" in r or not r.get("uncut_correct") or r["id"] not in traces:
            continue
        t = traces[r["id"]]
        at = chars_at([tuple(x) for x in t["trail"]], r["cut"])
        if at is None:
            continue
        out.append({**r, "feat": features(t["thinking"][:at], t.get("token_logprobs") or [], r["cut"])})
    return out


def group_folds(ids: list[str], k: int, seed: int) -> list[int]:
    """Fold index per row, assigning whole problems to folds."""
    import random
    groups = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    return [fold_of[i] for i in ids]


def oof_scores(X: list[list[float]], y: list[bool], ids: list[str], model: str, k: int, repeats: int) -> list[float]:
    """Mean out-of-fold score per row over repeated grouped K-fold."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xa, ya = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    acc = np.zeros(len(X))
    for rep in range(repeats):
        folds = np.asarray(group_folds(ids, k, rep))
        for f in range(k):
            te, tr = folds == f, folds != f
            if te.sum() == 0 or len(set(ya[tr])) < 2:
                continue
            if model == "lr":
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))
            else:
                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                                     min_samples_leaf=5, class_weight="balanced", random_state=rep)
            clf.fit(Xa[tr], ya[tr])
            acc[te] += clf.predict_proba(Xa[te])[:, 1]
    return (acc / repeats).tolist()


def coefficients(X: list[list[float]], y: list[bool], names: tuple[str, ...]) -> dict[str, float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(Xs, np.asarray(y, dtype=int))
    return {n: round(float(c), 3) for n, c in sorted(zip(names, clf.coef_[0]), key=lambda t: -abs(t[1]))}


def evaluate(scored: list[dict], k: int = 10, repeats: int = 5) -> dict:
    y = [bool(r["silent_correct"]) for r in scored]
    ids = [r["id"] for r in scored]
    k = min(k, len(set(ids)))
    out: dict = {"rows": len(scored), "problems": len(set(ids)), "positives": sum(y), "negatives": len(y) - sum(y),
                 "folds": k, "repeats": repeats, "auc": {}, "single_feature_auc": {}, "early_exit": {}}
    # the bench's judges on exactly these rows
    for name in SIGNALS:
        have = [r for r in scored if r.get(name) is not None]
        out["auc"][name] = round(auc([float(r[name]) for r in have], [bool(r["silent_correct"]) for r in have]), 4) \
            if have else None
    for name in FEATURES:
        out["single_feature_auc"][name] = round(auc([r["feat"][name] for r in scored], y), 4)
    sets = {"feat": FEATURES, "feat_text": TEXT_ONLY, "feat_logprob": LP_ONLY}
    extras = {"feat+deer": ("deer",), "feat+jev_done": ("jev_done",), "feat+deer+jev_done": ("deer", "jev_done")}
    for label, names in sets.items():
        X = [[r["feat"][n] for n in names] for r in scored]
        for model in ("lr", "gbt"):
            s = oof_scores(X, y, ids, model, k, repeats)
            key = f"{label}_{model}"
            out["auc"][key] = round(auc(s, y), 4)
            for r, v in zip(scored, s):
                r[key] = v
    for label, cols in extras.items():
        have = [r for r in scored if all(r.get(c) is not None for c in cols)]
        if len(have) < len(scored) // 2:
            continue
        X = [[r["feat"][n] for n in FEATURES] + [float(r[c]) for c in cols] for r in have]
        s = oof_scores(X, [bool(r["silent_correct"]) for r in have], [r["id"] for r in have], "lr", k, repeats)
        out["auc"][f"{label}_lr"] = round(auc(s, [bool(r["silent_correct"]) for r in have]), 4)
    out["lr_coefficients"] = coefficients([[r["feat"][n] for n in FEATURES] for r in scored], y, FEATURES)
    out["early_exit"]["oracle"] = oracle_exit(scored)
    for name in ("jev_done", "deer", "feat_lr", "feat_gbt"):
        for tau in THRESHOLDS:
            out["early_exit"][f"{name}>={tau}"] = early_exit(scored, name, tau)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", help="settledbench output directory (rows.jsonl + traces.jsonl)")
    ap.add_argument("--out", help="where to write the summary json (default: <bench>/summary-feat.json)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    bench = Path(args.bench)
    rows, traces = load(bench)
    if not traces:
        sys.exit(f"{bench}/traces.jsonl missing or empty: rerun settledbench on a commit that writes traces")
    scored = build(rows, traces)
    if len(scored) < 10:
        sys.exit(f"only {len(scored)} scorable rows")
    summary = evaluate(scored, args.folds, args.repeats)
    out = Path(args.out) if args.out else bench / "summary-feat.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
