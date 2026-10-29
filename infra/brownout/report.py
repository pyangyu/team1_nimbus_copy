"""Tables from a live sweep's results directory (brownout-results/ as e2e.sh leaves it).

  python infra/brownout/report.py brownout-results [--paired]

One row per run from results.json, the per-source split from each run's summary.json (mixed streams), and
with --paired the workflow-level comparison of every cut run against that seed's `none` run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_runs(root: Path) -> list[dict]:
    res = json.loads((root / "results.json").read_text(encoding="utf-8"))
    for r in res:
        name = f"{r['dataset']}-{r['policy'].replace(':', '-')}" + (f"-s{r['seed']}" if r.get("seed") else "")
        d = root / name
        r["_dir"] = d
        if (d / "summary.json").exists():
            r["_summary"] = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if (d / "workflows.jsonl").exists():
            r["_wf"] = [json.loads(l) for l in (d / "workflows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return res


def fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main_table(runs: list[dict]) -> str:
    cols = ["dataset", "policy", "seed", "max_frac", "exit_at", "accuracy", "deadline_met", "correct_and_on_time",
            "latency_p90_s", "thinking_tokens_mean", "think", "cut", "announced", "settled_cuts", "exits", "judged",
            "judge_mean_ms", "judge_mean_settled", "jev_exact", "online_exact"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in runs:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def source_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        s = r.get("_summary", {}).get("by_source") or {}
        for src, v in s.items():
            rows.append((r["policy"], r.get("seed"), src, v))
    if not rows:
        return ""
    cols = ["policy", "seed", "source", "n", "accuracy", "deadline_met", "correct_and_on_time", "thinking_tokens_mean"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for pol, seed, src, v in rows:
        out.append("| " + " | ".join([pol, fmt(seed), src] + [fmt(v.get(k)) for k in cols[3:]]) + " |")
    return "\n".join(out)


def paired_table(runs: list[dict]) -> str:
    base = {(r["dataset"], r.get("seed")): r for r in runs if r["policy"] == "none" and "_wf" in r}
    cols = ["policy", "seed", "n", "both_correct", "lost", "gained", "net", "late_in_none", "late_now"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in runs:
        b = base.get((r["dataset"], r.get("seed")))
        if b is None or "_wf" not in r or r is b:
            continue
        bw = {w["example_id"] + "|" + w["workflow_id"]: w for w in b["_wf"]}
        rw = {w["example_id"] + "|" + w["workflow_id"]: w for w in r["_wf"]}
        keys = [k for k in bw if k in rw]
        lost = sum(1 for k in keys if bw[k]["correct"] and not rw[k]["correct"])
        gained = sum(1 for k in keys if not bw[k]["correct"] and rw[k]["correct"])
        both = sum(1 for k in keys if bw[k]["correct"] and rw[k]["correct"])
        out.append("| " + " | ".join([r["policy"], fmt(r.get("seed")), str(len(keys)), str(both), str(lost), str(gained),
                                      f"{gained - lost:+d}",
                                      str(sum(1 for k in keys if bw[k].get("met_deadline") is False)),
                                      str(sum(1 for k in keys if rw[k].get("met_deadline") is False))]) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--paired", action="store_true")
    a = ap.parse_args()
    runs = load_runs(a.root)
    print(main_table(runs))
    st = source_table(runs)
    if st:
        print("\nPer source\n")
        print(st)
    if a.paired:
        print("\nPaired against none (same seed)\n")
        print(paired_table(runs))


if __name__ == "__main__":
    main()
