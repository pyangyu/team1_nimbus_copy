#!/usr/bin/env python3
"""Pool aime_drop_bench.py runs over several problem sets: paired differences with a 95% CI bootstrapped over
problems (all samples of a problem move together), and the same per set.

Expects <evals>/<set>/<model>/jobs.jsonl, with every model on a set run on the same natural traces
(round2_run.sh lays it out like that).

  python pool_evals.py --evals evals --out results/round2/pooled
"""
import argparse
import collections
import json
from pathlib import Path

from aime_drop_bench import cluster_ci

# (name, model a, arm a, model b, arm b); {x} is the budget left (0.25 or 0.1)
PAIRS = [
    ("untrained: notice - silent", "base", "warn_count_{x}", "base", "silent_{x}"),
    ("round 1: notice - its own silent", "r1", "warn_count_{x}", "r1", "silent_{x}"),
    ("round 2: notice - its own silent", "r2", "warn_count_{x}", "r2", "silent_{x}"),
    ("round 1 notice - untrained silent", "r1", "warn_count_{x}", "base", "silent_{x}"),
    ("round 2 notice - untrained silent", "r2", "warn_count_{x}", "base", "silent_{x}"),
    ("round 1 silent - untrained silent", "r1", "silent_{x}", "base", "silent_{x}"),
    ("round 2 notice - round 1 notice", "r2", "warn_count_{x}", "r1", "warn_count_{x}"),
]


def load(path):
    by = collections.defaultdict(dict)
    for line in path.open():
        j = json.loads(line)
        by[j["sid"]][j["key"]] = j
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", help="default: every folder in --evals")
    ap.add_argument("--allow", nargs="+", default=["0.25", "0.1"])
    ap.add_argument("--out", type=Path, help="writes <out>.json and <out>.txt")
    a = ap.parse_args()
    sets = a.sets or sorted(p.name for p in a.evals.iterdir() if p.is_dir())
    runs = {(s, m): load(a.evals / s / m / "jobs.jsonl")
            for s in sets for m in ("base", "r1", "r2") if (a.evals / s / m / "jobs.jsonl").exists()}

    def diff(only, ma, ka, mb, kb):
        d = collections.defaultdict(list)
        for s in only:
            A, B = runs.get((s, ma)), runs.get((s, mb))
            if not A or not B:
                continue
            for sid, arms in A.items():
                if ka in arms and kb in B.get(sid, {}):
                    d[f"{s}/{arms[ka]['id']}"].append(bool(arms[ka]["correct"]) - bool(B[sid][kb]["correct"]))
        if not d:
            return None
        m, lo, hi = cluster_ci(d)
        return {"pp": round(m, 1), "ci": [round(lo, 1), round(hi, 1)], "problems": len(d),
                "traces": sum(map(len, d.values()))}

    res, lines = {"sets": sets, "pooled": {}, "per_set": {}}, []
    for x in a.allow:
        lines.append(f"== {x} of the natural thinking left after the cut")
        for name, ma, ka, mb, kb in PAIRS:
            ka, kb = ka.format(x=x), kb.format(x=x)
            p = diff(sets, ma, ka, mb, kb)
            if p is None:
                continue
            res["pooled"][f"{name} @ {x}"] = p
            per = {s: diff([s], ma, ka, mb, kb) for s in sets}
            res["per_set"][f"{name} @ {x}"] = per
            lines.append(f"  {name:36} {p['pp']:+5.1f} [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}]  "
                         f"({p['problems']} problems, {p['traces']} traces)")
            lines.append("      " + "  ".join(f"{s} {v['pp']:+.1f}" for s, v in per.items() if v))
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
        a.out.with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
