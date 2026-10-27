#!/usr/bin/env python3
"""Before vs after on AIME 2026: same natural traces, same cut points, final answer only.

  base   aime_final   (untrained Qwen3-4B: aime_drop_bench.py --traces aime_drop --answer-mode final)
  tuned  aime_sft     (aime_drop_bench.py --model lora_sft/merged --traces aime_drop --answer-mode final)

Both runs cut the same base-model traces at the same point with the same W, so the only difference is the
model that continues (and answers) after the cut. The main question: does the tuned model with the notice
beat a silent cut of the base model?

  python compare_tuned.py --base $WORK/aime_final --tuned $WORK/aime_sft
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

from aime_drop_bench import cluster_ci


def load(d):
    by = collections.defaultdict(dict)
    for line in (d / "jobs.jsonl").open():
        j = json.loads(line)
        by[j["sid"]][j["key"]] = j
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--tuned", type=Path, required=True)
    a = ap.parse_args()
    base, tuned = load(a.base), load(a.tuned)
    allow = json.loads((a.tuned / "config.json").read_text())["allow"]
    keys = sorted({k for s in tuned.values() for k in s} & {k for s in base.values() for k in s})
    sids = sorted(s for s in tuned if s in base and all(k in tuned[s] and k in base[s] for k in keys))
    print(f"{len(sids)} traces in both runs, {len({base[s][keys[0]]['id'] for s in sids})} problems\n")

    def arm(by, k):
        rr = [by[s][k] for s in sids]
        out = {"acc": round(100 * sum(r["correct"] for r in rr) / len(rr), 1),
               "think": round(statistics.mean(r["think_used"] / r["L"] for r in rr), 3),
               "answer_tokens": round(statistics.mean(r["ans_tokens"] for r in rr))}
        if k.startswith("warn"):
            w = float(k.rsplit("_", 1)[1])
            out["closed_itself"] = round(sum(bool(r["closed_itself"]) for r in rr) / len(rr), 3)
            out["window_used_median"] = round(statistics.median((r["think_used"] - 0.5 * r["L"]) / (w * r["L"])
                                                                for r in rr), 2)
        return out

    res = {"traces": len(sids), "arms": {}, "paired_ci_pp": {}}
    print(f"{'arm':18s} {'base acc':>9s} {'tuned acc':>10s} {'base think':>11s} {'tuned think':>12s} "
          f"{'base ans':>9s} {'tuned ans':>10s} {'tuned closed':>13s} {'tuned W used':>13s}")
    for k in keys:
        b, t = arm(base, k), arm(tuned, k)
        res["arms"][k] = {"base": b, "tuned": t}
        print(f"{k:18s} {b['acc']:9.1f} {t['acc']:10.1f} {b['think']:11.2f} {t['think']:12.2f} "
              f"{b['answer_tokens']:9d} {t['answer_tokens']:10d} "
              f"{t.get('closed_itself', ''):>13} {t.get('window_used_median', ''):>13}")

    def diff(xb, kx, yb, ky):
        d = collections.defaultdict(list)
        for s in sids:
            d[base[s][kx]["id"]].append(int(xb[s][kx]["correct"]) - int(yb[s][ky]["correct"]))
        return cluster_ci(d)

    pairs = []
    for x in allow:
        pairs += [(f"tuned warn_count_{x} - base silent_{x}", tuned, f"warn_count_{x}", base, f"silent_{x}"),
                  (f"tuned warn_count_{x} - base warn_count_{x}", tuned, f"warn_count_{x}", base, f"warn_count_{x}"),
                  (f"tuned warn_soft_{x} - base silent_{x}", tuned, f"warn_soft_{x}", base, f"silent_{x}")]
    pairs.append(("tuned natural_final - base natural_final", tuned, "natural_final", base, "natural_final"))
    print("\npaired difference (pp), 95% CI bootstrapped over problems")
    for name, xb, kx, yb, ky in pairs:
        if kx in keys and ky in keys:
            res["paired_ci_pp"][name] = m = diff(xb, kx, yb, ky)
            print(f"  {name:44s} {m[0]:+6.1f}  [{m[1]:+.1f}, {m[2]:+.1f}]")
    (a.tuned / "compare.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved {a.tuned / 'compare.json'}")


if __name__ == "__main__":
    main()
