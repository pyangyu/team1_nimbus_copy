"""Summarize serve_eval.py runs: goodput, SLO attainment, accuracy, thinking, throughput, TTFT, and
(with --ref, the no-cut run of the same arrivals) how many problems the cuts lost or won.

    python serve_report.py serve/*.jsonl --ref serve/nocut.jsonl
    python serve_report.py serve/nocut.jsonl serve/tb.jsonl --timeline 5      # mean TTFT per 5 s of arrivals

goodput     share of ALL arrivals answered correctly with TTFT <= SLO (rejected = failure)
slo met     share of all arrivals with TTFT <= SLO (rejected = miss)
accuracy    share of all arrivals answered correctly (rejected = wrong)
think       mean thinking tokens per served request (before + after a cut)
thrpt       requests finished per minute inside the burst windows (each burst + the 60 s after it);
            the whole run when there are no --bursts
ttft p50/99 over served requests, seconds
cut         share of served requests that were cut (or budget-capped / early-exited)
lost / won  with --ref: among cut requests, correct in the no-cut run -> wrong here / wrong -> correct
react       seconds from the first burst's start until the mean TTFT of arrivals (bins of --bin s) is
            back within the SLO
"""
import argparse
import json
import statistics
from collections import defaultdict


def load(path):
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("meta"):
                meta = r
            else:
                rows.append(r)
    return meta, rows


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def windows(meta):
    out = []
    for b in filter(None, (meta.get("bursts") or "").split(",")):
        s, d, _ = (float(x) for x in b.split(":"))
        out.append((s, s + d + 60.0))
    return out


def reaction(meta, rows, bin_s):
    ws = windows(meta)
    if not ws:
        return None
    start, slo = ws[0][0], meta.get("slo", 10.0)
    bins = defaultdict(list)
    for r in rows:
        if r["ttft"] is not None and r["arrival"] >= start:
            bins[int((r["arrival"] - start) // bin_s)].append(r["ttft"])
    went_over = False
    for b in sorted(bins):
        m = statistics.mean(bins[b])
        if m > slo:
            went_over = True
        elif went_over:
            return b * bin_s
    return 0.0 if not went_over else float("inf")


def summarize(path, meta, rows, ref):
    n = len(rows)
    served = [r for r in rows if not r["rejected"]]
    good = sum(1 for r in rows if r["correct"] and r["slo_ok"]) / n
    slo = sum(1 for r in rows if r["slo_ok"]) / n
    acc = sum(1 for r in rows if r["correct"]) / n
    think = statistics.mean(r["think_tokens"] for r in served) / 1000 if served else float("nan")
    ws = windows(meta)
    if ws:
        mins = sum(e - s for s, e in ws) / 60
        done = sum(1 for r in served if r["finish"] is not None and any(s <= r["finish"] < e for s, e in ws))
    else:
        fin = [r["finish"] for r in served if r["finish"] is not None]
        mins = (max(fin) - min(r["arrival"] for r in rows)) / 60 if fin else float("nan")
        done = len(fin)
    ttfts = [r["ttft"] for r in served if r["ttft"] is not None]
    cut = [r for r in served if r["cut"]]
    name = meta.get("policy", path)
    if name in ("hard", "tb") and meta.get("select", "high") != "high":
        name += "/" + meta["select"]
    s = {"run": name, "n": n, "goodput": good, "slo": slo, "acc": acc, "think": think,
         "thrpt": done / mins if mins else float("nan"), "p50": pct(ttfts, 0.5), "p99": pct(ttfts, 0.99),
         "rej": 1 - len(served) / n, "cut": len(cut) / max(len(served), 1)}
    if ref:
        base = {r["rid"]: r["correct"] for r in ref}
        cs = [r for r in cut if r["rid"] in base]
        s["lost"] = sum(1 for r in cs if base[r["rid"]] and not r["correct"]) / max(len(cs), 1)
        s["won"] = sum(1 for r in cs if not base[r["rid"]] and r["correct"]) / max(len(cs), 1)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None, help="the nocut run of the same arrivals (for lost / won)")
    ap.add_argument("--timeline", type=float, default=0, help="also print mean TTFT per this many seconds of arrival")
    ap.add_argument("--bin", type=float, default=5.0, help="bin width for the reaction time")
    args = ap.parse_args()
    ref = load(args.ref)[1] if args.ref else None
    head = (f"{'run':28s} {'n':>5s} {'goodput':>8s} {'slo met':>8s} {'acc':>6s} {'think K':>8s} {'thrpt':>7s} "
            f"{'p50':>6s} {'p99':>6s} {'rej':>5s} {'cut':>5s}" + (f" {'lost':>6s} {'won':>6s}" if ref else "")
            + f" {'react':>6s}")
    print(head)
    loaded, caps = [], []
    for p in args.runs:
        meta, rows = load(p)
        if not rows:
            continue
        loaded.append((p, meta, rows))
        s = summarize(p, meta, rows, ref)
        rt = reaction(meta, rows, args.bin)
        name = f"{s['run']} ({p.split('/')[-1]})"[:28]
        line = (f"{name:28s} {s['n']:5d} {s['goodput']:8.3f} {s['slo']:8.1%} {s['acc']:6.3f} {s['think']:8.2f} "
                f"{s['thrpt']:7.1f} {s['p50']:6.1f} {s['p99']:6.1f} {s['rej']:5.0%} {s['cut']:5.0%}")
        if ref:
            line += f" {s['lost']:6.1%} {s['won']:6.1%}"
        if rt is None:
            line += f" {'-':>6s}"
        elif rt == 0.0:
            line += f" {'never':>6s}"                    # mean TTFT never went over the SLO
        elif rt == float("inf"):
            line += f" {'>end':>6s}"                     # still over the SLO when the arrivals ended
        else:
            line += f" {rt:6.1f}"
        print(line)
        svc = [r["finish"] - r["admit"] for r in rows if r["finish"] is not None and r["admit"] is not None]
        caps.append((s["run"], meta.get("slots"), statistics.mean(svc) if svc else float("nan")))
    for name, slots, svc in caps:
        if name == "nocut" and slots:
            print(f"\ncapacity from {name}: {slots} slots / {svc:.1f} s mean service = {slots / svc:.3f} req/s "
                  f"(use a low-load nocut run for this; --rate = load x capacity)")
    if args.timeline:
        print(f"\nmean TTFT (s) of requests arriving in each {args.timeline:.0f} s bin")
        bins = sorted({int(r['arrival'] // args.timeline) for _, _, rows in loaded for r in rows})
        print(f"{'t (s)':>7s} " + " ".join(f"{(m.get('policy') or p)[:9]:>9s}" for p, m, _ in loaded))
        for b in bins:
            cells = []
            for _, _, rows in loaded:
                xs = [r["ttft"] for r in rows if r["ttft"] is not None and int(r["arrival"] // args.timeline) == b]
                cells.append(f"{statistics.mean(xs):9.1f}" if xs else f"{'-':>9s}")
            print(f"{b * args.timeline:7.0f} " + " ".join(cells))


if __name__ == "__main__":
    main()
