import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("BROWNOUT_LIVE"),
                                reason="set BROWNOUT_LIVE=1 with LIFELINE_BASE_URL pointing at the Brownout gateway")

BASE = os.environ.get("LIFELINE_BASE_URL", "").rstrip("/")
ROOT = BASE.removesuffix("/v1")
KEY = os.environ.get("LIFELINE_API_KEY", "")
OUT = Path(os.environ.get("BROWNOUT_OUT", "runs/brownout-live"))
SPECS = os.environ.get("BROWNOUT_POLICIES", ",".join([
    "none", "reject", "uniform",
    "brownout:silent:jev",
    "brownout:announced:jev",
    "brownout:announced:jev-serial",
    "brownout:announced:online",
    "brownout:announced:oracle",
    "brownout:announced:jev-learn",
])).split(",")
HARD_SPECS = os.environ.get("BROWNOUT_HARD_POLICIES", ",".join([
    "none", "uniform",
    "brownout:announced:jev",
    "brownout:announced:jev-serial",
    "brownout:announced:online",
    "brownout:announced:oracle",
])).split(",")
# a spec is policy[:notice[:predictor[:cut]]][:f<frac>][:e<exit>]; cut = proportional (default), ranked or
# settled (Jev re-asked mid-decode); f caps the shed fraction, e sets the settled exit threshold.
# BROWNOUT_POLICIES=skip runs no sweep on the first dataset.
SPECS = [s for s in SPECS if s and s != "skip"]
HARD_SPECS = [s for s in HARD_SPECS if s and s != "skip"]
N = int(os.environ.get("BROWNOUT_N", "64"))
RATE = float(os.environ.get("BROWNOUT_RATE", "4.0"))
DATASET = os.environ.get("BROWNOUT_DATASET", "gsm8k")
HARD_DATASET = os.environ.get("BROWNOUT_HARD_DATASET", "math500")
SWEEPS = {}
if SPECS:
    SWEEPS[DATASET] = {"specs": SPECS, "n": N, "rate": RATE, "budget": 1024, "max_budget": 2048, "solo_s": 90}
if HARD_DATASET:
    SWEEPS[HARD_DATASET] = {"specs": HARD_SPECS, "n": int(os.environ.get("BROWNOUT_HARD_N", "32")),
                            "rate": float(os.environ.get("BROWNOUT_HARD_RATE", "2.0")),
                            "budget": 4096, "max_budget": 6144, "solo_s": 300}
SEEDS = [int(x) for x in os.environ.get("BROWNOUT_SEEDS", "0").split(",") if x.strip()]
# one sweep per seed, specs in order within it, so an oracle spec always follows that seed's `none`
CASES = [(ds, spec, seed) for seed in SEEDS for ds, sw in SWEEPS.items() for spec in sw["specs"]]
REJECT_WAITING = float(os.environ.get("BROWNOUT_REJECT_WAITING", "2"))
ADMIT_RATE = float(os.environ.get("BROWNOUT_ADMIT_RATE", "0.25"))
ADMIT_BURST = float(os.environ.get("BROWNOUT_ADMIT_BURST", "8"))
MAX_FRAC = float(os.environ.get("BROWNOUT_MAX_FRAC", "0.9"))
JUDGE_SIGNAL = os.environ.get("BROWNOUT_JUDGE_SIGNAL", "settled")
BANK = os.environ.get("BROWNOUT_BANK", "0") == "1"
LOADGEN_ARGS = shlex.split(os.environ.get("BROWNOUT_LOADGEN_ARGS", ""))  # e.g. --arrivals burstgpt
COUNTERS = ("think", "rejected", "deferred", "cut", "announced", "upstream_errors", "serial_waits", "serial_wait_s",
            "oracle_hits", "oracle_misses")
COLUMNS = ("dataset", "policy", "seed", "max_frac", "exit_at", "n", "accuracy", "deadline_met", "correct_and_on_time", "admitted_correct_and_on_time",
           "rejected_workflows",
           "latency_p50_s", "latency_p90_s", "thinking_tokens_mean", "think", "cut", "announced",
           "rejected", "pressured_samples", "jev_before_first_token", "jev_during_decode", "jev_tier_lowered",
           "jev_mean_wait_s", "jev_mean_call_s", "jev_failure_rate", "serial_wait_mean_s", "oracle_hit_rate",
           "cut_mode", "ranked_cuts_with_probs", "ranked_cuts_flat", "ranked_shed_tokens",
           "settled_cuts", "exits", "judged", "judge_mean_ms", "judge_mean_settled",
           "probe_labeled", "probe_probes", "probe_needed_below_used", "probe_errors",
           "paired_n", "online_exact", "jev_exact",
           "online_within_one", "jev_within_one", "online_under", "jev_under")
JEV_COUNTERS = ("before_first_token", "during_decode", "tier_lowered", "asked", "wait_s")
RESULTS: list[dict] = []


def gw(method: str, path: str, **kw) -> dict:
    r = httpx.request(method, ROOT + path, headers={"Authorization": f"Bearer {KEY}"}, timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def observed(stats: dict) -> int:
    return sum(v["labels"] + v["censored"] for v in stats["classifier"]["versions"].values())


def paired(stats: dict) -> dict:
    out = {src: {"n": 0, "exact": 0, "within_one": 0, "under": 0} for src in ("online", "jev")}
    for v in stats["classifier"]["versions"].values():
        for src, sc in v["paired"].items():
            for k in out[src]:
                out[src][k] += sc[k]
    return out


def compare(before: dict, after: dict) -> dict:
    b, a = paired(before), paired(after)
    n = a["jev"]["n"] - b["jev"]["n"]
    row = {"paired_n": n}
    for src in ("online", "jev"):
        for k in ("exact", "within_one", "under"):
            row[f"{src}_{k}"] = round((a[src][k] - b[src][k]) / n, 4) if n else None
    return row


def call_s(before: dict, after: dict) -> float | None:
    b, a = before["classifier"]["jev"] or {}, after["classifier"]["jev"] or {}
    calls = a.get("calls", 0) - b.get("calls", 0)
    return round((a.get("call_s", 0) - b.get("call_s", 0)) / calls, 3) if calls and "call_s" in a else None


def failure_rate(before: dict, after: dict) -> float | None:
    b, a = before["classifier"]["jev"] or {}, after["classifier"]["jev"] or {}
    calls = a.get("calls", 0) - b.get("calls", 0)
    return round((a.get("failures", 0) - b.get("failures", 0)) / calls, 3) if calls else None


def write_results() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in RESULTS:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in COLUMNS) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("dataset,spec,seed", CASES, ids=[f"{d}:{s}:s{x}" for d, s, x in CASES])
def test_policy_under_load(dataset, spec, seed):
    sw = SWEEPS[dataset]
    n, rate = sw["n"], sw["rate"]
    parts = spec.split(":")
    # optional trailing f<frac> (shed cap, default BROWNOUT_MAX_FRAC) and e<exit> (settled exit threshold)
    max_frac, exit_at = MAX_FRAC, None
    while parts and parts[-1][:1] in ("f", "e") and parts[-1][1:].replace(".", "", 1).isdigit():
        tok = parts.pop()
        if tok[0] == "f":
            max_frac = float(tok[1:])
        else:
            exit_at = float(tok[1:])
    policy, notice, predictor, mode = (parts + ["", "", ""])[:4]
    if predictor == "jev-learn" and "learner" not in gw("GET", "/v1/stats")["jev"]:
        pytest.skip("trained Jev: the online Jev learner is not deployed yet")
    gw("POST", "/admin/policy", json={"policy": policy, "notice": notice or "silent", "predictor": predictor or "jev",
                                        "cut": mode or "proportional", "max_frac": max_frac,
                                        "exit_at": exit_at, "judge_signal": JUDGE_SIGNAL, "bank": BANK,
                                        "reject_waiting": REJECT_WAITING, "admit_rate": ADMIT_RATE,
                                        "admit_burst": ADMIT_BURST, "reset": True})
    before = gw("GET", "/v1/stats")
    out = OUT / (f"{dataset}-{spec.replace(':', '-')}" + (f"-s{seed}" if seed else ""))
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--dataset", dataset, "--rate", str(rate),
                        "--n", str(n), "--budget", str(sw["budget"]), "--max-budget", str(sw["max_budget"]),
                        "--default-solo-s", str(sw["solo_s"]), "--seed", str(seed), "--out", str(out), *LOADGEN_ARGS],
                       capture_output=True, text=True, timeout=3600)
    (out / "loadgen.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, p.stderr[-2000:]
    after = wait_for_probes()
    s = json.loads((out / "summary.json").read_text())
    d = {k: after["gateway"][k] - before["gateway"][k] for k in COUNTERS}
    j = {k: after["jev"][k] - before["jev"][k] for k in JEV_COUNTERS}
    hist = [a - b for a, b in zip(after["pressure"]["p_hist_deciles"], before["pressure"]["p_hist_deciles"])]
    oracle = d["oracle_hits"] + d["oracle_misses"]
    jd = {k: after["judge"][k] - before["judge"][k] for k in ("answered", "call_ms", "settled_sum")}
    row = {"dataset": dataset, "policy": spec, "seed": seed, "max_frac": max_frac, "exit_at": exit_at,
           **{k: s.get(k) for k in COLUMNS if k in s},
           "rejected_workflows": s.get("rejected"),
           **d, "pressured_samples": sum(hist[1:]), "observed": observed(after) - observed(before),
           **{f"jev_{k}": j[k] for k in ("before_first_token", "during_decode", "tier_lowered")},
           "jev_mean_wait_s": round(j["wait_s"] / j["asked"], 3) if j["asked"] else None,
           "jev_mean_call_s": call_s(before, after),
           "jev_failure_rate": failure_rate(before, after),
           "serial_wait_mean_s": round(d["serial_wait_s"] / d["serial_waits"], 3) if d["serial_waits"] else None,
           "oracle_hit_rate": round(d["oracle_hits"] / oracle, 3) if oracle else None,
           "cut_mode": after["controller"]["mode"],
           **{f"ranked_{k}": after["controller"]["ranked"][k] - before["controller"]["ranked"][k]
              for k in ("cuts_with_probs", "cuts_flat", "shed_tokens")},
           **{k: after["controller"]["ranked"][k] - before["controller"]["ranked"][k] for k in ("settled_cuts", "exits")},
           "judged": jd["answered"],
           "judge_mean_ms": round(jd["call_ms"] / jd["answered"], 1) if jd["answered"] else None,
           "judge_mean_settled": round(jd["settled_sum"] / jd["answered"], 3) if jd["answered"] else None,
           **{f"probe_{k}": after["probe"][k] - before["probe"][k]
              for k in ("labeled", "probes", "needed_below_used", "errors")},
           **compare(before, after)}
    RESULTS.append(row)
    write_results()
    assert s["n"] == n and s["errors"] == 0 and d["upstream_errors"] == 0
    assert d["think"] > 0 and row["observed"] == d["think"]
    if policy == "none":
        assert d["cut"] == d["announced"] == d["rejected"] == 0
    if policy in ("uniform", "brownout") and row["pressured_samples"] >= 3:
        assert d["cut"] + d["announced"] > 0
    if policy == "reject" and n > ADMIT_BURST + ADMIT_RATE * n / rate + 1:
        assert s["rejected"] > 0 and d["rejected"] == s["rejected"]
    if policy != "reject":
        assert s["rejected"] == 0
    if policy == "none":
        assert row["paired_n"] > 0
    if predictor in ("jev", "jev-serial", ""):
        assert row["jev_failure_rate"] is not None and row["jev_failure_rate"] < 0.1
    if predictor == "jev-serial":
        assert d["serial_waits"] == d["think"] and row["jev_before_first_token"] > 0
    if predictor == "oracle":
        assert d["oracle_hits"] > 0
    if mode in ("ranked", "settled") and d["cut"] + d["announced"] > 0:
        assert row["ranked_cuts_with_probs"] + row["ranked_cuts_flat"] > 0
    if mode == "settled":
        assert row["judged"] > 0 and after["judge"]["failed"] - before["judge"]["failed"] < 0.1 * row["judged"] + 1
    if after["gateway"]["labels"] == "needed" and policy == "none":
        assert row["probe_labeled"] > 0 and row["probe_errors"] == 0


def wait_for_probes(timeout_s: float = 900.0) -> dict:
    """The gateway labels finished requests in the background; wait until every probe has landed."""
    t0 = time.time()
    while True:
        st = gw("GET", "/v1/stats")
        if st["probe"]["queued"] == 0 and st["probe"]["in_flight"] == 0:
            return st
        if time.time() - t0 > timeout_s:
            raise AssertionError(f"probes still running after {timeout_s}s: {st['probe']}")
        time.sleep(2)


def test_jev_answered_and_classifier_learned():
    st = gw("GET", "/v1/stats")
    jev = st["classifier"]["jev"]
    assert jev is not None and jev["calls"] > jev["failures"]
    assert observed(st) > 0
    (OUT / "gateway-stats.json").write_text(json.dumps(st, indent=2))
