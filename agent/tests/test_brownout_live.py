import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from math_agent import metrics, workload

pytestmark = pytest.mark.skipif(not os.environ.get("BROWNOUT_LIVE"),
                                reason="set BROWNOUT_LIVE=1 with LIFELINE_BASE_URL pointing at the Brownout gateway")

BASE = os.environ.get("LIFELINE_BASE_URL", "").rstrip("/")
ROOT = BASE.removesuffix("/v1")
KEY = os.environ.get("LIFELINE_API_KEY", "")
OUT = Path(os.environ.get("BROWNOUT_OUT", "runs/brownout-live"))
SUITE = os.environ.get("BROWNOUT_SUITE", "default")
REALISTIC = SUITE == "realistic"
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
N = int(os.environ.get("BROWNOUT_N", "64"))
RATE = float(os.environ.get("BROWNOUT_RATE", "4.0"))
DATASET = os.environ.get("BROWNOUT_DATASET", "gsm8k")
HARD_DATASET = os.environ.get("BROWNOUT_HARD_DATASET", "math500")
SWEEPS = {DATASET: {"specs": SPECS, "n": N, "rate": RATE, "budget": 1024, "max_budget": 2048, "solo_s": 90}}
if HARD_DATASET:
    SWEEPS[HARD_DATASET] = {"specs": HARD_SPECS, "n": int(os.environ.get("BROWNOUT_HARD_N", "32")),
                            "rate": float(os.environ.get("BROWNOUT_HARD_RATE", "2.0")),
                            "budget": 4096, "max_budget": 6144, "solo_s": 300}
CASES = [] if REALISTIC else [(ds, spec) for ds, sw in SWEEPS.items() for spec in sw["specs"]]
REAL_SPECS = os.environ.get("BROWNOUT_POLICIES", ",".join([
    "none", "uniform", "brownout:silent:online", "brownout:announced:online",
])).split(",") if REALISTIC else []
WORKLOAD = os.environ.get("BROWNOUT_WORKLOAD", workload.DEFAULT_WORKLOAD)
PROFILE = os.environ.get("BROWNOUT_PROFILE", workload.DEFAULT_PROFILE)
TRACE = os.environ.get("BROWNOUT_TRACE")
BASE_MODEL = os.environ.get("BROWNOUT_MODEL")
TRACE_ENV = {"--trace-log": "BROWNOUT_TRACE_LOG", "--trace-start-h": "BROWNOUT_TRACE_START_H",
             "--trace-window-h": "BROWNOUT_TRACE_WINDOW_H", "--trace-s": "BROWNOUT_TRACE_S", "--load": "BROWNOUT_LOAD"}
TRACE_ARGS = ["--trace", TRACE] + [x for k, v in TRACE_ENV.items() if os.environ.get(v)
                                   for x in (k, os.environ[v])] if TRACE else []
TRACE_S = float(os.environ.get("BROWNOUT_TRACE_S", "1800"))
SEED = int(os.environ.get("BROWNOUT_SEED", "0"))
TAIL_S = float(os.environ.get("BROWNOUT_TAIL_S", "1200"))
CAP_CONCURRENCY = os.environ.get("BROWNOUT_CAPACITY_CONCURRENCY")
CAP_WARMUP_S = float(os.environ.get("BROWNOUT_CAPACITY_WARMUP_S", "180"))
CAP_S = float(os.environ.get("BROWNOUT_CAPACITY_S", "600"))
CAPACITY: dict = {}
REAL_COLUMNS = ("policy", "n", "accuracy", "deadline_met", "correct_and_on_time", "goodput_baseline", "goodput_burst",
                "accuracy_baseline", "accuracy_burst", "cancelled", "latency_p50_s", "latency_p95_s",
                "thinking_tokens_mean", "think", "cut", "announced", "notices", "p_mean", "p_at_cap", "p_zero",
                "queue_frac", "p_mean_baseline", "p_mean_burst", "word_model_window", "word_model_exact",
                "word_model_within_one", "word_model_under", "jev_asked", "replica_think", "p_mean_replicas")
REJECT_WAITING = float(os.environ.get("BROWNOUT_REJECT_WAITING", "2"))
ADMIT_RATE = float(os.environ.get("BROWNOUT_ADMIT_RATE", "0.25"))
ADMIT_BURST = float(os.environ.get("BROWNOUT_ADMIT_BURST", "8"))
COUNTERS = ("think", "rejected", "deferred", "cut", "announced", "notices", "upstream_errors", "serial_waits", "serial_wait_s",
            "oracle_hits", "oracle_misses")
COLUMNS = ("dataset", "policy", "n", "accuracy", "deadline_met", "correct_and_on_time", "admitted_correct_and_on_time",
           "rejected_workflows",
           "latency_p50_s", "latency_p90_s", "thinking_tokens_mean", "think", "cut", "announced", "notices",
           "rejected", "pressured_samples", "jev_before_first_token", "jev_during_decode", "jev_tier_lowered",
           "jev_mean_wait_s", "jev_mean_call_s", "jev_failure_rate", "serial_wait_mean_s", "oracle_hit_rate", "paired_n", "online_exact", "jev_exact",
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


@pytest.mark.parametrize("dataset,spec", CASES, ids=[f"{d}:{s}" for d, s in CASES])
def test_policy_under_load(dataset, spec):
    sw = SWEEPS[dataset]
    n, rate = sw["n"], sw["rate"]
    policy, notice, predictor = (spec.split(":") + ["", ""])[:3]
    if predictor == "jev-learn" and "learner" not in gw("GET", "/v1/stats")["jev"]:
        pytest.skip("trained Jev: the online Jev learner is not deployed yet")
    gw("POST", "/admin/policy", json={"policy": policy, "notice": notice or "silent", "predictor": predictor or "jev",
                                        "reject_waiting": REJECT_WAITING, "admit_rate": ADMIT_RATE,
                                        "admit_burst": ADMIT_BURST, "reset": True})
    before = gw("GET", "/v1/stats")
    out = OUT / f"{dataset}-{spec.replace(':', '-')}"
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--dataset", dataset, "--rate", str(rate),
                        "--n", str(n), "--budget", str(sw["budget"]), "--max-budget", str(sw["max_budget"]),
                        "--default-solo-s", str(sw["solo_s"]), "--seed", "0", "--out", str(out)],
                       capture_output=True, text=True, timeout=3600)
    (out / "loadgen.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, p.stderr[-2000:]
    after = gw("GET", "/v1/stats")
    s = json.loads((out / "summary.json").read_text())
    d = {k: after["gateway"][k] - before["gateway"][k] for k in COUNTERS}
    j = {k: after["jev"][k] - before["jev"][k] for k in JEV_COUNTERS}
    hist = [a - b for a, b in zip(after["pressure"]["p_hist_deciles"], before["pressure"]["p_hist_deciles"])]
    oracle = d["oracle_hits"] + d["oracle_misses"]
    row = {"dataset": dataset, "policy": spec, **{k: s.get(k) for k in COLUMNS if k in s}, "rejected_workflows": s.get("rejected"),
           **d, "pressured_samples": sum(hist[1:]), "observed": observed(after) - observed(before),
           **{f"jev_{k}": j[k] for k in ("before_first_token", "during_decode", "tier_lowered")},
           "jev_mean_wait_s": round(j["wait_s"] / j["asked"], 3) if j["asked"] else None,
           "jev_mean_call_s": call_s(before, after),
           "jev_failure_rate": failure_rate(before, after),
           "serial_wait_mean_s": round(d["serial_wait_s"] / d["serial_waits"], 3) if d["serial_waits"] else None,
           "oracle_hit_rate": round(d["oracle_hits"] / oracle, 3) if oracle else None,
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


def wait_idle(timeout: float = 900) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        p = gw("GET", "/v1/stats")["pressure"]
        if p["waiting"] == 0 and p["running"] == 0:
            return True
        time.sleep(5)
    return False


def set_policy(spec: str) -> tuple[str, str, str]:
    policy, notice, predictor = (spec.split(":") + ["", ""])[:3]
    gw("POST", "/admin/policy", json={"policy": policy, "notice": notice or "silent", "predictor": predictor or "jev",
                                        "reject_waiting": REJECT_WAITING, "admit_rate": ADMIT_RATE,
                                        "admit_burst": ADMIT_BURST, "reset": True})
    return policy, notice, predictor


def realistic_loadgen(out: Path, *args: str, timeout: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--workload", WORKLOAD, "--level-budgets",
                        "--seed", str(SEED), "--out", str(out), *args,
                        *(["--model", BASE_MODEL] if BASE_MODEL and "--model" not in args else [])],
                       capture_output=True, text=True, timeout=timeout)
    (out / "loadgen.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads((out / "summary.json").read_text())


def pressure_since(t: float, out: Path) -> list[list]:
    r = gw("GET", "/v1/pressure", params={"since": t})
    reps = [x["samples"] for x in r.get("replicas") or [{"samples": r["samples"]}]]
    (out / "pressure.jsonl").write_text("".join(json.dumps([*x, k]) + "\n" for k, xs in enumerate(reps) for x in xs))
    return reps


def pressure_of(reps: list[list], phase_of=None) -> dict:
    out = metrics.pressure_summary([x for xs in reps for x in xs], phase_of)
    if len(reps) > 1:
        out["replicas"] = [metrics.pressure_summary(xs, phase_of) for xs in reps]
    return out


def word_model(stats: dict) -> dict:
    vs = list(stats["classifier"]["versions"].values())
    v = max(vs, key=lambda x: x["window"]) if vs else {}
    return {f"word_model_{k}": round(v[k], 4) if isinstance(v.get(k), float) else v.get(k)
            for k in ("window", "exact", "within_one", "under", "mean_abs_error")}


def write_realistic() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"suite": "realistic", "workload": WORKLOAD, "profile": None if TRACE else PROFILE,
                                                  "trace_args": TRACE_ARGS,
                                                  "seed": SEED, "capacity": CAPACITY, "runs": RESULTS}, indent=2))
    cap = CAPACITY.get("capacity_wf_s")
    arrivals = (f"trace `{' '.join(TRACE_ARGS)}` (profile from the trace, rates x C: `{RESULTS[0]['profile']}`)"
                if TRACE and RESULTS else f"trace `{' '.join(TRACE_ARGS)}`" if TRACE
                else f"profile `{PROFILE}` (rates x C)")
    lines = [f"Workload `{WORKLOAD}`, {arrivals}, seed {SEED}.", "",
             f"Capacity C = {cap} workflows/s ({CAPACITY.get('source', 'not measured')}; "
             f"{CAPACITY.get('capacity_think_tok_s')} thinking tokens/s over {CAPACITY.get('replicas')} vLLM "
             f"replica(s), {CAPACITY.get('window_n')} workflows "
             f"in the window, queue present in {CAPACITY.get('queue_frac')} of samples).", "",
             "| " + " | ".join(REAL_COLUMNS) + " |", "|" + "---|" * len(REAL_COLUMNS)]
    for r in RESULTS:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in REAL_COLUMNS) + " |")
    bands = sorted({b for r in RESULTS for b in r["by_band"]})
    if bands:
        lines += ["", "Correct and on time (accuracy, n) by band:", "",
                  "| policy | " + " | ".join(bands) + " |", "|" + "---|" * (len(bands) + 1)]
        for r in RESULTS:
            cells = []
            for b in bands:
                x = r["by_band"].get(b)
                cells.append(f"{x.get('correct_and_on_time')} ({x.get('accuracy')}, {x['n']})" if x else "")
            lines.append(f"| {r['policy']} | " + " | ".join(cells) + " |")
        lines += ["", "Pressure histogram (fraction of samples per decile of p; the last one includes p = 1):", "",
                  "| policy | " + " | ".join(f"{k / 10:.1f}" for k in range(10)) + " |", "|" + "---|" * 11]
        for r in RESULTS:
            lines.append(f"| {r['policy']} | " + " | ".join(str(h) for h in r["pressure"].get("p_hist_deciles", []))
                         + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(not REALISTIC, reason="set BROWNOUT_SUITE=realistic")
def test_realistic_capacity():
    set_policy("none")
    assert wait_idle(), "vLLM did not go idle before the capacity run"
    if os.environ.get("BROWNOUT_CAPACITY"):
        CAPACITY.update({"capacity_wf_s": float(os.environ["BROWNOUT_CAPACITY"]), "source": "BROWNOUT_CAPACITY"})
        write_realistic()
        return
    before = gw("GET", "/v1/stats")
    out = OUT / "capacity"
    k = int(CAP_CONCURRENCY) if CAP_CONCURRENCY else 32 * len(before.get("replicas") or [1])
    s = realistic_loadgen(out, "--closed", str(k), "--warmup-s", str(CAP_WARMUP_S),
                          "--duration-s", str(CAP_S), timeout=CAP_WARMUP_S + CAP_S + 900)
    after = gw("GET", "/v1/stats")
    ps = pressure_of(pressure_since(s["t0"] + CAP_WARMUP_S, out))
    CAPACITY.update({"source": "measured", **{k: s.get(k) for k in (
        "capacity_wf_s", "capacity_think_tok_s", "concurrency", "warmup_s", "duration_s", "accuracy",
        "latency_p50_s", "latency_p95_s", "thinking_tokens_mean", "by_band")},
        "window_n": s["n"], "queue_frac": ps.get("queue_frac"), "pressure": ps,
        "replicas": len(after.get("replicas") or [1])})
    write_realistic()
    assert after["gateway"]["upstream_errors"] == before["gateway"]["upstream_errors"]
    assert CAPACITY["capacity_wf_s"] > 0, "no workflow finished inside the capacity window"
    assert wait_idle(), "vLLM did not go idle after the capacity run"


@pytest.mark.parametrize("spec", REAL_SPECS)
def test_realistic_policy(spec):
    if not CAPACITY.get("capacity_wf_s"):
        pytest.fail("no capacity: test_realistic_capacity failed or did not run")
    cap = CAPACITY["capacity_wf_s"]
    spec, _, model = spec.partition("@")
    if spec.split(":")[2:3] == ["jev-learn"] and "learner" not in gw("GET", "/v1/stats")["jev"]:
        pytest.skip("trained Jev: the online Jev learner is not deployed yet")
    wait_idle()
    policy = set_policy(spec)[0]
    before = gw("GET", "/v1/stats")
    out = OUT / f"realistic-{spec.replace(':', '-')}{'-' + model if model else ''}"
    arrivals = [*(TRACE_ARGS or ["--profile", PROFILE]), *(["--model", model] if model else [])]
    span = TRACE_S if TRACE else workload.total_s(workload.parse_profile(PROFILE))
    s = realistic_loadgen(out, *arrivals, "--capacity", str(cap), "--tail-s", str(TAIL_S),
                          timeout=span + TAIL_S + 900)
    profile = workload.parse_profile(s["profile"])
    after = gw("GET", "/v1/stats")
    ps = pressure_of(pressure_since(s["t0"], out), lambda t: workload.phase_at(profile, t - s["t0"]))
    d = {k: after["gateway"][k] - before["gateway"][k] for k in COUNTERS}
    ph = s.get("by_phase", {})
    row = {"policy": spec + (f"@{model}" if model else ""), "model": model or None, "workload": WORKLOAD, "capacity_wf_s": cap, "profile": s["profile"],
           "trace": s.get("trace"),
           **{k: s.get(k) for k in ("n", "accuracy", "deadline_met", "correct_and_on_time", "cancelled",
                                     "latency_p50_s", "latency_p95_s", "thinking_tokens_mean", "wall_s")},
           **{f"goodput_{k}": v.get("correct_and_on_time") for k, v in ph.items()},
           **{f"accuracy_{k}": v.get("accuracy") for k, v in ph.items()},
           **{f"n_{k}": v.get("n") for k, v in ph.items()},
           **d, "observed": observed(after) - observed(before),
           **{k: ps.get(k) for k in ("p_mean", "p_at_cap", "p_zero", "queue_frac")},
           **{f"p_mean_{k}": v["p_mean"] for k, v in ps.get("by_phase", {}).items()},
           **word_model(after), "jev_asked": after["jev"]["asked"] - before["jev"]["asked"],
           "replica_think": [a["think"] - b["think"] for a, b in zip(after.get("replicas", []),
                                                                     before.get("replicas", []))],
           "p_mean_replicas": [x["p_mean"] for x in ps.get("replicas", []) if "p_mean" in x],
           "by_band": s.get("by_band", {}), "by_phase": ph, "by_segment": s.get("by_segment", {}),
           "by_band_phase": s.get("by_band_phase", {}), "pressure": ps}
    RESULTS.append(row)
    write_realistic()
    assert s["errors"] == 0 and d["upstream_errors"] == 0
    assert d["think"] - s["cancelled"] <= row["observed"] <= d["think"]
    if policy == "none":
        assert d["cut"] == d["announced"] == d["rejected"] == 0
    if after["classifier"]["jev"] is None:
        assert row["jev_asked"] == 0


def test_jev_answered_and_classifier_learned():
    st = gw("GET", "/v1/stats")
    (OUT / "gateway-stats.json").write_text(json.dumps(st, indent=2))
    jev = st["classifier"]["jev"]
    if not (REALISTIC and jev is None):
        assert jev is not None and jev["calls"] > jev["failures"]
    assert observed(st) > 0
