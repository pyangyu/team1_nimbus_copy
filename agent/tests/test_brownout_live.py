import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("BROWNOUT_LIVE"),
                                reason="set BROWNOUT_LIVE=1 with LIFELINE_BASE_URL pointing at the Brownout gateway")

BASE = os.environ.get("LIFELINE_BASE_URL", "").rstrip("/")
ROOT = BASE.removesuffix("/v1")
KEY = os.environ.get("LIFELINE_API_KEY", "")
OUT = Path(os.environ.get("BROWNOUT_OUT", "runs/brownout-live"))
SPECS = os.environ.get("BROWNOUT_POLICIES", "none,reject,uniform,brownout,brownout:announced").split(",")
N = int(os.environ.get("BROWNOUT_N", "24"))
RATE = float(os.environ.get("BROWNOUT_RATE", "1.0"))
DATASET = os.environ.get("BROWNOUT_DATASET", "gsm8k")
REJECT_WAITING = float(os.environ.get("BROWNOUT_REJECT_WAITING", "2"))
COUNTERS = ("think", "rejected", "deferred", "cut", "announced", "upstream_errors")
COLUMNS = ("policy", "n", "accuracy", "deadline_met", "correct_and_on_time", "rejected_workflows",
           "latency_p50_s", "latency_p90_s", "thinking_tokens_mean", "think", "cut", "announced",
           "rejected", "pressured_samples")
RESULTS: list[dict] = []


def gw(method: str, path: str, **kw) -> dict:
    r = httpx.request(method, ROOT + path, headers={"Authorization": f"Bearer {KEY}"}, timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def observed(stats: dict) -> int:
    return sum(v["labels"] + v["censored"] for v in stats["classifier"]["versions"].values())


def write_results() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in RESULTS:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in COLUMNS) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("spec", SPECS)
def test_policy_under_load(spec):
    policy, _, notice = spec.partition(":")
    gw("POST", "/admin/policy", json={"policy": policy, "notice": notice or "silent", "reject_waiting": REJECT_WAITING})
    before = gw("GET", "/v1/stats")
    out = OUT / spec.replace(":", "-")
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--dataset", DATASET, "--rate", str(RATE),
                        "--n", str(N), "--budget", "1024", "--max-budget", "2048", "--default-solo-s", "90",
                        "--seed", "0", "--out", str(out)], capture_output=True, text=True, timeout=3600)
    (out / "loadgen.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, p.stderr[-2000:]
    after = gw("GET", "/v1/stats")
    s = json.loads((out / "summary.json").read_text())
    d = {k: after["gateway"][k] - before["gateway"][k] for k in COUNTERS}
    hist = [a - b for a, b in zip(after["pressure"]["p_hist_deciles"], before["pressure"]["p_hist_deciles"])]
    row = {"policy": spec, **{k: s.get(k) for k in COLUMNS if k in s}, "rejected_workflows": s.get("rejected"),
           **d, "pressured_samples": sum(hist[1:]), "observed": observed(after) - observed(before)}
    RESULTS.append(row)
    write_results()
    assert s["n"] == N and s["errors"] == 0 and d["upstream_errors"] == 0
    assert d["think"] > 0 and row["observed"] == d["think"]
    if policy == "none":
        assert d["cut"] == d["announced"] == d["rejected"] == 0
    if policy in ("uniform", "brownout") and row["pressured_samples"] >= 3:
        assert d["cut"] + d["announced"] > 0
    if policy != "reject":
        assert s["rejected"] == 0


def test_jev_answered_and_classifier_learned():
    st = gw("GET", "/v1/stats")
    jev = st["classifier"]["jev"]
    assert jev is not None and jev["calls"] > jev["failures"]
    assert observed(st) > 0
    (OUT / "gateway-stats.json").write_text(json.dumps(st, indent=2))
