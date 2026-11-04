import json
import subprocess
import sys

import pytest
from test_agent_e2e import PROBLEMS, agent, data_dir  # noqa: F401
from vllm_stub import StubVLLM

from math_agent import metrics, trace, workload
from math_agent.datasets import Example
from math_agent.workload import Component, Segment


def test_parse_workload_single_and_mix():
    assert workload.parse_workload("math500:1-5") == [Component("math500", (1, 2, 3, 4, 5), 1.0)]
    assert workload.parse_workload("math500:3") == [Component("math500", (3,), 1.0)]
    assert workload.parse_workload("gsm8k=0.5, math500:4-5=0.5") == [
        Component("gsm8k", None, 0.5), Component("math500", (4, 5), 0.5)]
    assert workload.parse_workload("aime2026") == [Component("aime2026", None, 1.0)]


@pytest.mark.parametrize("bad", ["", "nope", "math500:0-2", "math500:4-2", "math500:6", "gsm8k:1-2",
                                 "math500=0", "math500:x"])
def test_parse_workload_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        workload.parse_workload(bad)


def test_budget_by_problem():
    lvl = lambda n: Example(f"m/{n}", "q", "1", "math", n)
    assert [workload.budget_for(lvl(n))[0] for n in range(1, 6)] == [2048, 2048, 4096, 6144, 6144]
    assert all(workload.budget_for(lvl(n))[1] >= workload.budget_for(lvl(n))[0] for n in range(1, 6))
    assert workload.budget_for(Example("gsm8k/0", "q", "1", "gsm8k"))[:2] == (1024, 2048)
    assert workload.budget_for(Example("aime_2026/1", "q", "1", "math"))[0] == 6144
    assert workload.band(lvl(3)) == "L3" and workload.band(Example("g", "q", "1", "gsm8k")) == "gsm8k"
    assert workload.band(Example("aime_2026/1", "q", "1", "math")) == "aime"


def test_sampler_mix_is_seeded_and_weighted():
    a = [Example(f"a/{i}", "q", "1", "gsm8k") for i in range(5)]
    b = [Example(f"b/{i}", "q", "1", "math", 5) for i in range(5)]
    comps = [Component("gsm8k", None, 3.0), Component("math500", (5,), 1.0)]

    def draw(seed):
        s = workload.Sampler(comps, [a, b], seed)
        return [s.draw().id for _ in range(2000)]
    assert draw(0) == draw(0) and draw(0) != draw(1)
    frac_a = sum(x.startswith("a/") for x in draw(0)) / 2000
    assert 0.7 < frac_a < 0.8


def test_parse_profile_and_phases():
    p = workload.parse_profile("60:0.7,30:1.5, 90:0.7")
    assert p == [Segment(60, 0.7), Segment(30, 1.5), Segment(90, 0.7)] and workload.total_s(p) == 180
    assert [s.phase for s in p] == ["baseline", "burst", "baseline"]
    assert [workload.segment_at(p, t) for t in (-1, 0, 59.9, 60, 89.9, 90, 179.9, 180)] == \
        [None, 0, 0, 1, 1, 2, 2, None]
    assert [workload.phase_at(p, t) for t in (10, 75, 100, 500)] == ["baseline", "burst", "baseline", "tail"]
    assert workload.parse_profile(workload.DEFAULT_PROFILE)
    for bad in ("", "60", "0:1", "60:-1"):
        with pytest.raises(ValueError):
            workload.parse_profile(bad)


def test_default_profile_shape():
    p = workload.parse_profile(workload.DEFAULT_PROFILE)
    assert 25 * 60 <= workload.total_s(p) <= 30 * 60
    assert {s.rate for s in p if s.phase == "burst"} == {1.2, 1.5, 2.0}
    assert all(60 <= s.duration_s <= 240 for s in p if s.phase == "burst")
    assert p[0].phase == p[-1].phase == "baseline"


def test_arrivals_are_seeded_and_follow_segment_rates():
    p = [Segment(2000, 0.5), Segment(1000, 2.0), Segment(500, 0.0)]
    a = workload.arrivals(p, 1.0, seed=0)
    assert a == workload.arrivals(p, 1.0, seed=0) and a != workload.arrivals(p, 1.0, seed=1)
    assert all(x[0] < y[0] for x, y in zip(a, a[1:]))
    assert all(workload.segment_at(p, t) == k for t, k in a)
    counts = [sum(k == i for _, k in a) for i in range(3)]
    assert 900 < counts[0] < 1100 and 1850 < counts[1] < 2150 and counts[2] == 0
    half = workload.arrivals(p, 0.5, seed=0)
    assert 1700 / 2 < len(half) < 3300 / 2


def test_breakdown_and_cancelled_workflows():
    w = lambda phase, correct, met, cancelled=False: {
        "phase": phase, "correct": correct, "met_deadline": met, "latency_s": 10.0, "rounds": 1, "llm_calls": 3,
        "prompt_tokens": 1, "thinking_tokens": 100, "answer_tokens": 1, "cancelled": cancelled,
        "error": "cancelled" if cancelled else None}
    rows = [w("baseline", True, True), w("baseline", True, False), w("burst", False, False, True)]
    s = metrics.summarize(rows)
    assert s["n"] == 3 and s["cancelled"] == 1 and s["errors"] == 0 and s["thinking_tokens_mean"] == 100
    assert s["correct_and_on_time"] == round(1 / 3, 4)
    b = metrics.breakdown(rows, "phase")
    assert b["baseline"]["correct_and_on_time"] == 0.5 and b["baseline"]["n"] == 2
    assert b["burst"]["n"] == 1 and b["burst"]["accuracy"] == 0 and "thinking_tokens_mean" not in b["burst"]


def test_pressure_summary():
    samples = [[0, 0.0, 0, 2], [1, 0.05, 1, 20], [2, 0.55, 5, 9], [3, 1.0, 12, 8]]
    s = metrics.pressure_summary(samples, lambda t: "burst" if t >= 2 else "baseline")
    assert s["p_samples"] == 4 and s["p_zero"] == 0.25 and s["p_at_cap"] == 0.25 and s["queue_frac"] == 0.75
    assert s["p_hist_deciles"] == [0.5, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.25]
    assert s["by_phase"]["burst"]["p_mean"] == 0.775 and s["by_phase"]["baseline"]["p_at_cap"] == 0
    assert metrics.pressure_summary([]) == {"p_samples": 0} and "tpot_ms_mean" not in s
    timed = metrics.pressure_summary([[0, 0.0, 0, 2, None], [1, 0.25, 0, 40, 50.0], [2, 0.0, 0, 30, 30.0]])
    assert timed["tpot_ms_mean"] == 40.0


def test_loadgen_profile_mode_is_reproducible(data_dir, tmp_path):
    args = ("--workload", "math500:5", "--level-budgets", "--profile", "1:0.5,1:3", "--capacity", "4",
            "--tail-s", "60", "--slack", "2,4", "--seed", "3", "--max-loops", "0")
    runs = []
    for k in range(2):
        with StubVLLM(PROBLEMS) as stub:
            wfs, _, summary = agent("loadgen", stub, data_dir, tmp_path / f"p{k}", *args)
            budgets = {r["headers"].get("x-lifeline-thinking-budget") for r in stub.requests
                       if r["headers"].get("x-lifeline-phase") == "think"}
        runs.append((sorted(wfs, key=lambda w: w["workflow_id"]), summary))
        assert budgets == {"6144"}
    (a, sa), (b, sb) = runs
    key = lambda ws: [(w["workflow_id"], w["example_id"], w["segment"], w["offset_s"], w["slack"]) for w in ws]
    assert key(a) == key(b) and sa["arrivals"] == len(a) > 0
    assert {w["phase"] for w in a} <= {"baseline", "burst"} and all(w["band"] == "L5" for w in a)
    assert all(w["phase"] == ("burst" if w["segment"] == 1 else "baseline") for w in a)
    assert sa["mode"] == "profile" and set(sa["by_phase"]) == {w["phase"] for w in a}
    assert sa["by_band"]["L5"]["n"] == len(a) and sa["errors"] == 0


def test_loadgen_closed_mode_reports_capacity(data_dir, tmp_path):
    with StubVLLM(PROBLEMS, delay_s=0.05) as stub:
        wfs, _, s = agent("loadgen", stub, data_dir, tmp_path / "c", "--workload", "math500:5", "--level-budgets",
                          "--closed", "3", "--warmup-s", "0.5", "--duration-s", "2", "--max-loops", "0")
    done = [w for w in wfs if not w.get("cancelled")]
    assert s["mode"] == "closed" and s["concurrency"] == 3 and s["started"] == len(wfs)
    assert s["capacity_wf_s"] == round(s["n"] / 2, 5) and s["n"] <= len(done)
    assert s["capacity_wf_s"] > 0 and s["capacity_think_tok_s"] > 0
    assert json.loads((tmp_path / "c" / "summary.json").read_text())["capacity_wf_s"] == s["capacity_wf_s"]


def test_loadgen_needs_one_arrival_mode():
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--n", "2"], capture_output=True, text=True)
    assert p.returncode != 0 and "exactly one of" in p.stderr
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--profile", "1:1"], capture_output=True, text=True)
    assert p.returncode != 0 and "--capacity" in p.stderr


def burstgpt_csv(data_dir, rows):
    d = data_dir / "burstgpt"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type"]
    lines += [f"{t},ChatGPT,10,{a},{10 + a},{lt}" for t, a, lt in rows]
    (d / "BurstGPT_1.csv").write_text("\n".join(lines) + "\n")


def test_trace_replay_keeps_burst_shape_and_weaves_levels(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFELINE_DATA", str(tmp_path))
    rows = [(t, 100 + t % 7, "Conversation log") for t in range(0, 3600, 10)]
    rows += [(t, 900 + t % 5, "Conversation log") for t in range(1800, 2160, 2)]
    rows += [(t, 0, "Conversation log") for t in range(5)] + [(t, 50, "API log") for t in range(0, 3600, 3)]
    burstgpt_csv(tmp_path, sorted(rows))
    trace.load.cache_clear()
    reqs = trace.load("conversation")
    assert len(reqs) == 360 + 180 and all(r.log == "Conversation log" and r.response_tokens > 0 for r in reqs)
    pool = [Example(f"m/{lv}/{i}", "q", "1", "math", lv) for lv in (1, 3, 5) for i in range(lv * 4)]
    kw = dict(start_h=0, window_h=1, duration_s=600, load=0.8, bin_s=60, seed=0)
    rp = trace.replay(reqs, pool, 0.5, **kw)
    again = trace.replay(reqs, pool, 0.5, **kw)
    assert [e.id for e in rp.examples] == [e.id for e in again.examples] and rp.arrivals == again.arrivals
    assert [e.id for e in trace.replay(reqs, pool, 0.5, **{**kw, "seed": 1}).examples] != [e.id for e in rp.examples]
    assert len(rp.arrivals) == 240 and rp.window_n == 540
    assert all(0 <= a < b < 600 for (a, _), (b, _) in zip(rp.arrivals, rp.arrivals[1:]))
    assert all(seg == int(off // 60) for off, seg in rp.arrivals)
    assert workload.total_s(rp.profile) == 600 and len(rp.profile) == 10
    assert [s.phase for s in rp.profile] == ["baseline"] * 5 + ["burst"] + ["baseline"] * 4
    assert abs(sum(s.rate * s.duration_s for s in rp.profile) / 600 - 0.8) < 0.01
    lv = [e.level for e in rp.examples]
    assert {k: lv.count(k) for k in (1, 3, 5)} == trace.level_quotas(pool, 240) == {1: 27, 3: 80, 5: 133}
    by_len = sorted(zip((r.response_tokens for r in rp.requests), lv))
    assert [l for _, l in by_len] == sorted(lv)
    assert all(l == 5 for a, l in zip(rp.requests, lv) if a.response_tokens >= 900)
    assert trace.spec(rp.profile).startswith("60:")


def test_trace_replay_rejects_too_few_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFELINE_DATA", str(tmp_path))
    burstgpt_csv(tmp_path, [(t, 100, "Conversation log") for t in range(0, 3600, 60)])
    trace.load.cache_clear()
    pool = [Example("m/1", "q", "1", "math", 1)]
    with pytest.raises(ValueError, match="lower the load"):
        trace.replay(trace.load(), pool, 1.0, start_h=0, window_h=1, duration_s=600, load=1.0)
    with pytest.raises(ValueError):
        trace.load("nope")


def test_loadgen_trace_mode(data_dir, tmp_path):
    burstgpt_csv(data_dir, sorted([(t, 100 + t % 50, "Conversation log") for t in range(0, 7200, 20)]
                                  + [(t, 700, "Conversation log") for t in range(3000, 3300, 3)]))
    args = ("--workload", "math500:5", "--level-budgets", "--trace", "burstgpt", "--trace-start-h", "0",
            "--trace-s", "3", "--load", "2", "--bin-s", "1", "--capacity", "1", "--tail-s", "60",
            "--seed", "3", "--max-loops", "0")
    with StubVLLM(PROBLEMS) as stub:
        wfs, _, s = agent("loadgen", stub, data_dir, tmp_path / "t", *args)
    assert s["mode"] == "trace" and s["arrivals"] == len(wfs) == 6 and s["errors"] == 0
    assert s["trace"]["window_n"] == 460 and s["trace"]["compression"] == 2400
    assert workload.total_s(workload.parse_profile(s["profile"])) == 3
    assert all("trace_t" in w and w["band"] == "L5" for w in wfs)
    p = subprocess.run([sys.executable, "-m", "math_agent.loadgen", "--trace", "burstgpt", "--capacity", "1",
                        "--workload", "gsm8k"], capture_output=True, text=True)
    assert p.returncode != 0 and "MATH-500" in p.stderr


def test_split_halves_a_pool_by_problem_id():
    pool = [Example(f"m/{i}", "q", "1", "math", 1 + i % 5) for i in range(400)]
    a, b = workload.split(pool, "a"), workload.split(pool, "b")
    assert {e.id for e in a}.isdisjoint(e.id for e in b) and len(a) + len(b) == len(pool)
    assert 150 < len(a) < 250 and workload.split(pool[::-1], "a") == a[::-1]
    with pytest.raises(ValueError):
        workload.split(pool, "c")


def test_loadgen_split_keeps_one_half(data_dir, tmp_path):
    seen = {}
    for part in workload.SPLITS:
        with StubVLLM(PROBLEMS) as stub:
            wfs, _, s = agent("loadgen", stub, data_dir, tmp_path / part, "--workload", "math500:5", "--level-budgets",
                              "--rate", "50", "--n", "8", "--split", part, "--max-loops", "0")
        seen[part] = {w["example_id"] for w in wfs}
        assert s["errors"] == 0 and len(wfs) == 8
    assert seen["b"] == {"easy"} and seen["a"] and "easy" not in seen["a"]
