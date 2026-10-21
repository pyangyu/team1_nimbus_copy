import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from vllm_stub import API_KEY, StubVLLM

from math_agent.llm import FINAL_PREFIX, NOTICES

EASY = "What is 6 times 7?"
RETRY = "What is 3 times 14 plus zero?"
STUBBORN = "What is 2 times 21 minus zero?"
LONG = "What is 7 times 6, worked out the long way?"
PROBLEMS = {
    EASY: {"id": "easy", "gold": "42", "think": 30},
    RETRY: {"id": "retry", "gold": "42", "think": 30, "wrong_first": ["41"]},
    STUBBORN: {"id": "stubborn", "gold": "42", "think": 30, "wrong_first": ["40", "41"]},
    LONG: {"id": "long", "gold": "42", "think": 5000},
}


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    (d / "math").mkdir(parents=True)
    rows = [{"unique_id": p["id"], "problem": q, "answer": p["gold"], "level": 5} for q, p in PROBLEMS.items()]
    (d / "math" / "math500.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def agent(module: str, stub: StubVLLM, data_dir: Path, out: Path, *args: str, api_key: str = API_KEY):
    env = {**os.environ, "LIFELINE_BASE_URL": stub.base_url, "LIFELINE_API_KEY": api_key,
           "LIFELINE_DATA": str(data_dir)}
    env.pop("LIFELINE_MODEL", None)
    p = subprocess.run([sys.executable, "-m", f"math_agent.{module}", "--out", str(out), *args],
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    wfs = [json.loads(l) for l in (out / "workflows.jsonl").read_text().splitlines()]
    calls = [json.loads(l) for l in (out / "calls.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    return wfs, calls, summary


def steps(reqs):
    return [(r["headers"]["x-lifeline-step"], r["headers"]["x-lifeline-phase"]) for r in reqs]


def test_solo_run_plans_solves_verifies_and_retries(data_dir, tmp_path):
    with StubVLLM(PROBLEMS) as stub:
        wfs, calls, summary = agent("run", stub, data_dir, tmp_path / "solo", "--n", "3", "--levels", "5",
                                    "--seed", "1")
    by = {w["example_id"]: w for w in wfs}
    assert set(by) <= {"easy", "retry", "stubborn", "long"} and len(by) == 3
    assert summary["n"] == 3 and summary["errors"] == 0
    assert len(calls) == sum(w["llm_calls"] for w in wfs)
    if "easy" in by:
        assert by["easy"]["correct"] and by["easy"]["rounds"] == 1 and by["easy"]["confidence"] == 0.9
    if "retry" in by:
        assert by["retry"]["correct"] and by["retry"]["rounds"] == 2
    if "stubborn" in by:
        assert not by["stubborn"]["correct"] and by["stubborn"]["rounds"] == 2 and by["stubborn"]["pred"] == "41"


def test_each_step_feeds_the_next(data_dir, tmp_path):
    with StubVLLM(PROBLEMS) as stub:
        wfs, _, _ = agent("run", stub, data_dir, tmp_path / "retry", "--id", "retry")
        reqs = stub.requests
    assert wfs[0]["correct"] and wfs[0]["rounds"] == 2
    assert steps(reqs) == [("plan", "answer"),
                           ("solve.0", "think"), ("solve.0", "answer"),
                           ("verify.0", "think"), ("verify.0", "answer"),
                           ("solve.1", "think"), ("solve.1", "answer")]
    assert len({r["headers"]["x-lifeline-workflow"] for r in reqs}) == 1
    solve0 = reqs[1]["body"]["prompt"]
    verify0 = reqs[3]["body"]["prompt"]
    solve1 = reqs[5]["body"]["prompt"]
    assert "Use multiplication." in solve0 and RETRY in solve0
    assert "Proposed final answer: 41" in verify0
    assert "Earlier attempts answered 41" in solve1
    budgets = [r["headers"]["x-lifeline-thinking-budget"] for r in reqs if r["headers"]["x-lifeline-phase"] == "think"]
    assert budgets == ["4096", "512", "6144"]


def test_more_rounds_let_it_recover_from_two_rejections(data_dir, tmp_path):
    with StubVLLM(PROBLEMS) as stub:
        wfs, _, _ = agent("run", stub, data_dir, tmp_path / "stubborn", "--id", "stubborn", "--max-loops", "2")
        solve2 = [r for r in stub.requests if r["headers"]["x-lifeline-step"] == "solve.2"][0]["body"]["prompt"]
    assert wfs[0]["correct"] and wfs[0]["rounds"] == 3
    assert "Earlier attempts answered 40, 41" in solve2


def test_budget_drop_with_notice_and_final_answer_only(data_dir, tmp_path):
    with StubVLLM(PROBLEMS) as stub:
        wfs, calls, _ = agent("run", stub, data_dir, tmp_path / "drop", "--id", "long", "--max-loops", "0",
                              "--drop-at", "1000", "--drop-to", "200", "--notice", "count",
                              "--answer-mode", "final")
        solve = [r for r in stub.requests if r["headers"]["x-lifeline-step"] == "solve.0"]
    think1, think2, answer = solve
    assert think1["body"]["max_tokens"] == 1000 and "x-lifeline-budget-drop" not in think1["headers"]
    assert think2["body"]["max_tokens"] == 200 and think2["headers"]["x-lifeline-budget-drop"] == "1000,200,count"
    assert think2["body"]["prompt"].endswith(NOTICES["count"].format(W=200))
    assert answer["body"]["prompt"].endswith(FINAL_PREFIX) and answer["body"]["max_tokens"] == 48
    assert wfs[0]["correct"] and wfs[0]["thinking_tokens"] == 1200
    call = next(c for c in calls if c["step"] == "solve.0")
    assert call["dropped"] and call["closed_itself"] is False and call["budget_exhausted"]


def test_wrong_api_key_is_reported_not_crashed(data_dir, tmp_path):
    with StubVLLM(PROBLEMS) as stub:
        env = {**os.environ, "LIFELINE_BASE_URL": stub.base_url, "LIFELINE_API_KEY": "wrong",
               "LIFELINE_DATA": str(data_dir), "LIFELINE_MODEL": "stub-qwen3"}
        p = subprocess.run([sys.executable, "-m", "math_agent.run", "--id", "easy", "--out", str(tmp_path / "bad")],
                           env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    w = json.loads((tmp_path / "bad" / "workflows.jsonl").read_text())
    assert "401" in w["error"] and not w["correct"]


def test_loadgen_runs_workflows_concurrently_with_deadlines(data_dir, tmp_path):
    with StubVLLM(PROBLEMS, delay_s=0.05) as stub:
        wfs, _, summary = agent("loadgen", stub, data_dir, tmp_path / "load", "--rate", "50", "--n", "6",
                                "--slack", "1.5,3", "--default-solo-s", "30", "--max-loops", "1")
        reqs = stub.requests
    assert {w["workflow_id"] for w in wfs} == {f"w{i:05d}" for i in range(6)}
    assert all("x-lifeline-deadline" in r["headers"] for r in reqs if r["headers"].get("x-lifeline-step"))
    order = [r["headers"]["x-lifeline-workflow"] for r in reqs]
    switches = sum(a != b for a, b in zip(order, order[1:]))
    assert switches > len(set(order)) - 1, "workflows ran one after another, not concurrently"
    assert summary["n"] == 6 and summary["deadline_met"] == 1.0 and set(summary["by_slack"]) <= {"1.5", "3.0"}
