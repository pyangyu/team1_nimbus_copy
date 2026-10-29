"""The settled judge: Jev re-asked mid-decode, and the settled cut mode."""
import asyncio
import json

import httpx
import pytest

from brownout_gateway.controller import Active, Controller
from brownout_gateway.judge import JudgeScheduler
from brownout_gateway.pressure import PressureMonitor
from budget_classifier import Buckets, JevPrior
from conftest import body, make, parse, think_headers


def onehot(k, n=10, p=0.9):
    return [p if i == k else (1 - p) / (n - 1) for i in range(n)]


def fake_judge(seen, settled_from: int, tier=5, delay=0.0):
    """Answers the tier/priority questions like the other fakes; the settled questions say 'settled' once the
    thinking so far holds at least settled_from tokens (the bench tells the fake how many via the state)."""
    async def handler(req):
        if delay:
            await asyncio.sleep(delay)
        b = json.loads(req.content)
        seen.append(b)
        ans = {}
        for q, d in b["questions"].items():
            if d["type"] == "noul":
                ans[q] = {"type": "noul", "noul": 0.85 if b["state"]["thinking_so_far"]["tokens"] >= settled_from else 0.1}
            elif q == "settled":
                hi = b["state"]["thinking_so_far"]["tokens"] >= settled_from
                ans[q] = {"type": "score", "probabilities": {"0": 0.02, "1": 0.08, "2": 0.3, "3": 0.6} if hi
                          else {"0": 0.5, "1": 0.4, "2": 0.08, "3": 0.02}}
            else:
                n = len(d["criteria"])
                pick = tier if q == "tier" else 1
                ans[q] = {"type": "score", "probabilities": {str(i): (0.9 if i == pick else 0.1 / (n - 1)) for i in range(n)}}
        return httpx.Response(200, json={"answers": ans})
    return httpx.MockTransport(handler)


def test_settled_mode_cuts_the_judged_request_first():
    ctl = Controller(PressureMonitor(), policy="brownout", mode="settled", interval=1)
    ctl.monitor.set(5, 10)
    a = ctl.start(Active("a", 1000, 1000, "normal", 1000))
    b = ctl.start(Active("b", 1000, 1000, "normal", 1000))
    for x in (a, b):
        x.set_probs(onehot(6), "test")
        x.used = 300
    a.settled = 0.95
    ctl.tick()
    # 1000 tokens to shed; a's remaining thinking costs 5% of b's, so a is cut to its floor first
    assert a.limit == 364 and b.limit == 636  # a gave 636 of the 1000; b pays the remaining 364
    assert ctl.ranked["settled_cuts"] == 1 and ctl.ranked["exits"] == 0


def test_ranked_mode_ignores_the_verdict():
    def run(mode, settled):
        ctl = Controller(PressureMonitor(), policy="brownout", mode=mode, interval=1)
        ctl.monitor.set(5, 10)
        a = ctl.start(Active("a", 1000, 1000, "normal", 1000))
        b = ctl.start(Active("b", 1000, 1000, "normal", 1000))
        a.set_probs(onehot(6), "test")
        b.set_probs(onehot(2), "test")  # b needs 128-256, below the 300 used: ranked cuts b first
        a.used = b.used = 300
        a.settled = settled
        ctl.tick()
        return a.limit, b.limit
    assert run("ranked", 0.95) == run("ranked", None) == (636, 364)
    assert run("settled", 0.95) == (364, 636), "the verdict overrides the prompt-level order in settled mode"


def test_exit_at_cuts_a_settled_request_outright():
    ctl = Controller(PressureMonitor(), policy="brownout", mode="settled", interval=1, exit_at=0.8)
    ctl.monitor.set(1, 10)  # p = 0.1: the shed target is only 10% of the budgets
    a = ctl.start(Active("a", 4000, 4000, "normal", 4000))
    b = ctl.start(Active("b", 4000, 4000, "normal", 4000))
    for x in (a, b):
        x.set_probs(onehot(7), "test")
        x.used = 1200
    a.settled, b.settled = 0.9, 0.5
    ctl.tick()
    assert a.limit == 1264 and ctl.ranked["exits"] == 1
    assert b.limit == 4000, "the exited request's tokens cover the shed target; b is spared"
    ctl.monitor.set(0, 10)
    ctl.tick()
    assert ctl.ranked["exits"] == 1


def test_judge_asks_jev_as_the_thinking_streams(run, tmp_path):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_judge(seen, settled_from=200))
    gw, fake, c = make(jev=jev, mode="settled", lengths={"hard": 700}, log_path=tmp_path / "gw.jsonl",
                       judge_from=100, judge_every=200)
    gw.monitor.set(1, 10)

    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        for _ in range(100):
            if not gw.judge.jobs and gw.judge.workers == 0:
                break
            await asyncio.sleep(0.01)
        return r
    out = parse(run(go()))
    judged = [b for b in seen if "settled" in b["questions"]]
    assert judged and judged[0]["state"]["thinking_so_far"]["tokens"] == 100
    assert judged[0]["questions"]["done"]["type"] == "noul" and len(judged[0]["questions"]["settled"]["criteria"]) == 4
    assert out["info"]["judgments"] >= 2 and out["info"]["settled"] == pytest.approx(0.9)
    assert out["info"]["judged_at"] >= 300
    st = gw.stats()["judge"]
    assert st["enabled"] and st["answered"] == out["info"]["judgments"] and st["failed"] == 0
    assert st["mean_settled"] is not None
    rows = [json.loads(l) for l in (tmp_path / "gw.jsonl").read_text().splitlines()]
    jrows = [r for r in rows if r.get("judge")]
    assert jrows and jrows[0]["used"] == 100 and jrows[0]["settled"] == pytest.approx(0.1)
    assert [r for r in rows if not r.get("judge")][0]["judgments"] == len(jrows)


def test_no_judging_outside_the_settled_mode(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_judge(seen, settled_from=200))
    gw, fake, c = make(jev=jev, mode="ranked", lengths={"hard": 700}, judge_from=100, judge_every=200)
    out = parse(run(c.post("/v1/completions", json=body("hard"), headers=think_headers())))
    assert not [b for b in seen if "settled" in b["questions"]] and out["info"]["judgments"] == 0
    gw.judge_always = True
    out = parse(run(c.post("/v1/completions", json=body("hard"), headers=think_headers(wf="w2"))))
    assert [b for b in seen if "settled" in b["questions"]] and out["info"]["judgments"] >= 1


def test_latest_judgment_wins_and_finished_requests_are_dropped():
    class GW:
        class clf:
            jev = object()

        class controller:
            active = {"a": None}

        monitor = PressureMonitor()
        log = None

    js = JudgeScheduler(GW(), concurrency=1)
    js.workers = 1  # pretend a worker is busy so nothing runs
    a = Active("a", 1000, 1000, "normal", 1000)
    gone = Active("gone", 1000, 1000, "normal", 1000)
    js.jobs = {}
    js.submit(a, "p", "t1", 100)
    js.submit(a, "p", "t1 t2", 300)
    js.submit(gone, "p", "x", 50)
    assert js.counts["submitted"] == 3 and js.counts["replaced"] == 1 and len(js.jobs) == 2
    job = js._next()
    assert job.a is a and job.thinking == "t1 t2" and js.counts["after_finish"] == 1 and not js.jobs


def test_admin_sets_exit_and_signal(run):
    gw, fake, c = make()
    r = run(c.post("/admin/policy", json={"policy": "brownout", "cut": "settled", "exit_at": 0.8, "judge_signal": "mean"}))
    assert r.status_code == 200 and r.json()["mode"] == "settled" and r.json()["exit_at"] == 0.8
    assert gw.judge.signal == "mean"
    assert run(c.post("/admin/policy", json={"policy": "brownout", "exit_at": 2})).status_code == 400
    assert run(c.post("/admin/policy", json={"policy": "brownout", "judge_signal": "nope"})).status_code == 400
    assert run(c.post("/admin/policy", json={"policy": "brownout", "exit_at": None})).json()["exit_at"] is None


def test_judge_value_follows_the_signal():
    class GW:
        class clf:
            jev = None
    js = JudgeScheduler(GW(), signal="mean")
    assert js.value({"settled": 0.8, "done": 0.4}) == pytest.approx(0.6)
    assert js.value({"settled": 0.8}) == 0.8 and js.value({"done": 0.4}) == 0.4
    js.signal = "done"
    assert js.value({"settled": 0.8, "done": 0.4}) == 0.4
