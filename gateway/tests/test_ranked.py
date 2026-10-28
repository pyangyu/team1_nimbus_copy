"""Ranked cuts and needed-budget labels."""
import asyncio

import pytest

from brownout_gateway.answers import final_answer
from brownout_gateway.controller import Active, Controller
from brownout_gateway.pressure import PressureMonitor
from budget_classifier import Buckets
from conftest import body, make, parse, think_headers


def onehot(k, n=10, p=0.9):
    return [p if i == k else (1 - p) / (n - 1) for i in range(n)]


async def drain(gw, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if not gw.prober.queue and gw.prober.in_flight == 0:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("probes did not drain")


def test_final_answer_extraction():
    assert final_answer("The final answer is \\boxed{\\dfrac{1}{2}}.") == "\\frac{1}{2}"
    assert final_answer("so it is 1,234.") == "1234"
    assert final_answer("no number here") is None


def test_ranked_takes_the_same_total_from_the_cheapest_requests():
    ctl = Controller(PressureMonitor(), policy="brownout", mode="ranked", interval=1)
    ctl.monitor.set(5, 10)  # p = 0.5
    done = ctl.start(Active("done", 1000, 1000, "normal", 1000))
    done.set_probs(onehot(2), "test")  # needs 128-256
    busy = ctl.start(Active("busy", 1000, 1000, "normal", 1000))
    busy.set_probs(onehot(6), "test")  # needs 2K-4K, more than it asked for
    ctl.tick()
    assert done.limit == 256 and busy.limit == 744
    assert (1000 - done.limit) + (1000 - busy.limit) == 1000 == ctl.ranked["shed_tokens"]
    assert ctl.cuts == 2 and ctl.ranked["cuts_with_probs"] == 2 and ctl.ranked["cuts_flat"] == 0
    ctl.tick()  # steady pressure: nothing more to shed
    assert done.limit == 256 and busy.limit == 744 and ctl.ranked["shed_tokens"] == 1000


def test_proportional_takes_the_same_from_everyone():
    ctl = Controller(PressureMonitor(), policy="brownout", mode="proportional", interval=1)
    ctl.monitor.set(5, 10)
    a = ctl.start(Active("a", 1000, 1000, "normal", 1000))
    a.set_probs(onehot(2), "test")
    b = ctl.start(Active("b", 1000, 1000, "normal", 1000))
    b.set_probs(onehot(6), "test")
    assert a.limit == b.limit == 500


def test_ranked_never_cuts_below_what_is_used_and_prefers_known_requests():
    ctl = Controller(PressureMonitor(), policy="brownout", mode="ranked", min_left=64, interval=1)
    ctl.monitor.set(9, 10)  # p = 0.9, shed 0.9 x 2000 = 1800 if it could
    far = ctl.start(Active("far", 1000, 1000, "normal", 1000))
    far.used = 900
    far.set_probs(onehot(1), "test")
    flat = ctl.start(Active("flat", 1000, 1000, "normal", 1000))
    ctl.tick()
    assert far.limit == 964 and flat.limit == 64
    assert ctl.ranked["cuts_flat"] == 1 and ctl.ranked["cuts_with_probs"] == 1


def test_conditional_folds_mass_past_used_and_past_requested():
    ctl = Controller(PressureMonitor(), buckets=Buckets())
    a = Active("a", 1000, 1000, "normal", 1000)
    a.used = 300
    a.set_probs(onehot(6), "test")
    c = ctl.conditional(a)
    assert c[0] == c[1] == c[2] == 0.0 and c[4] == pytest.approx(1 - 2 * (0.1 / 9) / (0.9 + 7 * 0.1 / 9), abs=0.05)
    assert sum(c) == pytest.approx(1.0) and all(v == 0.0 for v in c[5:])


def test_needed_labels_come_from_probing_the_thinking(run):
    gw, fake, c = make(policy="none", labels="needed", needed={"easy": 20, "hard": 1500})
    async def go():
        await c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="a"))
        await c.post("/v1/completions", json=body("hard", max_tokens=3000), headers=think_headers(wf="b"))
        await drain(gw)
    run(go())
    st = gw.stats()
    v = st["classifier"]["versions"]["fake"]
    assert v["labels"] == 2 and v["censored"] == 0 and st["classifier"]["pending"] == 0
    # easy: 50 tokens, the answer already holds at the 32 edge -> needed 31 -> bucket 0-32
    # hard: 2000 tokens, the answer differs at 1024 -> keeps its length 2000 -> bucket 1K-2K
    keys = {r["body"]["prompt"]: None for r in fake.requests if r["body"].get("stream")}
    assert sorted(gw.oracle.values()) == [31, 2000]
    p = st["probe"]
    assert p["labeled"] == 2 and p["probes"] == 4 and p["needed_below_used"] == 1 and p["errors"] == 0
    assert p["by_edge"]["32"] == {"probed": 1, "agreed": 1} and p["by_edge"]["1024"] == {"probed": 1, "agreed": 0}
    assert sorted(fake.probes) == [32, 50, 1024, 2000]


def test_budget_hit_without_a_matching_edge_stays_censored(run):
    gw, fake, c = make(policy="none", labels="needed", needed={"hard": 1500})
    async def go():
        await c.post("/v1/completions", json=body("hard", max_tokens=1200), headers=think_headers())
        await drain(gw)
    run(go())
    v = gw.stats()["classifier"]["versions"]["fake"]
    assert v["labels"] == 0 and v["censored"] == 1 and not gw.oracle


def test_budget_hit_with_a_matching_edge_is_an_exact_label(run):
    gw, fake, c = make(policy="none", labels="needed", needed={"hard": 400})
    async def go():
        await c.post("/v1/completions", json=body("hard", max_tokens=1200), headers=think_headers())
        await drain(gw)
    run(go())
    v = gw.stats()["classifier"]["versions"]["fake"]
    assert v["labels"] == 1 and v["censored"] == 0 and list(gw.oracle.values()) == [511]


def test_cut_requests_are_not_probed(run):
    gw, fake, c = make(policy="uniform", labels="needed", needed={"hard": 10})
    gw.monitor.set(9, 10)
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        await drain(gw)
        return r
    out = parse(run(go()))
    # capped by the controller at the start (limit 100 of 1000): not the request's own budget, so no probe
    assert out["finish"] == "length" and out["info"]["budget"] <= 100 and gw.prober.stats()["submitted"] == 0
    assert gw.stats()["classifier"]["versions"]["fake"]["censored"] == 1


def test_oracle_ranked_cuts_the_request_whose_need_is_known_to_be_met(run):
    gw, fake, c = make(policy="none", labels="needed", needed={"easy": 20, "hard": 1500},
                       lengths={"easy": 400, "hard": 2000})
    async def learn():
        await c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="a"))
        await c.post("/v1/completions", json=body("hard", max_tokens=3000), headers=think_headers(wf="b"))
        await drain(gw)
    run(learn())
    assert gw.oracle and min(gw.oracle.values()) == 31
    r = run(c.post("/admin/policy", json={"policy": "brownout", "predictor": "oracle", "cut": "ranked", "reset": True}))
    assert r.json()["mode"] == "ranked"
    gw.monitor.set(5, 10)
    async def go():
        t = asyncio.create_task(gw.controller.run())
        easy = c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="c"))
        hard = c.post("/v1/completions", json=body("hard", max_tokens=3000), headers=think_headers(wf="d"))
        out = await asyncio.gather(easy, hard)
        t.cancel()
        await drain(gw)
        return out
    easy, hard = (parse(x) for x in run(go()))
    # 0.5 x (1000 + 3000) = 2000 tokens to shed: easy (needs 32) is cut to its floor first, hard pays the rest
    assert easy["info"]["cut"] and easy["usage"]["completion_tokens"] <= 100
    assert hard["info"]["cut"] and 1000 <= hard["usage"]["completion_tokens"] < 2000
    st = gw.stats()["controller"]
    assert st["mode"] == "ranked" and st["ranked"]["cuts_with_probs"] == 2 and gw.counts["oracle_hits"] == 2


def test_online_and_jev_probs_reach_the_controller(run):
    gw, fake, c = make(policy="brownout", predictor="online", mode="ranked")
    seen = {}
    orig = gw.controller.start
    def start(a):
        seen[a.rid] = (a.probs_source, a.probs)
        return orig(a)
    gw.controller.start = start
    run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    (src, probs), = seen.values()
    assert src == "online" and len(probs) == 10 and sum(probs) == pytest.approx(1.0)


def test_admin_rejects_a_bad_cut_mode(run):
    gw, fake, c = make()
    assert run(c.post("/admin/policy", json={"policy": "brownout", "cut": "nope"})).status_code == 400
    assert run(c.post("/admin/policy", json={"policy": "brownout", "cut": "ranked"})).json()["mode"] == "ranked"
    assert gw.stats()["gateway"]["labels"] == "length" and gw.stats()["probe"]["enabled"] is False
