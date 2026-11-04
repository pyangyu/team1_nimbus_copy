import asyncio
import json

import httpx
import pytest

from brownout_gateway.pressure import parse_generated, parse_metrics, pressure, slowdown
from brownout_gateway.jevq import JevScheduler
from brownout_gateway.priority import heuristic
from brownout_gateway import create_app
from brownout_gateway.controller import Active, Controller
from brownout_gateway.pressure import PressureMonitor
from brownout_gateway.routing import Replica, Router
from budget_classifier import BudgetClassifier, Buckets, JevPrior
from conftest import KEY, body, make, parse, think_headers


def test_parse_metrics():
    text = ('# HELP x\nvllm:num_requests_waiting{model_name="q"} 6.0\n'
            'vllm:num_requests_running{model_name="q"} 3.0\nvllm:num_requests_swapped 1\n')
    assert parse_metrics(text) == (6.0, 3.0)
    assert pressure(6, 3) == 1.0 and pressure(1, 4) == 0.25 and pressure(0, 0) == 0.0


def test_priority_heuristic():
    assert heuristic("interactive", "solve.1", None) == "high"
    assert heuristic("agentic", "solve.1", 1_000_000, now=995) == "high"
    assert heuristic("agentic", "solve.3", None) == "low"
    assert heuristic("agentic", "solve.0", None) == "normal"
    assert heuristic("agentic", "plan", None) == "normal"


def test_answer_phase_passes_through(run):
    gw, fake, c = make()
    async def go():
        r = await c.post("/v1/completions", json={**body("easy"), "max_tokens": 20},
                         headers={"X-Lifeline-Phase": "answer", "X-Lifeline-Thinking-Budget": "1000"})
        return r
    r = run(go())
    assert r.status_code == 200 and parse(r)["text"] == "The answer is \\boxed{4}."
    assert fake.requests[0]["body"]["max_tokens"] == 20 and gw.counts["passthrough"] == 1


def test_think_learns_and_tags(run, tmp_path):
    gw, fake, c = make(log_path=tmp_path / "gw.jsonl")
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    out = parse(r)
    assert r.headers["x-lifeline-tier"] in gw.clf.buckets.labels() and r.headers["x-lifeline-budget"] == "1000"
    assert out["finish"] == "stop" and out["usage"]["completion_tokens"] == 50 and out["info"]["cut"] is False
    v = gw.clf.stats()["versions"]["fake"]
    assert v["labels"] == 1 and v["censored"] == 0
    row = json.loads((tmp_path / "gw.jsonl").read_text())
    assert row["used"] == 50 and row["natural"] and "easy" in row["prompt"]


def test_budget_hit_is_censored(run):
    gw, fake, c = make()
    out = parse(run(c.post("/v1/completions", json=body("hard", 300), headers=think_headers(budget=300))))
    assert out["finish"] == "length" and out["usage"]["completion_tokens"] == 300
    assert gw.clf.stats()["versions"]["fake"]["censored"] == 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_reject_and_none(run):
    gw, fake, c = make(policy="reject", admit_burst=0)
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert r.status_code == 429 and gw.counts["rejected"] == 1 and not fake.requests
    gw.controller.set_policy("none")
    assert run(c.post("/v1/completions", json=body("easy"), headers=think_headers())).status_code == 200


def test_reject_is_a_token_bucket_on_new_workflows(run):
    clock = Clock()
    gw, fake, c = make(policy="reject", admit_rate=0.5, admit_burst=2, clock=clock)
    codes = [run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf=f"w{i}"))).status_code
             for i in range(3)]
    clock.t = 2.0
    codes.append(run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="w3"))).status_code)
    codes.append(run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="w4"))).status_code)
    assert codes == [200, 200, 429, 200, 429] and gw.counts["rejected"] == 2


def test_answer_after_think_is_never_rejected(run):
    gw, fake, c = make(policy="reject", admit_burst=0)
    r = run(c.post("/v1/completions", json=body("easy"),
                   headers={"X-Lifeline-Phase": "answer", "X-Lifeline-Thinking-Budget": "1000"}))
    assert r.status_code == 200


def test_defer_waits_for_queue(run):
    gw, fake, c = make(policy="defer", reject_waiting=8, defer_poll=0.01)
    gw.monitor.set(20, 4)
    async def go():
        async def drain():
            await asyncio.sleep(0.05)
            gw.monitor.set(0, 4)
        t = asyncio.create_task(drain())
        r = await c.post("/v1/completions", json=body("easy"), headers=think_headers())
        await t
        return r
    r = run(go())
    assert r.status_code == 200 and gw.counts["deferred"] == 1 and gw.counts["deferred_s"] >= 0.04


def test_defer_times_out(run):
    gw, fake, c = make(policy="defer", reject_waiting=8, defer_timeout=0.03, defer_poll=0.01)
    gw.monitor.set(20, 4)
    assert run(c.post("/v1/completions", json=body("easy"), headers=think_headers())).status_code == 429


def cut_midstream(gw, fake, at=50, waiting=4, running=4):
    def on_token(i):
        if i == at:
            gw.monitor.set(waiting, running)
    fake.on_token = on_token


def test_uniform_cuts_in_flight(run):
    gw, fake, c = make(policy="uniform")
    cut_midstream(gw, fake)
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        return r
    out = parse(run(go()))
    assert out["finish"] == "length" and out["info"]["cut"] and out["info"]["cut_from"] == 1000
    assert 50 <= out["usage"]["completion_tokens"] <= 200
    assert fake.aborted == 1 and gw.counts["cut"] == 1
    assert gw.clf.stats()["versions"]["fake"]["censored"] == 1


def test_announced_cut_continues_with_notice(run):
    gw, fake, c = make(policy="uniform", notice="announced")
    cut_midstream(gw, fake)
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        return r
    out = parse(run(go()))
    assert out["info"]["announced"] and "<budget_cut left=" in out["text"]
    cont = fake.requests[-1]["body"]
    assert len(fake.requests) == 2 and "<budget_cut left=" in cont["prompt"] and cont["max_tokens"] < 200
    assert out["usage"]["completion_tokens"] <= 200 and gw.counts["announced"] == 1


def drop_limits(gw, fake, at: dict[int, int]):
    seen = [0]
    def on_token(i):
        if seen[0] in at:
            for a in gw.controller.active.values():
                a.limit = at[seen[0]]
        seen[0] += 1
    fake.on_token = on_token


def run_announced(run, tmp_path, at):
    gw, fake, c = make(policy="none", notice="announced", log_path=tmp_path / "gw.jsonl")
    drop_limits(gw, fake, at)
    out = parse(run(c.post("/v1/completions", json=body("hard"), headers=think_headers())))
    row = json.loads((tmp_path / "gw.jsonl").read_text().splitlines()[-1])
    return gw, fake, out, row


def test_announced_renotices_on_each_large_drop(run, tmp_path):
    gw, fake, out, row = run_announced(run, tmp_path, {100: 600, 200: 400})
    assert [r["body"]["max_tokens"] for r in fake.requests] == [1000, 499, 199]
    assert out["text"].count("<budget_cut left=") == 2
    assert "<budget_cut left=499>" in out["text"] and "<budget_cut left=199>" in out["text"]
    assert fake.requests[-1]["body"]["prompt"].endswith("<budget_cut left=199>\n")
    assert out["finish"] == "length" and out["usage"]["completion_tokens"] == 400
    assert out["info"]["announced"] and out["info"]["notices"] == 2 and not row["natural"]
    assert row["announced"] and row["notices"] == 2 and not row["cut"]
    assert gw.counts["announced"] == 1 and gw.counts["notices"] == 2 and gw.counts["cut"] == 0


def test_announced_small_drop_hard_stops_without_notice(run, tmp_path):
    gw, fake, out, row = run_announced(run, tmp_path, {100: 600, 200: 560})
    assert len(fake.requests) == 2 and out["text"].count("<budget_cut left=") == 1
    assert out["finish"] == "length" and out["usage"]["completion_tokens"] == 560
    assert row["notices"] == 1 and row["cut"] and gw.counts["notices"] == 1 and gw.counts["cut"] == 1


def test_announced_drop_past_used_hard_stops(run, tmp_path):
    gw, fake, out, row = run_announced(run, tmp_path, {100: 600, 300: 250})
    assert len(fake.requests) == 2 and out["text"].count("<budget_cut left=") == 1
    assert out["finish"] == "length" and out["usage"]["completion_tokens"] == 301
    assert row["notices"] == 1 and row["cut"] and gw.counts["cut"] == 1


def test_renotice_threshold():
    ctl = Controller(PressureMonitor(), renotice_min=64, renotice_frac=0.1)
    assert ctl.renotice(100) == 64 and ctl.renotice(2000) == 200


def test_no_cut_without_pressure(run):
    gw, fake, c = make(policy="brownout")
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        return r
    out = parse(run(go()))
    assert out["info"]["cut"] is False and out["usage"]["completion_tokens"] == 1000


def test_low_priority_is_cut_harder(run):
    gw, fake, c = make(policy="brownout")
    gw.monitor.set(3, 5)
    async def go():
        hi = c.post("/v1/completions", json=body("hard"), headers=think_headers(cls="interactive", wf="a"))
        lo = c.post("/v1/completions", json=body("hard"), headers=think_headers(step="solve.3", wf="b"))
        return await asyncio.gather(hi, lo)
    hi, lo = run(go())
    assert hi.headers["x-lifeline-priority"] == "high" and lo.headers["x-lifeline-priority"] == "low"
    assert parse(lo)["info"]["budget"] < parse(hi)["info"]["budget"]


def test_uniform_ignores_priority(run):
    gw, fake, c = make(policy="uniform")
    gw.monitor.set(3, 5)
    async def go():
        hi = c.post("/v1/completions", json=body("hard"), headers=think_headers(cls="interactive", wf="a"))
        lo = c.post("/v1/completions", json=body("hard"), headers=think_headers(step="solve.3", wf="b"))
        return await asyncio.gather(hi, lo)
    hi, lo = run(go())
    assert parse(lo)["info"]["budget"] == parse(hi)["info"]["budget"]


def fake_jev(seen, tier=3, prio=0):
    def handler(req):
        b = json.loads(req.content)
        seen.append(b)
        ans = {}
        for q, d in b["questions"].items():
            n = len(d["criteria"])
            pick = tier if q == "tier" else prio
            ans[q] = {"type": "score", "probabilities": {str(i): (0.9 if i == pick else 0.1 / (n - 1)) for i in range(n)}}
        return httpx.Response(200, json={"answers": ans})
    return httpx.MockTransport(handler)


def fake_jev_slow(seen, delay, **kw):
    inner = fake_jev(seen, **kw)

    async def handler(req):
        await asyncio.sleep(delay)
        return inner.handler(req)
    return httpx.MockTransport(handler)


def test_jev_answers_tier_and_priority_in_one_call(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, tier=3, prio=0))
    gw, fake, c = make(jev=jev)
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert len(seen) == 1 and set(seen[0]["questions"]) == {"tier", "priority"}
    info = parse(r)["info"]
    assert r.headers["x-lifeline-tier"] == "pending" and info["predictor"] == "jev"
    assert info["tier"] == info["jev_tier"] == "256-512" and info["online_tier"] in gw.clf.buckets.labels()
    assert info["priority"] == "low" and info["priority_source"] == "jev" and info["jev_at"] is not None


def test_online_predictor_sets_budgets_and_jev_only_shadows(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, tier=3, prio=0))
    gw, fake, c = make(jev=jev, predictor="online")
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    info = parse(r)["info"]
    assert len(seen) == 1 and info["predictor"] == "online" and info["jev_tier"] == "256-512"
    assert info["tier"] == info["online_tier"] == r.headers["x-lifeline-tier"] != "256-512"
    assert info["priority"] == "normal" and info["priority_source"] == "heuristic"
    paired = gw.clf.stats()["versions"]["fake"]["paired"]
    assert paired["online"]["n"] == paired["jev"]["n"] == 1


def test_jev_runs_beside_decode_and_cuts_mid_flight(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev_slow(seen, 0.03, tier=2, prio=1))
    gw, fake, c = make(jev=jev)
    gw.monitor.set(2, 4)
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        return r
    r = run(go())
    out = parse(r)
    assert r.headers["x-lifeline-tier"] == "pending" and out["info"]["budget"] == 500
    assert out["info"]["tier"] == "128-256" and out["info"]["jev_at"] > 0 and out["info"]["cut"]
    assert out["usage"]["completion_tokens"] < 300 < 500
    st = gw.jevq.stats()
    assert st["during_decode"] == 1 and st["tier_lowered"] == 1 and st["before_first_token"] == 0


def test_jev_serial_sets_the_budget_before_vllm_sees_the_request(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev_slow(seen, 0.03, tier=2, prio=1))
    gw, fake, c = make(jev=jev, predictor="jev-serial")
    gw.monitor.set(2, 4)
    r = run(c.post("/v1/completions", json=body("hard"), headers=think_headers()))
    out = parse(r)
    assert r.headers["x-lifeline-tier"] == "128-256" and out["info"]["jev_at"] == 0
    assert fake.requests[0]["body"]["max_tokens"] == 128 and out["usage"]["completion_tokens"] == 128
    assert gw.counts["serial_waits"] == 1 and gw.counts["serial_wait_s"] >= 0.03
    assert gw.jevq.stats()["before_first_token"] == 1


def test_jev_serial_gives_up_after_the_timeout(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev_slow(seen, 0.3, tier=2), timeout=1.0)
    gw, fake, c = make(jev=jev, predictor="jev-serial", serial_timeout=0.02)
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert r.status_code == 200 and r.headers["x-lifeline-tier"] == "pending"
    assert gw.counts["serial_waits"] == 1 and gw.counts["serial_wait_s"] < 0.2


def test_oracle_uses_the_length_from_an_uncut_run(run):
    gw, fake, c = make(policy="none")
    run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    run(c.post("/admin/policy", json={"policy": "brownout", "predictor": "oracle", "reset": True}))
    gw.monitor.set(2, 4)
    hit = run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="w2")))
    miss = run(c.post("/v1/completions", json=body("hard"), headers=think_headers(wf="w3")))
    assert hit.headers["x-lifeline-tier"] == "32-128" and parse(hit)["info"]["budget"] == 64
    assert miss.headers["x-lifeline-tier"] == "unknown" and parse(miss)["info"]["budget"] == 500
    assert gw.counts["oracle_hits"] == 1 and gw.counts["oracle_misses"] == 1 and gw.stats()["gateway"]["oracle_prompts"] == 1


def test_reset_forgets_learned_state(run):
    gw, fake, c = make()
    run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert gw.clf.stats()["versions"]["fake"]["labels"] == 1 and gw.clf.memory.items
    run(c.post("/admin/policy", json={"policy": "brownout", "reset": True}))
    assert gw.clf.stats()["versions"] == {} and not gw.clf.memory.items and not gw.clf.online.W.any()


def test_warm_reset_restores_the_snapshot(run):
    gw, fake, c = make()
    run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    run(c.post("/admin/policy", json={"snapshot": True}))
    run(c.post("/v1/completions", json=body("hard", 300), headers=think_headers(wf="w2", budget=300)))
    assert gw.clf.stats()["versions"]["fake"]["censored"] == 1
    run(c.post("/admin/policy", json={"policy": "brownout", "reset": "warm"}))
    v = gw.clf.stats()["versions"]["fake"]
    assert v["labels"] == 1 and v["censored"] == 0 and gw.clf.online.W.any() and gw.clf.memory.items
    run(c.post("/admin/policy", json={"policy": "brownout", "reset": True}))
    assert gw.clf.stats()["versions"] == {} and not gw.clf.online.W.any()


def test_jev_queue_serves_waiting_requests_first(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev_slow(seen, 0.02))
    clf = BudgetClassifier(dim=1 << 12, jev=jev)
    ctl = Controller(PressureMonitor())
    q = JevScheduler(clf, ctl)
    acts = {}
    for name in ("first", "decoding", "waiting", "finished"):
        clf.predict_online(name, f"prompt {name}", "m")
        acts[name] = ctl.start(Active(name, 1000, 1000, "normal", 1000))
    async def go():
        q.submit(acts["first"], "agentic", "solve.0", None)
        for name in ("decoding", "waiting", "finished"):
            q.submit(acts[name], "agentic", "solve.0", None)
        acts["decoding"].used = 40
        ctl.finish(acts["finished"])
        await asyncio.sleep(0.15)
    run(go())
    order = [b["state"]["request"]["prompt_head"].split()[-1] for b in seen]
    assert order == ["first", "waiting", "decoding"]
    st = q.stats()
    assert st["dropped_finished"] == 1 and st["before_first_token"] == 2 and st["during_decode"] == 1


def test_admin_auth_and_stats(run):
    gw, fake, c = make()
    async def go():
        bad = await c.post("/v1/completions", json=body("easy"), headers={"Authorization": "Bearer nope"})
        sw = await c.post("/admin/policy", json={"policy": "uniform", "notice": "announced", "reject_waiting": 3,
                                                 "admit_rate": 1.5, "admit_burst": 4, "predictor": "online"})
        badp = await c.post("/admin/policy", json={"policy": "uniform", "predictor": "nope"})
        inv = await c.post("/admin/policy", json={"policy": "nope"})
        models = await c.get("/v1/models")
        st = await c.get("/v1/stats")
        return bad, sw, inv, models, st, badp
    bad, sw, inv, models, st, badp = run(go())
    assert bad.status_code == 401 and inv.status_code == 400 and badp.status_code == 400
    assert sw.json()["predictor"] == "online"
    assert sw.json()["policy"] == "uniform" and sw.json()["notice"] == "announced" and gw.reject_waiting == 3
    assert gw.admit_rate == 1.5 and gw.admit_burst == 4 and gw.tokens == 4
    assert models.json()["data"][0]["id"] == "fake"
    assert {"gateway", "controller", "pressure", "jev", "classifier"} <= st.json().keys()


def test_pressure_series_since(run):
    gw, fake, c = make()
    now = iter([100.0, 101.0, 102.0])
    gw.monitor.clock = lambda: next(now)
    gw.monitor.set(0, 4)
    gw.monitor.set(2, 4)
    gw.monitor.set(9, 3)
    async def go():
        return await c.get("/v1/pressure", params={"since": 101}), await c.get("/v1/pressure", params={"since": "x"})
    r, bad = run(go())
    assert r.json()["samples"] == [[101.0, 0.5, 2.0, 4.0, None], [102.0, 1.0, 9.0, 3.0, None]]
    assert r.json()["columns"] == ["t", "p", "waiting", "running", "tpot_ms"] and bad.status_code == 400
    assert gw.monitor.stats()["p_hist_deciles"][0] == 1 and gw.monitor.stats()["p_hist_deciles"][9] == 1


def test_decode_time_is_pressure_past_its_slo():
    text = 'vllm:generation_tokens_total{engine="0"} 100.0\nvllm:generation_tokens_created 1.7e9\n'
    assert parse_generated(text) == 100.0 and parse_generated("vllm:num_requests_running 3\n") is None
    assert slowdown(0.05, 0.04) == pytest.approx(0.25) and slowdown(0.03, 0.04) == 0 and slowdown(0.2, 0.04) == 1
    assert slowdown(0.2, None) == 0 and slowdown(None, 0.04) == 0
    m = PressureMonitor(tpot_slo=0.04, rate_window=2.0)
    assert m.decode_time(0.0, 1000, 40) is None
    tpot = m.decode_time(1.0, 1800, 40)
    assert tpot == pytest.approx(0.05)
    m.set(0, 40, tpot)
    assert m.p == pytest.approx(0.25) and m.series[-1][4] == 50.0 and m.stats()["tpot_ms"] == 50.0
    m.set(30, 40, tpot)
    assert m.p == 0.75
    m.decode_time(2.0, 2600, 20)
    assert m.decode_time(3.0, 3400, 20) == pytest.approx((40 + 20 + 20) / 3 / 800) and len(m.generated) == 3
    assert m.decode_time(4.0, 100, 40) is None
    plain = PressureMonitor()
    plain.set(0, 40, 0.2)
    assert plain.p == 0 and plain.stats()["tpot_slo_ms"] is None


def test_poll_reads_the_generation_counter(run):
    texts = iter([f"vllm:num_requests_running 40\nvllm:num_requests_waiting 0\nvllm:generation_tokens_total {g}\n"
                  for g in (1000, 1800)])
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=next(texts))))
    now = iter([0.0, 0.0, 1.0, 1.0])
    m = PressureMonitor("http://vllm/metrics", http=http, clock=lambda: next(now), tpot_slo=0.04)
    run(m.poll_once())
    run(m.poll_once())
    assert m.errors == 0 and m.tpot == pytest.approx(0.05) and m.p == pytest.approx(0.25)
    assert [x[4] for x in m.since(0)] == [None, 50.0]


TWO = "http://vllm-a/v1,http://vllm-b/v1"


def test_router_picks_least_loaded_and_breaks_ties_round_robin():
    a, b, c = (Replica(u) for u in ("http://a/v1", "http://b/v1", "http://c/v1"))
    r = Router([a, b, c])
    assert [r.pick() for _ in range(3)] == [0, 1, 2]
    for rep, (w, n) in zip((a, b, c), ((0, 5), (2, 1), (0, 9))):
        rep.monitor.set(w, n)
    assert r.pick() == 1 and r.pick() == 1
    assert r.pick() == 0 and r.pick() == 1
    b.monitor.set(9, 9)
    assert r.pick() == 0 and a.routed == 3 and b.routed == 4 and c.routed == 1


def test_router_keeps_continuations_on_their_replica():
    r = Router([Replica("http://a/v1"), Replica("http://b/v1")])
    assert r.route("w1:solve.0", True) == 0 and r.route("w2:solve.0", True) == 1
    assert r.route("w1:solve.0", False) == 0 and r.route("w2:solve.0", False) == 1
    assert r.route("w1:solve.1", True) == 0 and r.route(None, False) == 1
    small = Router([Replica("http://a/v1")], max_sticky=2)
    for k in ("x", "y", "z"):
        small.route(k, True)
    assert list(small.sticky) == ["y", "z"]


def test_two_replicas_spread_new_calls_and_keep_think_answer_together(run):
    gw, fake, c = make(upstream=TWO)
    async def go():
        for wf in ("w1", "w2"):
            await c.post("/v1/completions", json=body("easy"), headers=think_headers(wf=wf))
            await c.post("/v1/completions", json={**body("easy"), "max_tokens": 20},
                         headers=think_headers(wf=wf) | {"X-Lifeline-Phase": "answer"})
        return await c.get("/v1/stats")
    st = run(go()).json()
    hosts = [(r["headers"]["x-lifeline-workflow"], r["headers"]["x-lifeline-phase"], r["host"]) for r in fake.requests]
    assert hosts == [("w1", "think", "vllm-a"), ("w1", "answer", "vllm-a"),
                     ("w2", "think", "vllm-b"), ("w2", "answer", "vllm-b")]
    assert [(x["url"], x["requests"], x["think"]) for x in st["replicas"]] == [
        ("http://vllm-a/v1", 2, 1), ("http://vllm-b/v1", 2, 1)]


def test_announced_continuation_stays_on_the_replica_and_cuts_by_its_pressure(run, tmp_path):
    gw, fake, c = make(upstream=TWO, policy="uniform", notice="announced", log_path=tmp_path / "gw.jsonl")
    busy, idle = gw.router.replicas[1].monitor, gw.router.replicas[0].monitor
    idle.set(0, 1)
    busy.set(0, 3)
    def on_token(i):
        if i == 50:
            busy.set(4, 4)
    fake.on_token = on_token
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers(wf="w1"))
        t.cancel()
        return r
    idle.set(0, 9)
    out = parse(run(go()))
    assert out["info"]["announced"] and out["info"]["replica"] == 1
    assert len(fake.requests) == 2 and {r["host"] for r in fake.requests} == {"vllm-b"}
    row = json.loads((tmp_path / "gw.jsonl").read_text().splitlines()[-1])
    assert row["replica"] == 1 and row["p_end"] == 1.0


def test_pressure_on_another_replica_does_not_cut(run):
    gw, fake, c = make(upstream=TWO, policy="uniform")
    gw.router.replicas[0].monitor.set(0, 1)
    gw.router.replicas[1].monitor.set(0, 5)
    def on_token(i):
        if i == 50:
            gw.router.replicas[1].monitor.set(8, 4)
    fake.on_token = on_token
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await c.post("/v1/completions", json=body("hard"), headers=think_headers())
        t.cancel()
        return r
    out = parse(run(go()))
    assert out["info"]["replica"] == 0 and not out["info"]["cut"] and out["usage"]["completion_tokens"] == 1000
    assert gw.controller.cutting and not gw.controller.cutting_at(0)


def test_pressure_endpoint_and_stats_per_replica(run):
    gw, fake, c = make(upstream=TWO)
    gw.router.replicas[0].monitor.set(1, 4)
    gw.router.replicas[1].monitor.set(8, 4)
    async def go():
        return await c.get("/v1/pressure"), await c.get("/v1/stats")
    pr, st = run(go())
    reps = pr.json()["replicas"]
    assert [x["url"] for x in reps] == ["http://vllm-a/v1", "http://vllm-b/v1"]
    assert [x["samples"][0][1] for x in reps] == [0.25, 1.0] and pr.json()["samples"] == reps[0]["samples"]
    p = st.json()["pressure"]
    assert p["p"] == 1.0 and p["waiting"] == 9 and p["samples"] == 2 and sum(p["p_hist_deciles"]) == 2


def test_upstream_error_is_returned(run):
    gw, fake, c = make()
    gw.upstream_key = "wrong"
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert r.status_code == 401 and gw.counts["upstream_errors"] == 1 and not gw.controller.active


def test_math_agent_client_through_gateway(run):
    llm = pytest.importorskip("math_agent.llm")
    gw, fake, _ = make(policy="uniform")
    cut_midstream(gw, fake)
    client = llm.LLMClient("http://gw/v1", api_key=KEY,
                           transport=httpx.ASGITransport(app=create_app(gw, background=False)))
    async def go():
        t = asyncio.create_task(gw.controller.run())
        r = await client.generate([{"role": "user", "content": "a hard question"}], llm.CallMeta("w1", "solve.0"),
                                  think_budget=1000, max_answer_tokens=50)
        t.cancel()
        await client.aclose()
        return r
    r = run(go())
    think, answer = r.phases
    assert think.tier in gw.clf.buckets.labels() and think.priority == "normal"
    assert think.gateway["cut"] and think.finish_reason == "length" and r.budget_exhausted
    assert llm.EARLY_STOP in fake.requests[-1]["body"]["prompt"] and r.text.endswith("\\boxed{4}.")


def test_math_agent_sees_rejection(run):
    llm = pytest.importorskip("math_agent.llm")
    gw, fake, _ = make(policy="reject", admit_burst=0)
    client = llm.LLMClient("http://gw/v1", api_key=KEY,
                           transport=httpx.ASGITransport(app=create_app(gw, background=False)))
    async def go():
        try:
            await client.generate([{"role": "user", "content": "easy"}], llm.CallMeta("w1", "solve.0"),
                                  think_budget=100, max_answer_tokens=10)
        finally:
            await client.aclose()
    with pytest.raises(llm.Rejected):
        run(go())


def test_admitted_workflow_is_never_rejected_midway(run):
    gw, fake, c = make(policy="reject", admit_rate=0, admit_burst=1)
    first = run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="early")))
    later = run(c.post("/v1/completions", json=body("easy"), headers=think_headers(step="verify.0", wf="early")))
    new = run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="late")))
    assert first.status_code == 200 and later.status_code == 200 and new.status_code == 429


def test_policy_switch_forgets_admitted_workflows(run):
    gw, fake, c = make(policy="none")
    assert run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="w0"))).status_code == 200
    run(c.post("/admin/policy", json={"policy": "reject", "admit_rate": 0, "admit_burst": 0}))
    assert run(c.post("/v1/completions", json=body("easy"), headers=think_headers(wf="w0"))).status_code == 429
