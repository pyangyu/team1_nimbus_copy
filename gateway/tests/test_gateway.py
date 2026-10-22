import asyncio
import json

import httpx
import pytest

from brownout_gateway.pressure import parse_metrics, pressure
from brownout_gateway.priority import PriorityRefiner, heuristic
from brownout_gateway import create_app
from brownout_gateway.controller import Active
from budget_classifier import Buckets, JevPrior
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


def test_reject_and_none(run):
    gw, fake, c = make(policy="reject", reject_waiting=8)
    gw.monitor.set(20, 4)
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert r.status_code == 429 and gw.counts["rejected"] == 1 and not fake.requests
    gw.controller.set_policy("none")
    assert run(c.post("/v1/completions", json=body("easy"), headers=think_headers())).status_code == 200


def test_answer_after_think_is_never_rejected(run):
    gw, fake, c = make(policy="reject", reject_waiting=8)
    gw.monitor.set(20, 4)
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


def test_jev_answers_tier_and_priority_in_one_call(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, tier=3, prio=0))
    gw, fake, c = make(jev=jev)
    r = run(c.post("/v1/completions", json=body("easy"), headers=think_headers()))
    assert len(seen) == 1 and set(seen[0]["questions"]) == {"tier", "priority"}
    assert r.headers["x-lifeline-tier"] == "256-512" and r.headers["x-lifeline-priority"] == "low"
    assert parse(r)["info"]["priority_source"] == "jev"


def test_refiner_updates_priority_in_background(run):
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, prio=2))
    ref = PriorityRefiner(jev)
    a = Active("r", 1000, 1000, "normal", 1000)
    async def go():
        ref.refine(a, "agentic", "solve.1", None, "prompt")
        ref.refine(Active("s", 1, 1, "normal", 1), "agentic", "solve.1", None, "prompt")
        await asyncio.sleep(0.05)
    run(go())
    assert a.priority == "high" and a.priority_source == "jev" and ref.stats() == {"asked": 1, "answered": 1, "dropped_busy": 1}


def test_admin_auth_and_stats(run):
    gw, fake, c = make()
    async def go():
        bad = await c.post("/v1/completions", json=body("easy"), headers={"Authorization": "Bearer nope"})
        sw = await c.post("/admin/policy", json={"policy": "uniform", "notice": "announced", "reject_waiting": 3})
        inv = await c.post("/admin/policy", json={"policy": "nope"})
        models = await c.get("/v1/models")
        st = await c.get("/v1/stats")
        return bad, sw, inv, models, st
    bad, sw, inv, models, st = run(go())
    assert bad.status_code == 401 and inv.status_code == 400
    assert sw.json()["policy"] == "uniform" and sw.json()["notice"] == "announced" and gw.reject_waiting == 3
    assert models.json()["data"][0]["id"] == "fake"
    assert {"gateway", "controller", "pressure", "priority", "classifier"} <= st.json().keys()


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
    gw, fake, _ = make(policy="reject", reject_waiting=1)
    gw.monitor.set(5, 1)
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
