import asyncio
import json

import httpx

from budget_classifier import BudgetClassifier, Buckets, JevPrior
from conftest import workload


def fake_jev(seen, bucket=7, fail=False):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if fail:
            return httpx.Response(503)
        n = len(body["questions"]["tier"]["criteria"])
        probs = {str(i): (0.91 if i == bucket else 0.01) for i in range(n)}
        return httpx.Response(200, json={"answers": {"tier": {"type": "score", "probabilities": probs,
                                                              "score": float(bucket), "confidence": 0.9}}})
    return httpx.MockTransport(handler)


def test_request_is_a_ten_level_score_question():
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen))
    p = asyncio.run(jev.ask("Prove it.", []))
    q = seen[0]["questions"]["tier"]
    assert q["type"] == "score" and len(q["criteria"]) == 10 and q["criteria"][0].startswith("0-32")
    assert p.argmax() == 7 and abs(p.sum() - 1) < 1e-9


def test_jev_is_asked_every_call_and_scored_beside_the_online_model():
    seen = []
    clf = BudgetClassifier(dim=1 << 14, jev=JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, 1)))
    rows = [r for r in workload(600) if r["prompt"].startswith("What is")][:60]

    async def go():
        sources = []
        for r in rows:
            sources.append((await clf.predict(r["id"], r["prompt"], r["model"])).source)
            clf.observe(r["id"], r["thinking_tokens"], model=r["model"])
        return sources
    sources = asyncio.run(go())
    assert set(sources) == {"jev"} and len(seen) == len(rows)
    paired = clf.stats()["versions"]["m1"]["paired"]
    assert paired["jev"]["n"] == paired["online"]["n"] == len(rows)
    assert paired["jev"]["exact"] == len(rows) > paired["online"]["exact"]
    examples = seen[-1]["state"]["recent_requests"]
    assert examples and {"prompt", "thinking_tokens_used", "bucket"} <= examples[0].keys()


def test_predict_online_never_waits_for_jev():
    seen = []
    clf = BudgetClassifier(dim=1 << 12, jev=JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen)))
    pred = clf.predict_online("a", "What is 2 + 2?", "m1")
    assert pred.source == "cold" and not seen
    refined, _ = asyncio.run(clf.ask_jev("a"))
    assert refined.bucket == 7 and len(seen) == 1
    clf.observe("a", 50, model="m1")
    assert asyncio.run(clf.ask_jev("a")) == (None, {}) and len(seen) == 1


def test_jev_failure_falls_back():
    seen = []
    clf = BudgetClassifier(dim=1 << 12,
                           jev=JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, fail=True)))
    pred = asyncio.run(clf.predict("a", "What is 2 + 2?", "m1"))
    assert pred.source == "cold" and clf.jev.failures == 1 and "HTTPStatusError" in clf.stats()["jev"]["last_error"]


def test_busy_jev_is_skipped_not_queued():
    seen = []
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=slow_jev(seen), max_inflight=1)

    async def go():
        return await asyncio.gather(jev.ask("a", []), jev.ask("b", []))
    a, b = asyncio.run(go())
    assert a is not None and b is None and jev.calls == 1 and jev.skipped_busy == 1 and jev.inflight == 0


def slow_jev(seen, delay=0.05):
    inner = fake_jev(seen)

    async def handler(req):
        await asyncio.sleep(delay)
        return inner.handler(req)
    return httpx.MockTransport(handler)


def test_calls_spread_over_several_servers():
    hosts = []
    inner = slow_jev([])

    async def handler(req):
        hosts.append(req.url.host)
        return await inner.handler(req)
    jev = JevPrior(Buckets(), "http://jev1/v1/systemone,http://jev2/v1/systemone", max_inflight=2,
                   transport=httpx.MockTransport(handler))

    async def go():
        return await asyncio.gather(jev.ask("a", []), jev.ask("b", []))
    a, b = asyncio.run(go())
    assert a is not None and b is not None and sorted(hosts) == ["jev1", "jev2"]
    assert jev.call_s > 0 and set(jev.url_inflight.values()) == {0}


def test_settled_judge_asks_a_score_and_a_noul_question():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        probs = {"0": 0.05, "1": 0.15, "2": 0.3, "3": 0.5}
        return httpx.Response(200, json={"answers": {
            "settled": {"type": "score", "probabilities": probs, "score": 2.25, "confidence": 0.5},
            "done": {"type": "noul", "noul": 0.8}}})
    jev = JevPrior(Buckets(), "http://jev/v1/systemone", transport=httpx.MockTransport(handler))
    out = asyncio.run(jev.ask_settled("Solve x.", "I think x = 4. Let me verify." * 200, 1500))
    q, st = seen[0]["questions"], seen[0]["state"]
    assert q["settled"]["type"] == "score" and len(q["settled"]["criteria"]) == 4 and q["done"]["type"] == "noul"
    assert st["thinking_so_far"]["tokens"] == 1500 and st["thinking_so_far"]["truncated_at_start"]
    assert len(st["thinking_so_far"]["tail"]) == 1500 and st["request"]["prompt_head"] == "Solve x."
    assert abs(out["settled"] - 0.8) < 1e-9 and out["done"] == 0.8 and abs(out["level"] - 2.25) < 1e-9
    assert out["ms"] >= 0 and jev.failures == 0
