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


def test_jev_drives_cold_start_then_fades():
    seen = []
    clf = BudgetClassifier(dim=1 << 14, warmup=20, jev_until=0.9,
                           jev=JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen)))
    rows = list(workload(400))

    async def go():
        sources = []
        for r in rows:
            sources.append((await clf.predict(r["id"], r["prompt"], r["model"])).source)
            clf.observe(r["id"], r["thinking_tokens"], model=r["model"])
        return sources
    sources = asyncio.run(go())
    assert sources[0] == "jev" and sources[1] == "blend" and sources[-1] == "online"
    assert len(seen) < len(rows)
    examples = seen[-1]["state"]["recent_requests"]
    assert examples and {"prompt", "thinking_tokens_used", "bucket"} <= examples[0].keys()


def test_jev_failure_falls_back():
    seen = []
    clf = BudgetClassifier(dim=1 << 12,
                           jev=JevPrior(Buckets(), "http://jev/v1/systemone", transport=fake_jev(seen, fail=True)))
    pred = asyncio.run(clf.predict("a", "What is 2 + 2?", "m1"))
    assert pred.source == "cold" and clf.jev.failures == 1 and "HTTPStatusError" in clf.stats()["jev"]["last_error"]
