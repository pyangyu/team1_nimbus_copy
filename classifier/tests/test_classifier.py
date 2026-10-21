import asyncio

import numpy as np

from budget_classifier import BudgetClassifier
from conftest import workload


def run(clf, rows):
    async def go():
        out = []
        for r in rows:
            pred = await clf.predict(r["id"], r["prompt"], r.get("model"))
            res = clf.observe(r["id"], r["thinking_tokens"], finished=r.get("finished", True), model=r.get("model"))
            out.append((pred, res))
        return out
    return asyncio.run(go())


def acc(clf, model):
    return clf.stats()["versions"][model]


def test_cold_start_without_jev():
    clf = BudgetClassifier(dim=1 << 14)
    pred = run(clf, [next(workload(1))])[0][0]
    assert pred.source == "cold" and 0 <= pred.bucket < 10 and pred.budget == clf.buckets.budget(pred.bucket)
    assert abs(sum(pred.probs) - 1) < 1e-3


def test_learns_online_from_finished_requests():
    clf = BudgetClassifier(dim=1 << 14)
    run(clf, workload(600))
    s = acc(clf, "m1")
    assert s["labels"] == 600 and s["exact"] >= 0.85 and s["within_one"] >= 0.97


def test_model_upgrade_reconverges():
    clf = BudgetClassifier(dim=1 << 14, window=100)
    run(clf, workload(600, seed=1, model="m1"))
    rows = list(workload(400, seed=2, scale=4.0, model="m2"))
    out = run(clf, rows)
    hits = [p.bucket == clf.buckets.index(r["thinking_tokens"]) for (p, _), r in zip(out, rows)]
    assert clf.upgrades == 1 and clf.current_model == "m2"
    assert sum(hits[:10]) <= 5
    assert sum(hits[20:100]) >= 72
    assert acc(clf, "m2")["exact"] >= 0.85


def test_censored_skip_leaves_weights_alone():
    clf = BudgetClassifier(dim=1 << 12, censored="skip")
    r = next(workload(1))
    run(clf, [{**r, "finished": False}])
    assert clf.online.updates == 0 and acc(clf, "m1")["censored"] == 1 and acc(clf, "m1")["labels"] == 0


def test_censored_update_fixes_underestimates():
    clf = BudgetClassifier(dim=1 << 14)
    rows = [{**r, "thinking_tokens": 300} for r in workload(300, seed=3)]
    run(clf, rows)
    hard = [{**r, "id": f"cut-{i}", "thinking_tokens": 9000, "finished": False}
            for i, r in enumerate(workload(300, seed=4))]
    run(clf, hard)
    pred = run(clf, [{**next(workload(1, seed=5)), "id": "probe"}])[0][0]
    assert pred.bucket >= clf.buckets.index(9000)
    assert acc(clf, "m1")["censored"] == 300 and acc(clf, "m1")["labels"] == 301


def test_shadow_follows_pressure():
    clf = BudgetClassifier(dim=1 << 12, active_at=0.1)
    assert clf.shadow
    clf.set_pressure(0.3)
    assert not clf.shadow
    pred = run(clf, [next(workload(1))])[0][0]
    assert pred.shadow is False


def test_quantile_raises_budget():
    lo, hi = BudgetClassifier(dim=1 << 12, quantile=0.2), BudgetClassifier(dim=1 << 12, quantile=0.9)
    p = np.array([0.0, 0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0])
    assert lo._pick(p) == 1 and hi._pick(p) == 3


def test_unknown_request_and_observe_with_prompt():
    clf = BudgetClassifier(dim=1 << 12)
    assert clf.observe("nope", 100)["used"] is False
    res = clf.observe("late", 100, prompt="What is 2 + 2?", model="m1")
    assert res["used"] and res["predicted"] is None and acc(clf, "m1")["window"] == 0


def test_pending_is_bounded():
    clf = BudgetClassifier(dim=1 << 12, max_pending=5)
    async def go():
        for r in workload(20):
            await clf.predict(r["id"], r["prompt"], r["model"])
    asyncio.run(go())
    assert len(clf.pending) == 5


def test_save_and_load_roundtrip(tmp_path):
    clf = BudgetClassifier(dim=1 << 12)
    run(clf, workload(100))
    clf.save(tmp_path / "state.npz")
    other = BudgetClassifier(dim=1 << 12)
    other.load(tmp_path / "state.npz")
    r = next(workload(1, seed=9))
    a = run(clf, [r])[0][0]
    b = run(other, [{**r, "id": "x"}])[0][0]
    assert a.bucket == b.bucket and other.current_model == "m1" and acc(other, "m1")["labels"] == 101
