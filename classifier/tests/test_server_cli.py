import json

from starlette.testclient import TestClient

from budget_classifier import BudgetClassifier
from budget_classifier.cli import cli
from budget_classifier.server import create_app
from conftest import workload


def test_http_predict_observe_stats(tmp_path):
    state = tmp_path / "s.npz"
    with TestClient(create_app(BudgetClassifier(dim=1 << 12), state_path=state, save_every=2)) as c:
        assert c.get("/health").json() == {"ok": True}
        p = c.post("/v1/predict", json={"request_id": "r1", "prompt": "What is 2 + 2?", "model": "m1"}).json()
        assert p["shadow"] is True and p["source"] == "cold" and "budget" in p
        o = c.post("/v1/observe", json={"request_id": "r1", "thinking_tokens": 50}).json()
        assert o["used"] and o["bucket"] == 1
        assert c.post("/v1/pressure", json={"p": 0.5}).json()["shadow"] is False
        assert c.post("/v1/predict", json={"prompt": "x"}).status_code == 400
        assert c.post("/v1/observe", json={"request_id": "r1", "thinking_tokens": "lots"}).status_code == 400
        assert c.get("/v1/stats").json()["versions"]["m1"]["labels"] == 1
    assert state.exists()


def test_replay_cli(tmp_path):
    data = tmp_path / "rows.jsonl"
    rows = list(workload(300, model="a")) + list(workload(300, seed=1, scale=4.0, model="b"))
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out.jsonl"
    cli(["replay", str(data), "--out", str(out), "--report-every", "100", "--state", str(tmp_path / "st.npz")])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["model"] for x in lines[:-1]] == ["a", "a", "a", "b", "b", "b"]
    final = lines[-1]["final"]
    assert final["upgrades"] == 1 and final["versions"]["b"]["exact"] > 0.8
    assert (tmp_path / "st.npz").exists()
