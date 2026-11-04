import json
import math
import random

import pytest

from math_agent import settledfeat as sf


def test_candidates_are_normalized_and_ordered():
    text = "So the answer is 12.\n... wait, recompute. \\boxed{ 12 } ... Final answer: 13"
    c = sf.candidates(text)
    assert [a for _, a in c] == ["12", "12", "13"]
    assert c[0][0] < c[1][0] < c[2][0]


def test_features_count_tail_phrases_and_candidate_agreement():
    prefix = "Let me compute. The answer is 7. Wait, but actually hmm.\nSo the answer is 7. \\boxed{7}"
    f = sf.features(prefix, [-0.1] * 200, 200)
    assert set(f) == set(sf.FEATURES)
    assert f["has_candidate"] == 1.0 and f["n_candidates"] == 1.0 and f["last_repeats"] == 3.0
    assert f["last_two_agree"] == 1.0 and f["backtrack_per_k"] > 0 and f["verify_per_k"] > 0
    assert abs(f["log_tokens"] - math.log(200)) < 1e-9
    assert abs(f["lp_mean_tail"] + 0.1) < 1e-9 and f["lp_surprise_tail"] == 0.0


def test_features_tolerate_missing_logprobs_and_no_candidates():
    f = sf.features("just thinking without a number", [], 5)
    assert f["has_candidate"] == 0.0 and f["frac_since_new"] == 0.0 and f["lp_mean_tail"] == 0.0


def test_group_folds_keep_a_problem_in_one_fold():
    ids = ["a", "a", "b", "c", "c", "d", "e"]
    folds = sf.group_folds(ids, 3, 0)
    by = {}
    for i, f in zip(ids, folds):
        by.setdefault(i, set()).add(f)
    assert all(len(v) == 1 for v in by.values()) and set(folds) == {0, 1, 2}


def test_evaluate_runs_grouped_cv_on_synthetic_rows(tmp_path):
    pytest.importorskip("sklearn")
    rng = random.Random(0)
    rows, traces = [], []
    for p in range(24):
        settled = p % 2 == 0
        thinking = ("The answer is 5. Yes, correct, \\boxed{5}. " * 40) if settled else \
            ("Wait, but hmm, maybe 3? Actually no, 4? " * 40)
        trail = [(n, min(len(thinking), n * 4)) for n in range(1, 401)]
        traces.append({"id": f"p{p}", "thinking": thinking, "trail": trail,
                       "token_logprobs": [rng.uniform(-0.2, 0) if settled else rng.uniform(-3, 0) for _ in range(400)]})
        for cut in (100, 200, 300):
            rows.append({"id": f"p{p}", "cut": cut, "used": 400, "uncut_correct": True,
                         "silent_correct": settled or cut == 300, "jev_done": 0.9 if settled else 0.3,
                         "jev_settled": 0.8 if settled else 0.2, "deer": 0.7, "probe_match": settled})
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    rows2, traces2 = sf.load(tmp_path)
    scored = sf.build(rows2, traces2)
    assert len(scored) == 72
    s = sf.evaluate(scored, k=4, repeats=2)
    assert s["rows"] == 72 and s["problems"] == 24 and s["folds"] == 4
    assert s["auc"]["feat_lr"] > 0.9 and s["auc"]["feat_gbt"] > 0.9 and s["auc"]["jev_done"] > 0.8
    assert "oracle" in s["early_exit"] and "feat_lr>=0.7" in s["early_exit"]
    assert set(s["lr_coefficients"]) == set(sf.FEATURES)
