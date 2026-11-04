import pytest

from budget_classifier.tailfeat import FEATURES, FeatureJudge, candidates, features


def test_candidates_normalize_and_keep_order():
    c = candidates("So the answer is 12.\n... wait. \\boxed{ 12 } ... Final answer: 13")
    assert [a for _, a in c] == ["12", "12", "13"] and c[0][0] < c[1][0] < c[2][0]


def test_features_cover_the_schema_and_tolerate_empty_input():
    f = features("The answer is 7. Wait, but hmm.\nSo the answer is 7. \\boxed{7}", [-0.1] * 200, 200)
    assert set(f) == set(FEATURES) and f["last_repeats"] == 3.0 and f["last_two_agree"] == 1.0
    g = features("", [], 0)
    assert set(g) == set(FEATURES) and g["has_candidate"] == 0.0 and g["lp_mean_tail"] == 0.0


def test_shipped_weights_load_and_score_sensibly():
    j = FeatureJudge.load()
    assert j.names == FEATURES and j.meta["rows"] > 100
    settled = j.score("The answer is 5. Therefore the answer is 5. Yes, correct. \\boxed{5}" * 3, [-0.05] * 400, 400)
    floundering = j.score("Wait, but hmm, maybe 3? Actually no, 4? Hold on, wrong." * 3, [-3.0] * 60, 60)
    assert 0.0 <= floundering < settled <= 1.0


def test_bad_weights_are_rejected():
    with pytest.raises(ValueError):
        FeatureJudge({"features": ["a", "b"], "mean": [0], "scale": [1], "coef": [1], "intercept": 0})
