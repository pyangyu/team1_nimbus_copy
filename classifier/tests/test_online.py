import numpy as np

from budget_classifier.features import BIAS, featurize
from budget_classifier.online import OnlineModel


def test_featurize_is_deterministic_and_normalized():
    a, b = featurize("Prove that 2 + 2 = 4"), featurize("Prove that 2 + 2 = 4")
    assert a == b and a[BIAS] == 1.0
    norm = sum(v * v for i, v in a.items() if i != BIAS)
    assert abs(norm - 1.0) < 1e-6


def test_cold_model_is_uniform():
    m = OnlineModel(10, dim=1 << 12)
    assert np.allclose(m.proba(featurize("hello", 1 << 12)), 0.1)


def test_learns_two_classes():
    m = OnlineModel(10, dim=1 << 14)
    easy, hard = featurize("translate cat into french", 1 << 14), featurize("aime find all ordered pairs", 1 << 14)
    for _ in range(50):
        m.update_exact(easy, 1)
        m.update_exact(hard, 8)
    assert m.proba(easy).argmax() == 1
    assert m.proba(hard).argmax() == 8


def test_soft_target_is_ordinal():
    t = OnlineModel(10, dim=16).target(4)
    assert t.argmax() == 4 and t[3] == t[5] and t[3] > t[2] > t[1]


def test_censored_update_only_pushes_up():
    m = OnlineModel(10, dim=1 << 12)
    x = featurize("long proof", 1 << 12)
    for _ in range(30):
        m.update_at_least(x, 6)
    p = m.proba(x)
    assert p[6:].sum() > 0.9
    m2 = OnlineModel(10, dim=1 << 12)
    m2.update_at_least(x, 0)
    assert m2.updates == 0
