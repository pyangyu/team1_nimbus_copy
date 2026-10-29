import random

from math_agent.datasets import Example, burstgpt
from math_agent.datasets.math import source
from math_agent.loadgen import draw_mixed, parse_map


def test_burstgpt_arrivals_keep_the_mean_rate_and_the_clustering():
    off = burstgpt.arrivals(64, 2.0, seed=3)
    assert len(off) == 64 and off[0] == 0.0 and all(a <= b for a, b in zip(off, off[1:]))
    assert abs(off[-1] - 63 / 2.0) < 1e-6, "the window is rescaled so the mean rate is exactly 2/s"
    assert burstgpt.arrivals(64, 2.0, seed=3) == off and burstgpt.arrivals(64, 2.0, seed=4) != off
    # at the scale of one run (64 arrivals) the trace's gaps are Poisson-like on average (CV about 1) and
    # bursty in the tail; the trace's burstiness lives in how the rate moves over minutes, not within seconds
    cvs = sorted(burstgpt.burstiness(burstgpt.arrivals(64, 2.0, seed=s)) for s in range(100))
    assert 0.9 < sum(cvs) / len(cvs) < 1.3 and cvs[90] > 1.2
    assert burstgpt.arrivals(1, 2.0, 0) == [0.0] and burstgpt.arrivals(0, 2.0, 0) == []


def test_window_takes_consecutive_arrivals():
    counts = [0, 3, 0, 0, 2, 1]
    t = burstgpt.window(counts, 4, random.Random(0))
    assert len(t) == 4 and all(1 <= x < 6 for x in t)


def test_mixed_sources_and_maps():
    pool = [Example("gsm8k/1", "q", "1", "gsm8k"), Example("test/algebra/1.json", "q", "1", "math", 4),
            Example("aime_2026/1", "q", "1", "math")]
    assert [source(e) for e in pool] == ["gsm8k", "math500", "aime2026"]
    drawn = draw_mixed(pool, 300, {"gsm8k": 0.5, "math500": 0.35, "aime2026": 0.15}, random.Random(0))
    counts = {s: sum(source(e) == s for e in drawn) for s in ("gsm8k", "math500", "aime2026")}
    assert counts["gsm8k"] > counts["math500"] > counts["aime2026"] > 0
    assert parse_map("gsm8k=1024,math500=4096", int) == {"gsm8k": 1024, "math500": 4096} and parse_map(None) == {}
