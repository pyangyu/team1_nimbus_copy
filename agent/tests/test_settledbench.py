import math

from math_agent import settledbench as sb


def test_cut_points_are_edges_and_fractions_below_the_length():
    pts = sb.cut_points(3000)
    assert pts[0] >= 32 and all(a < b for a, b in zip(pts, pts[1:])) and all(p < 3000 for p in pts)
    assert 256 in pts and 2048 in pts and int(3000 * 0.9) in pts
    # 0.25 * 3000 = 750 sits within 8% of the length after 512 and collapses into it; so does 2250 after 2048
    assert pts == [256, 512, 1024, 1500, 2048, 2700]
    assert sb.cut_points(20) == []


def test_chars_at_uses_the_first_event_reaching_the_token_count():
    trail = [(1, 3), (2, 7), (5, 20), (6, 25)]
    assert sb.chars_at(trail, 2) == 7 and sb.chars_at(trail, 3) == 20 and sb.chars_at(trail, 9) is None


def test_deer_confidence_stops_at_the_closing_brace():
    toks, lps = ["4", "}", ".", "x"], [math.log(0.5), math.log(1.0), math.log(0.01), None]
    assert abs(sb.deer_confidence(toks, lps) - 0.75) < 1e-9
    assert sb.deer_confidence([], []) is None


def test_auc_and_early_exit():
    assert sb.auc([0.9, 0.2, 0.5], [True, False, True]) == 1.0
    assert sb.auc([0.5, 0.5], [True, False]) == 0.5 and sb.auc([1.0], [True]) is None
    rows = [{"id": "a", "cut": 100, "used": 1000, "uncut_correct": True, "silent_correct": False, "jev_settled": 0.2},
            {"id": "a", "cut": 500, "used": 1000, "uncut_correct": True, "silent_correct": True, "jev_settled": 0.9},
            {"id": "b", "cut": 100, "used": 400, "uncut_correct": False, "silent_correct": False, "jev_settled": 0.95}]
    e = sb.early_exit(rows, "jev_settled", 0.8)
    assert e == {"accuracy": 0.5, "mean_thinking": 300.0, "exited": 2, "n": 2}
    assert sb.early_exit(rows, "jev_settled", 0.99)["mean_thinking"] == 700.0
    o = sb.oracle_exit(rows)
    assert o["accuracy"] == 0.5 and o["mean_thinking"] == 450.0 and o["exited"] == 1


def test_summary_ranks_signals_and_simulates_exits():
    rows = [{"id": "a", "cut": 100, "frac": 0.1, "used": 1000, "natural": True, "uncut_correct": True,
             "silent_correct": False, "jev_settled": 0.1, "jev_done": 0.2, "deer": 0.3, "probe_match": False,
             "jev_ms": 80},
            {"id": "a", "cut": 800, "frac": 0.8, "used": 1000, "natural": True, "uncut_correct": True,
             "silent_correct": True, "jev_settled": 0.9, "jev_done": 0.8, "deer": 0.9, "probe_match": True,
             "jev_ms": 90}]
    s = sb.summarize(rows)
    assert s["problems"] == 1 and s["auc"]["jev_settled"]["auc"] == 1.0 and s["auc"]["probe_match"]["auc"] == 1.0
    assert s["by_depth"]["frac>0.75"]["silent_kept_right"] == 1.0 and s["jev_ms_mean"] == 85.0
    assert s["early_exit"]["jev_settled>=0.8"] == {"accuracy": 1.0, "mean_thinking": 800.0, "exited": 1, "n": 1}
    assert s["early_exit"]["oracle"]["mean_thinking"] == 800.0
