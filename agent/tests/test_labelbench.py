from math_agent import labelbench as lb


def test_notices_format_with_the_token_count():
    for text in lb.NOTICE_TEXTS.values():
        assert "256" in text.format(N=256, W=256)


def test_summary_conditions_on_the_probe_label():
    rows = [{"id": "a", "frac": 0.3, "uncut_correct": True, "probe_match": True, "silent_correct": True},
            {"id": "a", "frac": 0.6, "uncut_correct": True, "probe_match": False, "silent_correct": False},
            {"id": "b", "frac": 0.6, "uncut_correct": False, "probe_match": True, "silent_correct": False}]
    s = lb.summarize(rows, ["silent"])
    assert s["problems"] == 2 and s["all"]["silent_kept_given_match"] == 1.0
    assert s["all"]["silent_kept_given_mismatch"] == 0.0 and s["all"]["silent_n_match"] == 1
