from math_agent.grading import extract, grade, last_boxed


def test_last_boxed_nested_and_last():
    assert last_boxed("first \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("no answer") is None
    assert last_boxed("\\fbox{7}") == "7"


def test_grade_normalisation():
    assert grade("\\dfrac{1}{2}", "\\frac{1}{2}")
    assert grade("$90^\\circ$", "90")
    assert grade("1,000", "1000")
    assert grade("42.0", "42")
    assert not grade("41", "42")
    assert not grade(None, "42")


def test_gsm8k_falls_back_to_last_number():
    assert extract("so she makes 18 dollars", "gsm8k") == "18"
    assert extract("so she makes 18 dollars", "math") is None
    assert grade("$18", "18", "gsm8k")
