import pytest

from lifeline_agent import dsl
from lifeline_agent.datasets import score


def test_arith_chain():
    assert dsl.execute("subtract(5829, 5735), divide(#0, 5735)") == pytest.approx(94 / 5735)


def test_constants_and_percent():
    assert dsl.execute("multiply(2, const_100)") == 200
    assert dsl.execute("add(const_m1, 3)") == 2
    assert dsl.execute("divide(21.0%, 1)") == pytest.approx(0.21)


def test_greater():
    assert dsl.execute("greater(3, 2)") == "yes"


def test_table_ops_parse_finqa_cells():
    rows = {"settlements": ["-13 ( 13 )", "-19 ( 19 )", "-2 ( 2 )"],
            "gross margin percentage": ["27.5% ( 27.5 % )", "27.9% ( 27.9 % )"],
            "net sales": ["$ 6,207", "$ 5742"]}
    assert dsl.execute("table_average(settlements, none)", rows) == pytest.approx(-34 / 3)
    assert dsl.execute("table_max(gross margin percentage, none)", rows) == pytest.approx(0.279)
    assert dsl.execute("table_sum(Net Sales, none)", rows) == 6207 + 5742


@pytest.mark.parametrize("bad", ["", "foo(1, 2)", "add(1)", "divide(1, 0)", "add(#3, 1)", "add(x, 1)"])
def test_errors(bad):
    with pytest.raises(dsl.ProgramError):
        dsl.execute(bad)


def test_score():
    assert score(0.0164, 0.0164) == (True, True)
    assert score(1.64, 0.0164) == (False, True)  # percent vs fraction
    assert score(0.0165, 0.0164) == (False, True)
    assert score(0.02, 0.0164) == (False, False)
    assert score("yes", "yes") == (True, True)
    assert score(None, 1.0) == (False, False)
