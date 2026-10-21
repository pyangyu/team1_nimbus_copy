import asyncio

from conftest import fake_vllm

from math_agent.datasets import Example
from math_agent.llm import EARLY_STOP, FINAL_PREFIX, NOTICES, BudgetDrop, CallMeta, LLMClient
from math_agent.workflow import AgentConfig, run_workflow

EX = Example("t/1", "What is 6 times 7?", "42", "math")


def run(answers, cfg=AgentConfig(), think_len=3, deadline=None):
    seen = []
    llm = LLMClient("http://fake/v1", transport=fake_vllm(answers, seen, think_len))

    async def go():
        try:
            return await run_workflow(EX, llm, cfg, workflow_id="w1", deadline=deadline)
        finally:
            await llm.aclose()

    return asyncio.run(go()), seen


def think_requests(seen, step):
    return [s for s in seen if s["headers"]["x-lifeline-step"] == step
            and s["headers"]["x-lifeline-phase"] == "think"]


def test_happy_path_one_round():
    r, seen = run({"plan": ["multiply"], "solve": ["so \\boxed{42}"], "verify": ["0.95"]})
    assert r.error is None
    assert r.correct and r.rounds == 1 and r.pred == "42" and r.confidence == 0.95
    assert [s["headers"]["x-lifeline-step"] for s in seen] == \
        ["plan", "solve.0", "solve.0", "verify.0", "verify.0"]
    assert all(s["headers"]["x-lifeline-class"] == "agentic" and s["headers"]["x-lifeline-workflow"] == "w1"
               for s in seen)
    assert r.llm_calls == 3


def test_rejected_answer_retries_with_bigger_budget():
    cfg = AgentConfig(think_budget=100, max_loops=2)
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{41}", "\\boxed{42}"], "verify": ["0.2", "0.9"]}, cfg=cfg)
    assert r.rounds == 2 and r.correct
    assert [t["headers"]["x-lifeline-thinking-budget"] for t in think_requests(seen, "solve.0")
            + think_requests(seen, "solve.1")] == ["100", "200"]
    retry_prompt = think_requests(seen, "solve.1")[0]["body"]["prompt"]
    assert "Earlier attempts answered 41" in retry_prompt


def test_last_round_is_not_verified():
    cfg = AgentConfig(max_loops=0)
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{42}"], "verify": ["0.9"]}, cfg=cfg)
    assert r.correct and not any(s["headers"]["x-lifeline-step"].startswith("verify") for s in seen)


def test_no_boxed_answer_retries_without_verify():
    cfg = AgentConfig(max_loops=1)
    r, seen = run({"plan": ["x"], "solve": ["I am not sure", "\\boxed{42}"], "verify": ["0.9"]}, cfg=cfg)
    steps = [s["headers"]["x-lifeline-step"] for s in seen]
    assert "verify.0" not in steps and r.rounds == 2 and r.correct


def test_budget_exhaustion_forces_answer_with_early_stop():
    cfg = AgentConfig(think_budget=2, max_loops=0)
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{42}"]}, cfg=cfg, think_len=50)
    answer = [s for s in seen if s["headers"]["x-lifeline-phase"] == "answer"
              and s["headers"]["x-lifeline-step"] == "solve.0"][0]["body"]["prompt"]
    assert answer.endswith(EARLY_STOP + "</think>\n\n") and r.correct


def test_final_answer_mode():
    cfg = AgentConfig(max_loops=0, answer_mode="final", final_tokens=48)
    r, seen = run({"plan": ["x"], "solve": ["42}"]}, cfg=cfg)
    answer = [s for s in seen if s["headers"]["x-lifeline-phase"] == "answer"
              and s["headers"]["x-lifeline-step"] == "solve.0"][0]["body"]
    assert answer["prompt"].endswith("</think>\n\n" + FINAL_PREFIX) and answer["max_tokens"] == 48
    assert r.pred == "42" and r.correct


def test_budget_drop_with_count_notice():
    cfg = AgentConfig(think_budget=100, max_loops=0, drop=BudgetDrop(at=10, to=5, notice="count"))
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{42}"]}, cfg=cfg, think_len=50)
    first, second = think_requests(seen, "solve.0")
    assert first["body"]["max_tokens"] == 10 and "x-lifeline-budget-drop" not in first["headers"]
    assert second["body"]["max_tokens"] == 5 and second["headers"]["x-lifeline-budget-drop"] == "10,5,count"
    notice = NOTICES["count"].format(W=5)
    assert second["body"]["prompt"].count(notice) == 1 and second["body"]["prompt"].endswith(notice)
    solve = next(t for t in r.trace if t["step"] == "solve.0")
    assert solve["dropped"] and solve["closed_itself"] is False and solve["thinking_tokens"] == 15


def test_budget_drop_model_closes_itself():
    cfg = AgentConfig(think_budget=100, max_loops=0, drop=BudgetDrop(at=10, to=20, notice="soft"))
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{42}"]}, cfg=cfg, think_len=15)
    answer = [s for s in seen if s["headers"]["x-lifeline-phase"] == "answer"
              and s["headers"]["x-lifeline-step"] == "solve.0"][0]["body"]["prompt"]
    assert EARLY_STOP not in answer
    solve = next(t for t in r.trace if t["step"] == "solve.0")
    assert solve["dropped"] and solve["closed_itself"] is True and solve["thinking_tokens"] == 15


def test_no_drop_when_thinking_ends_first():
    cfg = AgentConfig(think_budget=100, max_loops=0, drop=BudgetDrop(at=10, to=5))
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{42}"]}, cfg=cfg, think_len=3)
    assert len(think_requests(seen, "solve.0")) == 1
    assert not next(t for t in r.trace if t["step"] == "solve.0")["dropped"]


def test_deadline_header_and_flag():
    r, seen = run({"plan": ["x"], "solve": ["\\boxed{1}"], "verify": ["1"]}, deadline=1.0)
    assert seen[0]["headers"]["x-lifeline-deadline"] == "1000"
    assert r.met_deadline is False and not r.correct


def test_plain_client_budget_zero_is_one_request():
    seen = []
    llm = LLMClient("http://fake/v1", transport=fake_vllm({"s": ["hi"]}, seen))

    async def go():
        try:
            return await llm.generate([{"role": "user", "content": "q"}], CallMeta("w", "s"),
                                      think_budget=0, max_answer_tokens=4)
        finally:
            await llm.aclose()

    res = asyncio.run(go())
    assert len(seen) == 1 and seen[0]["body"]["prompt"].endswith("<think>\n\n</think>\n\n") and res.text == "hi"
