"""End-to-end workflow against a fake vLLM /v1/completions (SSE) server."""
import asyncio
import json

import httpx
import pytest

from lifeline_agent.datasets import Chunk, Example
from lifeline_agent.llm import LLMClient
from lifeline_agent.retrieval import Retriever
from lifeline_agent.workflow import AgentConfig, run_workflow

CORPUS = {"ACME/2009": [
    Chunk("p1:text_0", "ACME/2009", "net revenue was $ 5829 million in 2009 compared to $ 5735 million in 2008 ."),
    Chunk("p1:text_1", "ACME/2009", "the company opened two new plants ."),
    Chunk("p1:table_0", "ACME/2009", "net revenue: 2009 = 5829; 2008 = 5735",
          row_name="net revenue", cells=["5829", "5735"]),
]}
EX = Example("ACME/2009/page_1.pdf-1", "what was the percent change in net revenue from 2008 to 2009?",
             94 / 5735, "ACME/2009")


def fake_vllm(answers: dict[str, list[str]], seen: list[dict], think_len: int = 3):
    """answers maps a step prefix to successive answer texts."""
    counters = {k: 0 for k in answers}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        body = json.loads(req.content)
        h = {k: v for k, v in req.headers.items() if k.startswith("x-lifeline")}
        seen.append({"headers": h, "body": body})
        step, phase = h["x-lifeline-step"], h["x-lifeline-phase"]
        if phase == "think":
            n = min(think_len, body["max_tokens"])
            text, fin = "hmm " * n, ("length" if think_len > body["max_tokens"] else "stop")
        else:
            key = next(k for k in answers if step.startswith(k))
            i = min(counters[key], len(answers[key]) - 1)
            counters[key] += 1
            text, fin, n = answers[key][i], "stop", 5
        events = [{"choices": [{"text": text, "finish_reason": None}]},
                  {"choices": [{"text": "", "finish_reason": fin}]},
                  {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": n}}]
        sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def run(answers, cfg=AgentConfig(), think_len=3, deadline=None):
    seen = []
    llm = LLMClient("http://fake/v1", transport=fake_vllm(answers, seen, think_len))

    async def go():
        try:
            return await run_workflow(EX, Retriever(CORPUS), llm, cfg, workflow_id="w1", deadline=deadline)
        finally:
            await llm.aclose()

    return asyncio.run(go()), seen


def test_happy_path_one_round():
    r, seen = run({"plan": ["net revenue 2008\nnet revenue 2009"],
                   "reason": ["program: subtract(5829, 5735), divide(#0, 5735)"],
                   "check": ["0.95"]})
    assert r.error is None
    assert r.correct and r.rounds == 1 and r.confidence == 0.95
    # plan (1 req) + reason (think + answer) + check (1 req)
    assert [s["headers"]["x-lifeline-step"] for s in seen] == ["plan", "reason.0", "reason.0", "check.0"]
    assert all(s["headers"]["x-lifeline-class"] == "agentic" and s["headers"]["x-lifeline-workflow"] == "w1"
               for s in seen)
    assert r.llm_calls == 3 and r.thinking_tokens == 3


def test_low_confidence_widens_and_raises_budget():
    cfg = AgentConfig(k=1, k_step=2, think_budget=100, max_loops=2)
    r, seen = run({"plan": ["net revenue"],
                   "reason": ["program: divide(5829, 1)", "program: subtract(5829, 5735), divide(#0, 5735)"],
                   "check": ["0.2", "0.9"]}, cfg=cfg)
    assert r.rounds == 2 and r.correct
    budgets = [s["headers"]["x-lifeline-thinking-budget"] for s in seen if s["headers"]["x-lifeline-phase"] == "think"]
    assert budgets == ["100", "200"]
    ks = [t["k"] for t in r.trace if t["step"] == "retrieve"]
    assert ks == [1, 3]


def test_budget_exhaustion_forces_answer():
    cfg = AgentConfig(think_budget=2, max_loops=0)
    r, seen = run({"plan": ["x"], "reason": ["program: subtract(5829, 5735), divide(#0, 5735)"],
                   "check": ["0.9"]}, cfg=cfg, think_len=50)
    answer_prompt = [s for s in seen if s["headers"]["x-lifeline-phase"] == "answer"
                     and s["headers"]["x-lifeline-step"] == "reason.0"][0]["body"]["prompt"]
    assert "I have to give the answer now" in answer_prompt and answer_prompt.endswith("</think>\n\n")
    assert r.correct


def test_bad_program_retries_without_check():
    cfg = AgentConfig(max_loops=1)
    r, seen = run({"plan": ["x"], "reason": ["I am not sure", "program: subtract(5829, 5735), divide(#0, 5735)"],
                   "check": ["0.9"]}, cfg=cfg)
    steps = [s["headers"]["x-lifeline-step"] for s in seen]
    assert "check.0" not in steps and "check.1" in steps
    assert r.correct


def test_deadline_header_and_flag():
    r, seen = run({"plan": ["x"], "reason": ["program: add(1, 1)"], "check": ["1"]}, deadline=1.0)
    assert seen[0]["headers"]["x-lifeline-deadline"] == "1000"
    assert r.met_deadline is False and not r.correct
