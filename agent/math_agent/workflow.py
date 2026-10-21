"""One agentic workflow = one math problem, run as a small state machine:

  PLAN -> SOLVE -> VERIFY --(confident or out of loops)--> ANSWER
            ^         |
            +---------+  rejected: raise the thinking budget, tell SOLVE what was rejected

PLAN has no thinking. SOLVE is the long reasoning call (thinking budget, optional
mid-thinking BudgetDrop, optional final-answer-only mode). VERIFY is a short
reasoning call that returns a confidence.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import prompts
from .datasets import Example
from .grading import extract, grade
from .llm import BudgetDrop, CallMeta, CallResult, LLMClient


@dataclass
class AgentConfig:
    think_budget: int = 4096
    budget_growth: float = 2.0
    max_budget: int = 6144
    max_loops: int = 1  # extra SOLVE rounds after the first
    verify_budget: int = 512
    conf_threshold: float = 0.7
    plan_tokens: int = 128
    answer_tokens: int = 1024
    verify_tokens: int = 8
    answer_mode: str = "full"
    final_tokens: int = 48
    drop: BudgetDrop | None = None


@dataclass
class WorkflowResult:
    workflow_id: str
    example_id: str
    question: str
    gold: str
    pred: str | None = None
    confidence: float | None = None
    correct: bool = False
    rounds: int = 0
    arrival: float = 0.0
    start: float = 0.0
    end: float = 0.0
    deadline: float | None = None
    met_deadline: bool | None = None
    latency_s: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)

    def record(self) -> dict:
        return asdict(self)


_NUM = re.compile(r"[01](?:\.\d+)?|\.\d+")


def parse_confidence(text: str) -> float:
    m = _NUM.search(text)
    return min(1.0, float(m.group())) if m else 0.0


async def run_workflow(ex: Example, llm: LLMClient, cfg: AgentConfig,
                       *, workflow_id: str | None = None, arrival: float | None = None,
                       deadline: float | None = None,
                       on_call: Callable[[CallResult], None] | None = None) -> WorkflowResult:
    wid = workflow_id or uuid.uuid4().hex[:12]
    now = time.time()
    res = WorkflowResult(wid, ex.id, ex.question, ex.gold, arrival=arrival or now, start=now,
                         deadline=deadline)
    deadline_ms = int(deadline * 1000) if deadline else None

    async def call(step: str, messages: list[dict], budget: int, max_tokens: int, **kw) -> CallResult:
        r = await llm.generate(messages, CallMeta(wid, step, deadline_ms),
                               think_budget=budget, max_answer_tokens=max_tokens, **kw)
        res.llm_calls += 1
        res.prompt_tokens += r.prompt_tokens
        res.thinking_tokens += r.thinking_tokens
        res.answer_tokens += r.answer_tokens
        res.trace.append({"step": step, "budget": budget, "latency_s": round(r.latency_s, 3),
                          "thinking_tokens": r.thinking_tokens, "answer_tokens": r.answer_tokens,
                          "exhausted": r.budget_exhausted, "dropped": r.dropped,
                          "closed_itself": r.closed_itself, "text": r.text[-300:]})
        if on_call:
            on_call(r)
        return r

    try:
        plan = await call("plan", prompts.plan_messages(ex.question), 0, cfg.plan_tokens)
        budget, rejected = cfg.think_budget, []
        for rnd in range(cfg.max_loops + 1):
            res.rounds = rnd + 1
            r = await call(f"solve.{rnd}", prompts.solve_messages(ex.question, plan.text, rejected),
                           budget, cfg.answer_tokens, answer_mode=cfg.answer_mode,
                           final_tokens=cfg.final_tokens, drop=cfg.drop)
            pred = extract(r.text, ex.dataset)
            res.trace.append({"step": "extract", "pred": pred})
            if pred is not None:
                res.pred = pred
            if rnd == cfg.max_loops:
                break
            if pred is not None:
                v = await call(f"verify.{rnd}", prompts.verify_messages(ex.question, pred),
                               cfg.verify_budget, cfg.verify_tokens)
                res.confidence = parse_confidence(v.text)
                if res.confidence >= cfg.conf_threshold:
                    break
                rejected.append(pred)
            budget = int(min(cfg.max_budget, budget * cfg.budget_growth))
    except Exception as e:  # a failed call still produces a scored (wrong) workflow
        res.error = f"{type(e).__name__}: {e}"
    res.end = time.time()
    res.latency_s = res.end - res.start
    if deadline is not None:
        res.met_deadline = res.end <= deadline
    res.correct = grade(res.pred, ex.gold, ex.dataset)
    return res
