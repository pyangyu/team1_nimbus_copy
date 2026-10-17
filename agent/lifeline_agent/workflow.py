"""One agentic workflow = one question, run as a small state machine:

  PLAN -> RETRIEVE -> REASON -> EXECUTE -> CHECK --(confident or out of loops)--> ANSWER
                ^                                   |
                +------ widen k, raise budget ------+

PLAN, REASON and CHECK are LLM calls; RETRIEVE and EXECUTE are local.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import dsl, prompts
from .datasets import Example, score
from .llm import CallMeta, CallResult, LLMClient
from .retrieval import Retriever


@dataclass
class AgentConfig:
    k: int = 8
    k_step: int = 8
    think_budget: int = 1024
    budget_growth: float = 2.0
    max_budget: int = 4096
    max_loops: int = 2  # extra REASON rounds after the first
    conf_threshold: float = 0.7
    plan_tokens: int = 128
    answer_tokens: int = 160
    check_tokens: int = 8


@dataclass
class WorkflowResult:
    workflow_id: str
    example_id: str
    question: str
    gold: float | str
    pred: float | str | None = None
    program: str = ""
    confidence: float | None = None
    correct: bool = False
    correct_lenient: bool = False
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


_PROG = re.compile(r"program\s*:\s*(.+)", re.I)
_NUM = re.compile(r"[01](?:\.\d+)?|\.\d+")


def parse_program(text: str) -> str:
    m = _PROG.search(text)
    line = m.group(1) if m else next((l for l in text.splitlines() if "(" in l), text)
    return line.strip().strip("`").strip()


def parse_confidence(text: str) -> float:
    m = _NUM.search(text)
    return min(1.0, float(m.group())) if m else 0.0


async def run_workflow(ex: Example, retriever: Retriever, llm: LLMClient, cfg: AgentConfig,
                       *, workflow_id: str | None = None, arrival: float | None = None,
                       deadline: float | None = None,
                       on_call: Callable[[CallResult], None] | None = None) -> WorkflowResult:
    wid = workflow_id or uuid.uuid4().hex[:12]
    now = time.time()
    res = WorkflowResult(wid, ex.id, ex.question, ex.gold, arrival=arrival or now, start=now,
                         deadline=deadline)
    deadline_ms = int(deadline * 1000) if deadline else None

    async def call(step: str, messages: list[dict], budget: int, max_tokens: int) -> CallResult:
        r = await llm.generate(messages, CallMeta(wid, step, deadline_ms),
                               think_budget=budget, max_answer_tokens=max_tokens)
        res.llm_calls += 1
        res.prompt_tokens += r.prompt_tokens
        res.thinking_tokens += r.thinking_tokens
        res.answer_tokens += r.answer_tokens
        res.trace.append({"step": step, "budget": budget, "latency_s": round(r.latency_s, 3),
                          "thinking_tokens": r.thinking_tokens, "answer_tokens": r.answer_tokens,
                          "exhausted": r.budget_exhausted, "text": r.text[:300]})
        if on_call:
            on_call(r)
        return r

    try:
        plan = await call("plan", prompts.plan_messages(ex.question, ex.doc), 0, cfg.plan_tokens)
        query = ex.question + "\n" + plan.text
        k, budget = cfg.k, cfg.think_budget
        for rnd in range(cfg.max_loops + 1):
            res.rounds = rnd + 1
            chunks = retriever.search(ex.doc, query, k)
            res.trace.append({"step": "retrieve", "k": k, "chunks": [c.id for c in chunks]})
            r = await call(f"reason.{rnd}", prompts.reason_messages(ex.question, ex.doc, chunks),
                           budget, cfg.answer_tokens)
            program = parse_program(r.text)
            rows = {c.row_name: c.cells for c in chunks if c.row_name}
            try:
                value = dsl.execute(program, rows)
                exec_err = None
            except dsl.ProgramError as e:
                value, exec_err = None, str(e)
            res.trace.append({"step": "execute", "program": program, "value": value, "error": exec_err})
            res.program, res.pred = program, value
            last = rnd == cfg.max_loops
            if value is not None:
                c = await call(f"check.{rnd}", prompts.check_messages(ex.question, chunks, program, value),
                               0, cfg.check_tokens)
                res.confidence = parse_confidence(c.text)
                if res.confidence >= cfg.conf_threshold:
                    break
            if not last:
                k += cfg.k_step
                budget = int(min(cfg.max_budget, budget * cfg.budget_growth))
    except Exception as e:  # a failed call still produces a scored (wrong) workflow
        res.error = f"{type(e).__name__}: {e}"
    res.end = time.time()
    res.latency_s = res.end - res.start
    if deadline is not None:
        res.met_deadline = res.end <= deadline
    res.correct, res.correct_lenient = score(res.pred, ex.gold)
    return res
