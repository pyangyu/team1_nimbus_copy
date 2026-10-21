from __future__ import annotations

import argparse
import os

from .datasets import math as mathsets
from .llm import NOTICES, BudgetDrop, LLMClient
from .workflow import AgentConfig


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=os.environ.get("LIFELINE_BASE_URL", "http://localhost:8000/v1"),
                   help="OpenAI-compatible endpoint: vLLM now, the gateway later")
    p.add_argument("--api-key", default=os.environ.get("LIFELINE_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.environ.get("LIFELINE_MODEL"))
    p.add_argument("--dataset", default="math500", choices=mathsets.DATASETS)
    p.add_argument("--levels", type=int, nargs="+", default=[4, 5],
                   help="MATH-500 levels to keep (ignored for other datasets)")
    p.add_argument("--budget", type=int, default=AgentConfig.think_budget, help="initial thinking budget (tokens)")
    p.add_argument("--max-budget", type=int, default=AgentConfig.max_budget)
    p.add_argument("--max-loops", type=int, default=AgentConfig.max_loops)
    p.add_argument("--verify-budget", type=int, default=AgentConfig.verify_budget)
    p.add_argument("--conf", type=float, default=AgentConfig.conf_threshold)
    p.add_argument("--answer-mode", default=AgentConfig.answer_mode, choices=["full", "final"],
                   help="final: after </think> only the final answer may be written")
    p.add_argument("--drop-at", type=int, help="SOLVE: thinking tokens before the budget drops")
    p.add_argument("--drop-to", type=int, help="SOLVE: thinking tokens allowed after the drop")
    p.add_argument("--notice", default="none", choices=list(NOTICES),
                   help="text inserted at the drop: none (silent), soft, count")
    p.add_argument("--out", default=None, help="output directory (default runs/<name>-<time>)")


def config_from(args) -> AgentConfig:
    if (args.drop_at is None) != (args.drop_to is None):
        raise SystemExit("--drop-at and --drop-to go together")
    drop = BudgetDrop(args.drop_at, args.drop_to, args.notice) if args.drop_at is not None else None
    return AgentConfig(think_budget=args.budget, max_budget=args.max_budget, max_loops=args.max_loops,
                       verify_budget=args.verify_budget, conf_threshold=args.conf,
                       answer_mode=args.answer_mode, drop=drop)


def examples_from(args):
    return mathsets.load(args.dataset, args.levels)


def client_from(args) -> LLMClient:
    return LLMClient(args.base_url, args.api_key, args.model)
