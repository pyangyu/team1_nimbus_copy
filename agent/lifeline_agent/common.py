from __future__ import annotations

import argparse
import os

from .llm import LLMClient
from .workflow import AgentConfig


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=os.environ.get("LIFELINE_BASE_URL", "http://localhost:8000/v1"),
                   help="OpenAI-compatible endpoint: vLLM now, the Lifeline Gateway later")
    p.add_argument("--api-key", default=os.environ.get("LIFELINE_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.environ.get("LIFELINE_MODEL"))
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--budget", type=int, default=AgentConfig.think_budget, help="initial thinking budget (tokens)")
    p.add_argument("--max-budget", type=int, default=AgentConfig.max_budget)
    p.add_argument("--max-loops", type=int, default=AgentConfig.max_loops)
    p.add_argument("--k", type=int, default=AgentConfig.k)
    p.add_argument("--conf", type=float, default=AgentConfig.conf_threshold)
    p.add_argument("--out", default=None, help="output directory (default runs/<name>-<time>)")


def config_from(args) -> AgentConfig:
    return AgentConfig(k=args.k, think_budget=args.budget, max_budget=args.max_budget,
                       max_loops=args.max_loops, conf_threshold=args.conf)


def client_from(args) -> LLMClient:
    return LLMClient(args.base_url, args.api_key, args.model)
