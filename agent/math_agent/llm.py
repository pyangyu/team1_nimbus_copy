"""OpenAI-compatible completions client with a hard thinking budget.

Every call goes to one base_url. Today that is vLLM; later it is the gateway,
which reads the X-Lifeline-* headers to track deadlines and pick a
tier. vLLM ignores them.

Thinking budget (budget forcing, Qwen3 chat format):
  phase "think":  prompt + "<think>\\n", max_tokens=budget, stop at </think>
  phase "answer": prompt + thinking + "</think>\\n\\n", max_tokens=answer cap
If the budget runs out mid-thought, Qwen's early-stop line is appended before
</think> so the model answers instead of continuing to reason. Budget 0 uses
Qwen3's empty think block (non-thinking mode) and makes one request.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import httpx

EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution "
              "based on the thinking directly now.\n")
FINAL_PREFIX = "The final answer is \\boxed{"
NOTICES = {
    "none": "",
    "soft": "\n\nWait, I'm running out of time. I need to wrap up my reasoning quickly and give the "
            "final answer soon.\n",
    "count": "\n\nI only have about {W} tokens of thinking left, so I must finish my reasoning quickly "
             "and move to the final answer.\n",
}


class Rejected(RuntimeError):
    pass


@dataclass
class BudgetDrop:
    at: int
    to: int
    notice: str = "none"

    def header(self) -> str:
        return f"{self.at},{self.to},{self.notice}"


@dataclass
class CallMeta:
    workflow_id: str
    step: str
    deadline_ms: int | None = None
    tier: str | None = None  # optional hint; normally the gateway decides
    cls: str = "agentic"


@dataclass
class Phase:
    phase: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float | None = None
    latency_s: float = 0.0
    finish_reason: str | None = None
    tier: str | None = None  # echoed back by the gateway, if any
    priority: str | None = None
    gateway: dict | None = None


@dataclass
class CallResult:
    workflow_id: str
    step: str
    budget: int
    text: str
    thinking: str = ""
    started_at: float = 0.0
    latency_s: float = 0.0
    ttft_s: float | None = None
    prompt_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    budget_exhausted: bool = False
    dropped: bool = False
    closed_itself: bool | None = None
    phases: list[Phase] = field(default_factory=list)

    def record(self) -> dict:
        d = asdict(self)
        d.pop("thinking")
        return d


def render_chat(messages: list[dict]) -> str:
    """Qwen3 ChatML, ending with an open assistant turn."""
    out = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(out) + "<|im_start|>assistant\n"


class LLMClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str | None = None,
                 timeout: float = 900.0, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = httpx.AsyncClient(timeout=timeout, transport=transport,
                                      headers={"Authorization": f"Bearer {api_key}"},
                                      limits=httpx.Limits(max_connections=1000))

    async def aclose(self) -> None:
        await self.http.aclose()

    async def resolve_model(self) -> str:
        if not self.model:
            r = await self.http.get(f"{self.base_url}/models")
            r.raise_for_status()
            self.model = r.json()["data"][0]["id"]
        return self.model

    def _headers(self, meta: CallMeta, budget: int, phase: str,
                 drop: BudgetDrop | None = None) -> dict[str, str]:
        h = {"X-Lifeline-Class": meta.cls, "X-Lifeline-Workflow": meta.workflow_id,
             "X-Lifeline-Step": meta.step, "X-Lifeline-Phase": phase,
             "X-Lifeline-Thinking-Budget": str(budget)}
        if drop is not None:
            h["X-Lifeline-Budget-Drop"] = drop.header()
        if meta.deadline_ms is not None:
            h["X-Lifeline-Deadline"] = str(meta.deadline_ms)
        if meta.tier:
            h["X-Lifeline-Tier"] = meta.tier
        return h

    async def _complete(self, prompt: str, max_tokens: int, stop: list[str] | None,
                        headers: dict[str, str], phase: str, temperature: float) -> tuple[str, Phase]:
        body = {"model": await self.resolve_model(), "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}, "skip_special_tokens": False}
        if stop:
            body["stop"] = stop
        ph = Phase(phase)
        t0 = time.perf_counter()
        parts: list[str] = []
        async with self.http.stream("POST", f"{self.base_url}/completions",
                                    json=body, headers=headers) as r:
            if r.status_code == 429:
                raise Rejected(f"429: {(await r.aread())[:300]!r}")
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code}: {(await r.aread())[:300]!r}")
            ph.tier = r.headers.get("x-lifeline-tier")
            ph.priority = r.headers.get("x-lifeline-priority")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                for ch in ev.get("choices") or []:
                    if ch.get("text"):
                        if ph.ttft_s is None:
                            ph.ttft_s = time.perf_counter() - t0
                        parts.append(ch["text"])
                    if ch.get("finish_reason"):
                        ph.finish_reason = ch["finish_reason"]
                if ev.get("usage"):
                    ph.prompt_tokens = ev["usage"]["prompt_tokens"]
                    ph.completion_tokens = ev["usage"]["completion_tokens"]
                if ev.get("lifeline"):
                    ph.gateway = ev["lifeline"]
        ph.latency_s = time.perf_counter() - t0
        return "".join(parts), ph

    async def generate(self, messages: list[dict], meta: CallMeta, *, think_budget: int,
                       max_answer_tokens: int, temperature: float = 0.0,
                       answer_mode: str = "full", final_tokens: int = 48,
                       drop: BudgetDrop | None = None) -> CallResult:
        base = render_chat(messages)
        res = CallResult(meta.workflow_id, meta.step, think_budget, text="", started_at=time.time())
        t0 = time.perf_counter()
        if think_budget <= 0:
            prefix = base + "<think>\n\n</think>\n\n"
        else:
            first = min(think_budget, drop.at) if drop else think_budget
            thought, ph = await self._complete(base + "<think>\n", first, ["</think>"],
                                               self._headers(meta, think_budget, "think"),
                                               "think", temperature)
            res.phases.append(ph)
            res.thinking_tokens = ph.completion_tokens
            exhausted = ph.finish_reason == "length"
            if drop and exhausted and drop.at < think_budget:
                res.dropped = True
                thought += NOTICES[drop.notice].format(W=drop.to)
                more, ph = await self._complete(base + "<think>\n" + thought, drop.to, ["</think>"],
                                                self._headers(meta, drop.to, "think", drop),
                                                "think", temperature)
                res.phases.append(ph)
                thought += more
                res.thinking_tokens += ph.completion_tokens
                exhausted = ph.finish_reason == "length"
                res.closed_itself = not exhausted
            res.thinking = thought
            res.budget_exhausted = exhausted
            prefix = base + "<think>\n" + thought + (EARLY_STOP if exhausted else "") + "</think>\n\n"
        final = answer_mode == "final"
        if final:
            prefix += FINAL_PREFIX
        text, ph = await self._complete(prefix, final_tokens if final else max_answer_tokens, ["<|im_end|>"],
                                        self._headers(meta, think_budget, "answer"),
                                        "answer", temperature)
        res.phases.append(ph)
        res.text = (FINAL_PREFIX + text if final else text).strip()
        res.answer_tokens = ph.completion_tokens
        res.prompt_tokens = res.phases[0].prompt_tokens
        res.ttft_s = res.phases[0].ttft_s
        res.latency_s = time.perf_counter() - t0
        return res
