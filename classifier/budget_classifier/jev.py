from __future__ import annotations

import time

import numpy as np
import httpx

from .buckets import Buckets
from .memory import Example

OBJECTIVE = (
    "A reasoning model is about to answer this request. Predict how many thinking tokens it will "
    "spend before it closes its reasoning, if nothing limits it. Recent requests to the same model "
    "and the thinking tokens they actually used are listed for calibration."
)
OBJECTIVE_NEEDED = (
    "A reasoning model is about to answer this request. Predict the smallest thinking budget at which "
    "its final answer is already the same as with unlimited thinking: the point past which more thinking "
    "no longer changes the answer. Recent requests to the same model and the thinking budget each one "
    "turned out to need are listed for calibration."
)
LABELS = {
    "length": (OBJECTIVE, "thinking_tokens_used", "How many thinking tokens will the model use on this request?"),
    "needed": (OBJECTIVE_NEEDED, "thinking_tokens_needed",
               "How many thinking tokens does the model need before its answer stops changing?"),
}

# the settled judge: Jev reads the end of a request's thinking so far and says whether the answer is fixed
OBJECTIVE_SETTLED = (
    "A reasoning model is in the middle of thinking about this request. The end of its thinking so far is "
    "shown. The server may have to stop the thinking here and make the model answer at once. Judge whether "
    "the final answer is already settled: if stopped now, would the model give the same answer as with "
    "unlimited thinking?"
)
SETTLED_LEVELS = [
    "still exploring: no candidate answer yet, or the approach is still being chosen",
    "has a candidate answer but is reconsidering it, finding problems, or trying another approach",
    "has an answer and is checking it; the check is going well and the answer is unlikely to change",
    "settled: the final answer is fixed and the remaining thinking only restates or re-verifies it",
]
SETTLED_FROM = 2  # P(settled) = mass on these levels and above
DONE_QUESTION = ("If the thinking were stopped right here and the model had to give its final answer "
                 "immediately, would it give the same answer as with unlimited thinking?")


def settled_state(prompt: str, thinking: str, used: int, *, head_chars: int = 1000, tail_chars: int = 1500) -> dict:
    req = {"prompt_chars": len(prompt), "prompt_head": prompt[:head_chars]}
    tail = thinking[-tail_chars:] if len(thinking) > tail_chars else thinking
    return {"objective": OBJECTIVE_SETTLED, "request": req,
            "thinking_so_far": {"tokens": int(used), "chars": len(thinking),
                                "truncated_at_start": len(thinking) > tail_chars, "tail": tail}}


def settled_questions() -> dict:
    return {"settled": {"type": "score", "instructions": "How far along is the thinking toward a fixed final answer?",
                        "criteria": SETTLED_LEVELS},
            "done": {"type": "noul", "instructions": DONE_QUESTION}}


def settled_from_answers(answers: dict | None) -> dict | None:
    """P(settled) from the score question, P(done) from the noul question, and the mean score level."""
    if not answers:
        return None
    out = {}
    try:
        probs = answers["settled"]["probabilities"]
        p = [float(probs[str(i)]) for i in range(len(SETTLED_LEVELS))]
        if all(np.isfinite(p)) and sum(p) > 0:
            p = [v / sum(p) for v in p]
            out["settled"] = sum(p[SETTLED_FROM:])
            out["level"] = sum(i * v for i, v in enumerate(p))
            out["levels"] = [round(v, 4) for v in p]
    except Exception:
        pass
    try:
        out["done"] = float(answers["done"]["noul"])
    except Exception:
        pass
    return out or None


class JevPrior:
    def __init__(self, buckets: Buckets, url: str | list[str] = "http://127.0.0.1:8791/v1/systemone", *,
                 api_key: str | None = None, model: str = "open-jev", timeout: float = 2.0,
                 head_chars: int = 1000, tail_chars: int = 300, max_inflight: int = 1,
                 transport: httpx.AsyncBaseTransport | None = None, label: str = "length"):
        if label not in LABELS:
            raise ValueError(f"label must be one of {tuple(LABELS)}")
        self.label = label
        self.urls = [u.strip() for u in (url.split(",") if isinstance(url, str) else url) if u.strip()]
        self.url_inflight = {u: 0 for u in self.urls}
        self.buckets, self.url, self.model = buckets, self.urls[0], model
        self.head_chars, self.tail_chars = head_chars, tail_chars
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(timeout=timeout, headers=headers, transport=transport)
        self.calls = self.failures = self.skipped_busy = self.inflight = 0
        self.call_s = 0.0
        self.max_inflight = max_inflight
        self.last_error: str | None = None

    async def aclose(self) -> None:
        await self.http.aclose()

    def request_body(self, prompt: str, examples: list[Example]) -> dict:
        req = {"prompt_chars": len(prompt), "prompt_head": prompt[: self.head_chars]}
        if len(prompt) > self.head_chars + self.tail_chars:
            req["prompt_tail"] = prompt[-self.tail_chars:]
        elif len(prompt) > self.head_chars:
            req["prompt_head"] = prompt
        objective, field, instructions = LABELS[self.label]
        state = {
            "objective": objective,
            "request": req,
            "recent_requests": [{"prompt": e.prompt, field: e.thinking_tokens,
                                 "bucket": self.buckets.label(e.bucket)} for e in examples],
        }
        question = {
            "type": "score",
            "instructions": instructions,
            "criteria": [f"{lab} thinking tokens" for lab in self.buckets.labels()],
        }
        return {"model": self.model, "state": state, "questions": {"tier": question}}

    async def post(self, state, questions: dict) -> dict | None:
        if self.max_inflight and self.inflight >= self.max_inflight:
            self.skipped_busy += 1
            return None
        self.calls += 1
        self.inflight += 1
        url = min(self.urls, key=self.url_inflight.__getitem__)
        self.url_inflight[url] += 1
        t0 = time.monotonic()
        try:
            r = await self.http.post(url, json={"model": self.model, "state": state, "questions": questions})
            r.raise_for_status()
            return r.json()["answers"]
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        finally:
            self.call_s += time.monotonic() - t0
            self.url_inflight[url] -= 1
            self.inflight -= 1

    def _probs(self, answers: dict | None) -> np.ndarray | None:
        try:
            probs = answers["tier"]["probabilities"]
            p = np.array([float(probs[str(i)]) for i in range(self.buckets.n)])
            if not np.isfinite(p).all() or p.sum() <= 0:
                raise ValueError("bad probabilities")
            return p / p.sum()
        except Exception as e:
            if answers is not None:
                self.failures += 1
                self.last_error = f"{type(e).__name__}: {e}"
            return None

    async def ask_with(self, prompt: str, examples: list[Example], extra: dict | None = None
                       ) -> tuple[np.ndarray | None, dict]:
        body = self.request_body(prompt, examples)
        questions = {**body["questions"], **(extra or {})}
        answers = await self.post(body["state"], questions)
        return self._probs(answers), {k: v for k, v in (answers or {}).items() if k in (extra or {})}

    async def ask(self, prompt: str, examples: list[Example]) -> np.ndarray | None:
        return (await self.ask_with(prompt, examples))[0]

    async def ask_settled(self, prompt: str, thinking: str, used: int) -> dict | None:
        """Judge a request's thinking so far: {"settled": P(answer fixed), "done": P(same answer if stopped now),
        "level": mean level, "levels": [...], "ms": call time}. None when Jev is busy or fails."""
        state = settled_state(prompt, thinking, used, head_chars=self.head_chars)
        t0 = time.monotonic()
        answers = await self.post(state, settled_questions())
        out = settled_from_answers(answers)
        if out is None:
            if answers is not None:
                self.failures += 1
                self.last_error = "settled: bad answers"
            return None
        out["ms"] = round((time.monotonic() - t0) * 1000, 1)
        return out
