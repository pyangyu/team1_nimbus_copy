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


class JevPrior:
    def __init__(self, buckets: Buckets, url: str | list[str] = "http://127.0.0.1:8791/v1/systemone", *,
                 api_key: str | None = None, model: str = "open-jev", timeout: float = 2.0,
                 head_chars: int = 1000, tail_chars: int = 300, max_inflight: int = 1,
                 transport: httpx.AsyncBaseTransport | None = None):
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
        state = {
            "objective": OBJECTIVE,
            "request": req,
            "recent_requests": [{"prompt": e.prompt, "thinking_tokens_used": e.thinking_tokens,
                                 "bucket": self.buckets.label(e.bucket)} for e in examples],
        }
        question = {
            "type": "score",
            "instructions": "How many thinking tokens will the model use on this request?",
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
