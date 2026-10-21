from __future__ import annotations

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
    def __init__(self, buckets: Buckets, url: str = "http://127.0.0.1:8791/v1/systemone", *,
                 api_key: str | None = None, model: str = "open-jev", timeout: float = 2.0,
                 head_chars: int = 1500, tail_chars: int = 500,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.buckets, self.url, self.model = buckets, url, model
        self.head_chars, self.tail_chars = head_chars, tail_chars
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(timeout=timeout, headers=headers, transport=transport)
        self.calls = self.failures = 0
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

    async def ask(self, prompt: str, examples: list[Example]) -> np.ndarray | None:
        self.calls += 1
        try:
            r = await self.http.post(self.url, json=self.request_body(prompt, examples))
            r.raise_for_status()
            probs = r.json()["answers"]["tier"]["probabilities"]
            p = np.array([float(probs[str(i)]) for i in range(self.buckets.n)])
            if not np.isfinite(p).all() or p.sum() <= 0:
                raise ValueError("bad probabilities")
            return p / p.sum()
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return None
