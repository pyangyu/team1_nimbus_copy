import asyncio
import json

import httpx
import pytest

from brownout_gateway import Controller, Gateway, PressureMonitor, create_app
from budget_classifier import BudgetClassifier

KEY = "test-key"
TOK = "z "


class FakeVLLM:
    def __init__(self, lengths: dict[str, int], delay: float = 0.001, continuous: bool = True):
        self.lengths, self.delay, self.continuous = lengths, delay, continuous
        self.requests: list[dict] = []
        self.aborted = 0
        self.on_token = None

    def natural(self, prompt: str) -> int:
        return next(n for k, n in self.lengths.items() if k in prompt)

    async def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        body = json.loads(req.content)
        self.requests.append({"headers": dict(req.headers), "body": body, "host": req.url.host})
        if req.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "Unauthorized"})
        if req.headers.get("x-lifeline-phase") == "answer":
            pieces, fin = ["The answer is \\boxed{4}."][: body["max_tokens"]], "stop"
        else:
            done = body["prompt"].rpartition("<think>\n")[2].count(TOK)
            left = max(0, self.natural(body["prompt"]) - done)
            n = min(left, body["max_tokens"])
            pieces, fin = [TOK] * n, ("length" if left > body["max_tokens"] else "stop")
        n = len(pieces)

        async def gen():
            finished = False
            try:
                for i, piece in enumerate(pieces):
                    await asyncio.sleep(self.delay)
                    if self.on_token:
                        self.on_token(i)
                    ev = {"choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                    if self.continuous:
                        ev["usage"] = {"prompt_tokens": 10, "completion_tokens": i + 1}
                    yield f"data: {json.dumps(ev)}\n\n".encode()
                yield f"data: {json.dumps({'choices': [{'index': 0, 'text': '', 'finish_reason': fin}]})}\n\n".encode()
                yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': n}})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                finished = True
            finally:
                if not finished:
                    self.aborted += 1

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())


def make(lengths=None, policy="brownout", notice="silent", jev=None, predictor="jev", upstream="http://vllm/v1", **kw):
    fake = FakeVLLM(lengths or {"easy": 50, "hard": 2000})
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    monitor = PressureMonitor()
    clf = BudgetClassifier(dim=1 << 12, jev=jev)
    ctl = Controller(monitor, policy=policy, notice=notice, predictor=predictor, interval=0.002)
    gw = Gateway(upstream, clf, ctl, monitor, upstream_key=KEY, client_key=KEY, http=http, **kw)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gw, background=False)),
                               base_url="http://gw", headers={"Authorization": f"Bearer {KEY}"})
    return gw, fake, client


def think_headers(step="solve.0", cls="agentic", wf="w1", budget=1000, **extra):
    h = {"X-Lifeline-Phase": "think", "X-Lifeline-Step": step, "X-Lifeline-Class": cls,
         "X-Lifeline-Workflow": wf, "X-Lifeline-Thinking-Budget": str(budget)}
    h.update(extra)
    return h


def body(prompt, max_tokens=1000):
    return {"model": "fake", "prompt": prompt + "<think>\n", "max_tokens": max_tokens, "stream": True,
            "stop": ["</think>"]}


def parse(resp: httpx.Response) -> dict:
    text, finish, info, usage = "", None, None, None
    for line in resp.text.splitlines():
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        ev = json.loads(line[5:])
        for ch in ev.get("choices") or []:
            text += ch.get("text", "")
            finish = ch.get("finish_reason") or finish
        usage = ev.get("usage") or usage
        info = ev.get("lifeline") or info
    return {"text": text, "finish": finish, "info": info, "usage": usage}


@pytest.fixture
def run():
    def _run(coro):
        return asyncio.run(coro)
    return _run
