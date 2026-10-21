import json

import httpx


def fake_vllm(answers: dict[str, list[str]], seen: list[dict], think_len: int = 3):
    counters = {k: 0 for k in answers}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        body = json.loads(req.content)
        h = {k: v for k, v in req.headers.items() if k.startswith("x-lifeline")}
        seen.append({"headers": h, "body": body})
        step, phase = h["x-lifeline-step"], h["x-lifeline-phase"]
        if phase == "think":
            done = body["prompt"].rpartition("<think>\n")[2].count("hmm ")
            left = max(0, think_len - done)
            n = min(left, body["max_tokens"])
            text, fin = "hmm " * n, ("length" if left > body["max_tokens"] else "stop")
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
