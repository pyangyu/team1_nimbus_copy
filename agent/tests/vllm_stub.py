import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = "test-key"
THINK = "hmm "


class StubModel:
    def __init__(self, problems: dict[str, dict]):
        self.problems = problems

    def _problem(self, prompt: str) -> dict:
        return next(p for q, p in self.problems.items() if q in prompt)

    def think(self, prompt: str, max_tokens: int) -> tuple[str, str]:
        p = self._problem(prompt)
        left = max(0, p["think"] - prompt.rpartition("<think>\n")[2].count(THINK))
        n = min(left, max_tokens)
        return THINK * n, "length" if left > max_tokens else "stop"

    def answer(self, step: str, prompt: str) -> str:
        p = self._problem(prompt)
        if step == "plan":
            return "Use multiplication."
        if step.startswith("verify"):
            m = re.search(r"Proposed final answer: (.+)", prompt)
            return "0.9" if m and m.group(1).strip() == p["gold"] else "0.1"
        rejected = re.search(r"Earlier attempts answered ([^,]+(?:, [^,]+)*), which", prompt)
        tried = set(rejected.group(1).split(", ")) if rejected else set()
        pending = [a for a in p.get("wrong_first", []) if a not in tried]
        ans = pending[0] if pending else p["gold"]
        if prompt.endswith("\\boxed{"):
            return ans + "}"
        return f"The answer is \\boxed{{{ans}}}."


class StubVLLM:
    def __init__(self, problems: dict[str, dict], delay_s: float = 0.0):
        self.model = StubModel(problems)
        self.delay_s = delay_s
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _auth(self) -> bool:
                if self.headers.get("Authorization") == f"Bearer {API_KEY}":
                    return True
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.dumps({"error": "Unauthorized"}).encode()
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return False

            def do_GET(self):
                if not self._auth():
                    return
                body = json.dumps({"data": [{"id": "stub-qwen3"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if not self._auth():
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                h = {k.lower(): v for k, v in self.headers.items() if k.lower().startswith("x-lifeline")}
                with stub.lock:
                    stub.requests.append({"headers": h, "body": body})
                time.sleep(stub.delay_s)
                if h.get("x-lifeline-phase") == "think":
                    text, fin = stub.model.think(body["prompt"], body["max_tokens"])
                    n = text.count(THINK)
                else:
                    text, fin = stub.model.answer(h.get("x-lifeline-step", ""), body["prompt"]), "stop"
                    n = len(text.split())
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [{"choices": [{"text": t, "finish_reason": None}]} for t in re.findall(r"\S+\s*", text)]
                events += [{"choices": [{"text": "", "finish_reason": fin}]},
                           {"choices": [], "usage": {"prompt_tokens": len(body["prompt"].split()),
                                                     "completion_tokens": n}}]
                for e in events:
                    self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()

    def by_workflow(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in self.requests:
            out.setdefault(r["headers"]["x-lifeline-workflow"], []).append(r)
        return out
