"""Open-Jev HTTP server that scores concurrent requests together.

Open-Jev's own server (jev.server) runs every request under one lock, so a burst of calls is
scored one at a time. This server queues requests, waits up to --window-ms for more, and scores
all of their candidate sequences in shared forward passes. Run it with Open-Jev's venv, which
provides the `jev` package; it uses only Open-Jev's public pieces and the standard library.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from jev.api import compile_request, format_response
from jev.metrics import softmax
from jev.serving import candidate_batches, load_predictor
from jev.server import strict_json

PATHS = ("/v1/inference", "/v1/systemone", "/api/jev")


def candidates(record: dict) -> int:
    return 1 if record["kind"] == "noul" else len(record["options"])


def compile_one(predictor, request) -> list[dict]:
    if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
        raise ValueError("request requires state and questions")
    if request.get("model") not in (None, "open-jev", "jev-latest", predictor.model_name):
        raise ValueError("requested model is not loaded; see /v1/models")
    if not isinstance(request["questions"], dict) or len(request["questions"]) > predictor.max_questions:
        raise ValueError("questions must be an object within the server question limit")
    records = compile_request(request["state"], request["questions"])
    if sum(candidates(r) for r in records) > predictor.max_candidates:
        raise ValueError("request exceeds the candidate limit")
    return records


def predict_many(predictor, requests: list) -> list:
    """One result per request: a response dict, or the exception that request raised."""
    out: list = [None] * len(requests)
    flat, owner = [], []
    for i, req in enumerate(requests):
        try:
            records = compile_one(predictor, req)
        except (ValueError, KeyError, TypeError, UnicodeError) as e:
            out[i] = e
            continue
        flat.extend(records)
        owner.extend([i] * len(records))
    if not flat:
        return out
    start, input_tokens = time.perf_counter(), 0
    logits: list[list[float]] = [[] for _ in flat]
    for batch in candidate_batches(flat, predictor.batch_size):
        rows, token_count = predictor.scorer.score([piece for _, piece in batch])
        if len(rows) != len(batch):
            raise RuntimeError("backend returned the wrong number of rows")
        input_tokens += token_count
        for (index, piece), row in zip(batch, rows):
            expected = 2 if piece["kind"] == "noul" else len(piece["options"])
            if len(row) != expected or any(not math.isfinite(v) for v in row):
                raise RuntimeError("backend returned invalid logits")
            logits[index].extend(row)
    seconds = time.perf_counter() - start
    total, batched = len(flat), len(set(owner))
    for i in sorted(set(owner)):
        idx = [k for k, o in enumerate(owner) if o == i]
        records = [flat[k] for k in idx]
        result = format_response(records, [softmax(logits[k], predictor.temperature) for k in idx])
        result.update(model=predictor.model_name,
                      usage={"input_tokens": round(input_tokens * len(idx) / total), "output_tokens": 0},
                      metadata={"method": predictor.method, "temperature": predictor.temperature,
                                "candidate_sequences": sum(candidates(r) for r in records),
                                "inference_seconds": seconds, "batched_requests": batched,
                                **predictor.provenance})
        out[i] = result
    return out


class Batcher:
    def __init__(self, predictor, *, window_s: float = 0.005, max_requests: int = 16):
        self.predictor, self.window_s, self.max_requests = predictor, window_s, max_requests
        self.q: queue.Queue = queue.Queue()
        self.batches = self.requests = 0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, request) -> Future:
        f: Future = Future()
        self.q.put((request, f))
        return f

    def _run(self) -> None:
        while True:
            items = [self.q.get()]
            deadline = time.monotonic() + self.window_s
            while len(items) < self.max_requests:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    items.append(self.q.get(timeout=left))
                except queue.Empty:
                    break
            self.batches += 1
            self.requests += len(items)
            try:
                results = predict_many(self.predictor, [r for r, _ in items])
            except Exception as e:
                results = [e] * len(items)
            for (_, f), res in zip(items, results):
                if isinstance(res, BaseException):
                    f.set_exception(res)
                else:
                    f.set_result(res)


def make_server(batcher: Batcher, host: str, port: int, max_body_bytes: int = 4 * 1024 * 1024):
    predictor = batcher.predictor

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass

        def send(self, status: int, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if urlsplit(self.path).path == "/health":
                return self.send(200, {"status": "ready", "model": predictor.model_name, "method": predictor.method,
                                       "batches": batcher.batches, "requests": batcher.requests})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if urlsplit(self.path).path not in PATHS:
                return self.send(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "Content-Type must be application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > max_body_bytes:
                    return self.send(413, {"error": "request body size is outside server limits"})
                request = strict_json(self.rfile.read(length))
                return self.send(200, batcher.submit(request).result())
            except (ValueError, KeyError, TypeError, UnicodeError) as e:
                return self.send(422, {"error": str(e)})
            except Exception as e:
                return self.send(500, {"error": "model inference failed", "error_type": type(e).__name__})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32, help="candidate sequences per forward pass")
    ap.add_argument("--window-ms", type=float, default=5.0, help="how long to wait for more requests")
    ap.add_argument("--max-requests", type=int, default=16, help="requests scored together at most")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args(argv)
    predictor = load_predictor(checkpoint=a.checkpoint, device=a.device, batch_size=a.batch_size)
    batcher = Batcher(predictor, window_s=a.window_ms / 1000, max_requests=a.max_requests)
    server = make_server(batcher, a.host, a.port)
    print(json.dumps({"url": f"http://{a.host}:{server.server_port}", "model": predictor.model_name,
                      "window_ms": a.window_ms, "max_requests": a.max_requests}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
