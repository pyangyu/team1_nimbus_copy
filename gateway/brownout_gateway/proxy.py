from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from budget_classifier import BudgetClassifier

from .controller import Active, Controller
from .jevq import JevScheduler
from .pressure import PressureMonitor
from .priority import heuristic
from .routing import Replica, Router

DEFAULT_NOTICE = "\n\n<budget_cut left={N}>\n"
NOTICE_TEXTS = {"tag": DEFAULT_NOTICE,
                "countdown": "\n\nI only have about {N} tokens of thinking left, so I must finish my reasoning "
                             "quickly and move to the final answer.\n"}


def sse(ev: dict | str) -> str:
    return f"data: {ev if isinstance(ev, str) else json.dumps(ev)}\n\n"


class Gateway:
    def __init__(self, upstream: str | list[str], clf: BudgetClassifier, controller: Controller,
                 monitor: PressureMonitor, *, monitors: list[PressureMonitor] | None = None,
                 upstream_key: str | None = None, client_key: str | None = None,
                 log_path: str | Path | None = None,
                 reject_waiting: float = 8, defer_timeout: float = 60.0, defer_poll: float = 0.25,
                 admit_rate: float = 0.25, admit_burst: float = 8, clock=time.monotonic,
                 notice_text: str = DEFAULT_NOTICE, serial_timeout: float = 10.0,
                 http: httpx.AsyncClient | None = None):
        urls = [u.strip().rstrip("/") for u in (upstream.split(",") if isinstance(upstream, str) else upstream)
                if u.strip()]
        monitors = monitors or [monitor] + [PressureMonitor() for _ in urls[1:]]
        if len(monitors) != len(urls):
            raise ValueError("one pressure monitor per upstream")
        self.router = Router([Replica(u, m) for u, m in zip(urls, monitors)])
        self.upstream = urls[0]
        self.clf, self.controller, self.monitor = clf, controller, monitors[0]
        controller.monitors = monitors
        self.jevq = JevScheduler(clf, controller, concurrency=clf.jev.max_inflight if clf.jev else 1)
        self.upstream_key, self.client_key = upstream_key, client_key
        self.reject_waiting, self.defer_timeout, self.defer_poll = reject_waiting, defer_timeout, defer_poll
        self.notice_text, self.serial_timeout = notice_text, serial_timeout
        self.admit_rate, self.admit_burst, self.clock = admit_rate, admit_burst, clock
        self.refill()
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0),
                                              limits=httpx.Limits(max_connections=1000))
        self.log = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log = open(log_path, "a", encoding="utf-8")
        self.oracle: dict[str, int] = {}
        self.admitted: OrderedDict[str, float] = OrderedDict()
        self.max_admitted = 100_000
        self.counts = {"requests": 0, "think": 0, "passthrough": 0, "rejected": 0, "deferred": 0,
                       "deferred_s": 0.0, "cut": 0, "announced": 0, "notices": 0, "upstream_errors": 0,
                       "serial_waits": 0, "serial_wait_s": 0.0, "oracle_hits": 0, "oracle_misses": 0}

    def _up_headers(self, h: dict[str, str]) -> dict[str, str]:
        out = {k: v for k, v in h.items() if k.startswith("x-lifeline")}
        if self.upstream_key:
            out["Authorization"] = f"Bearer {self.upstream_key}"
        return out

    def authorized(self, request: Request) -> bool:
        return not self.client_key or request.headers.get("authorization") == f"Bearer {self.client_key}"

    async def _open(self, body: dict, h: dict[str, str], replica: int = 0) -> httpx.Response:
        url = self.router.replicas[replica].url
        req = self.http.build_request("POST", f"{url}/completions", json=body, headers=self._up_headers(h))
        return await self.http.send(req, stream=True)

    def waiting(self) -> float:
        return sum(r.monitor.waiting for r in self.router.replicas)

    def refill(self) -> None:
        self.tokens, self.tokens_at = float(self.admit_burst), self.clock()

    def take_token(self) -> bool:
        now = self.clock()
        self.tokens = min(self.admit_burst, self.tokens + (now - self.tokens_at) * self.admit_rate)
        self.tokens_at = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True

    async def admit(self) -> Response | None:
        pol = self.controller.policy
        if pol == "reject" and not self.take_token():
            self.counts["rejected"] += 1
            return JSONResponse({"error": "rate limited", "admit_rate": self.admit_rate}, status_code=429)
        if pol == "defer" and self.waiting() > self.reject_waiting:
            self.counts["deferred"] += 1
            t0 = time.perf_counter()
            while self.waiting() > self.reject_waiting:
                if time.perf_counter() - t0 > self.defer_timeout:
                    self.counts["deferred_s"] += time.perf_counter() - t0
                    self.counts["rejected"] += 1
                    return JSONResponse({"error": "deferred too long", "waiting": self.waiting()}, status_code=429)
                await asyncio.sleep(self.defer_poll)
            self.counts["deferred_s"] += time.perf_counter() - t0
        return None

    async def completions(self, request: Request) -> Response:
        if not self.authorized(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        body = await request.json()
        h = {k.lower(): v for k, v in request.headers.items()}
        self.counts["requests"] += 1
        phase = h.get("x-lifeline-phase")
        budget = int(h.get("x-lifeline-thinking-budget") or 0)
        continuation = "x-lifeline-budget-drop" in h
        wf = h.get("x-lifeline-workflow")
        new_call = phase is None or (phase == "think" and not continuation) or (phase == "answer" and budget <= 0)
        if new_call and (wf is None or wf not in self.admitted):
            if (rejected := await self.admit()) is not None:
                return rejected
            if wf is not None:
                self.admitted[wf] = time.time()
                while len(self.admitted) > self.max_admitted:
                    self.admitted.popitem(last=False)
        k = self.router.route(f"{wf}:{h.get('x-lifeline-step', '-')}" if wf is not None else None, new_call)
        self.router.replicas[k].requests += 1
        if phase == "think" and not continuation and body.get("stream"):
            return await self.think(body, h, k)
        return await self.passthrough(body, h, k)

    async def passthrough(self, body: dict, h: dict[str, str], replica: int = 0) -> Response:
        self.counts["passthrough"] += 1
        r = await self._open(body, h, replica)
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code,
                                 media_type=r.headers.get("content-type"), background=BackgroundTask(r.aclose))

    async def think(self, body: dict, h: dict[str, str], replica: int = 0) -> Response:
        self.counts["think"] += 1
        self.router.replicas[replica].think += 1
        t0 = time.time()
        wf, step = h.get("x-lifeline-workflow", "-"), h.get("x-lifeline-step", "-")
        cls = h.get("x-lifeline-class", "agentic")
        dl = h.get("x-lifeline-deadline")
        dl = int(dl) if dl and dl.isdigit() else None
        prompt = body["prompt"]
        requested = int(body.get("max_tokens") or h.get("x-lifeline-thinking-budget") or 4096)
        rid = f"{wf}:{step}:{uuid.uuid4().hex[:8]}"
        pred = self.clf.predict_online(rid, prompt, body.get("model"))
        key = hashlib.sha256(prompt.encode()).hexdigest()
        prio = heuristic(cls, step, dl)
        mon = self.router.replicas[replica].monitor
        if self.controller.predictor == "oracle":
            known = self.oracle.get(key)
            self.counts["oracle_hits" if known is not None else "oracle_misses"] += 1
            k = self.clf.buckets.index(known) if known is not None else None
            a = Active(rid, requested, requested if k is None else self.clf.buckets.budget(k), prio, requested,
                       tier="unknown" if k is None else self.clf.buckets.label(k), replica=replica)
        elif self.controller.predictor == "online" or self.clf.jev is None:
            a = Active(rid, requested, pred.budget, prio, requested, tier=pred.label, replica=replica)
        else:
            a = Active(rid, requested, requested, prio, requested, tier="pending", replica=replica)
        self.controller.start(a)
        answered = self.jevq.submit(a, cls, step, dl)
        if self.controller.predictor == "jev-serial" and answered is not None:
            t1 = time.perf_counter()
            try:
                await asyncio.wait_for(asyncio.shield(answered), self.serial_timeout)
            except asyncio.TimeoutError:
                pass
            self.counts["serial_waits"] += 1
            self.counts["serial_wait_s"] += time.perf_counter() - t1
            self.controller.start(a)
        first = a.limit
        up = {**body, "max_tokens": a.limit, "stream": True,
              "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        r = await self._open(up, h, replica)
        if r.status_code >= 400:
            content = await r.aread()
            await r.aclose()
            self.counts["upstream_errors"] += 1
            self.controller.finish(a)
            self.clf.pending.pop(rid, None)
            return Response(content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        meta = {"rid": rid, "workflow": wf, "step": step, "class": cls, "model": body.get("model"),
                "requested": requested, "online_tier": pred.label, "online_budget": pred.budget, "source": pred.source,
                "first_limit": first, "p_start": round(mon.p, 3), "replica": replica, "t0": t0, "prompt": prompt,
                "key": key}
        headers = {"X-Lifeline-Tier": a.tier, "X-Lifeline-Priority": a.priority,
                   "X-Lifeline-Budget": str(first)}
        return StreamingResponse(self._stream(a, up, h, r, meta), media_type="text/event-stream", headers=headers)

    async def _stream(self, a: Active, up: dict, h: dict[str, str], r: httpx.Response, meta: dict):
        parts: list[str] = []
        done_before, seg, prompt_tokens = 0, 0, None
        finish, cut, notices = None, False, 0
        cap, last_left = a.limit, 0
        try:
            while True:
                action, left = None, 0
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    u = ev.get("usage")
                    text = ""
                    for ch in ev.get("choices") or []:
                        if ch.get("text"):
                            text += ch["text"]
                            if not u:
                                seg += 1
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                    if u:
                        seg = u.get("completion_tokens", seg)
                        if prompt_tokens is None:
                            prompt_tokens = u.get("prompt_tokens")
                    a.used = done_before + seg
                    if text:
                        parts.append(text)
                        yield sse({"object": "text_completion", "model": up.get("model"),
                                   "choices": [{"index": 0, "text": text, "finish_reason": None}]})
                    if a.limit < cap and finish is None:
                        if (self.controller.notice == "announced" and a.used < a.limit
                                and (not notices or cap - a.limit >= self.controller.renotice(last_left))):
                            action, left = "announce", a.limit - a.used
                            break
                        if a.used >= a.limit:
                            action = "cut"
                            break
                await r.aclose()
                if action is None:
                    break
                if action == "cut":
                    cut, finish = True, "length"
                    self.counts["cut"] += 1
                    break
                notices += 1
                self.counts["notices"] += 1
                if notices == 1:
                    self.counts["announced"] += 1
                notice = self.notice_text.format(N=left)
                parts.append(notice)
                yield sse({"object": "text_completion", "model": up.get("model"),
                           "choices": [{"index": 0, "text": notice, "finish_reason": None}]})
                done_before, seg, cap, last_left, finish = a.used, 0, a.limit, left, None
                r = await self._open({**up, "prompt": up["prompt"] + "".join(parts), "max_tokens": left}, h, a.replica)
                if r.status_code >= 400:
                    await r.aread()
                    self.counts["upstream_errors"] += 1
                    cut, finish = True, "length"
                    break
            finish = finish or "length"
            announced = notices > 0
            info = {"tier": a.tier, "online_tier": meta["online_tier"], "jev_tier": a.jev_tier, "jev_at": a.jev_at,
                    "predictor": self.controller.predictor, "priority": a.priority,
                    "priority_source": a.priority_source,
                    "budget": meta["first_limit"], "limit": a.limit, "cut": cut or announced,
                    "cut_from": a.cut_from, "announced": announced, "notices": notices,
                    "policy": self.controller.policy, "replica": a.replica}
            yield sse({"object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": finish}]})
            yield sse({"object": "text_completion", "choices": [],
                       "usage": {"prompt_tokens": prompt_tokens or 0, "completion_tokens": a.used,
                                 "total_tokens": (prompt_tokens or 0) + a.used},
                       "lifeline": info})
            yield sse("[DONE]")
        finally:
            await r.aclose()
            self.controller.finish(a)
            natural = finish == "stop" and not cut and not notices
            if natural:
                self.oracle[meta["key"]] = a.used
            obs = self.clf.observe(a.rid, a.used, finished=natural, model=meta["model"])
            if self.log is not None:
                row = {**{k: v for k, v in meta.items() if k not in ("t0", "key")}, "ts": meta["t0"],
                       "latency_s": round(time.time() - meta["t0"], 3), "used": a.used, "finish": finish,
                       "natural": natural, "cut": cut, "announced": notices > 0, "notices": notices,
                       "limit": a.limit,
                       "cut_from": a.cut_from, "priority": a.priority, "priority_source": a.priority_source,
                       "p_end": round(self.controller.p(a), 3), "policy": self.controller.policy,
                       "notice": self.controller.notice, "observed": obs.get("label"),
                       "tier": a.tier, "tier_budget": a.tier_budget, "jev_tier": a.jev_tier, "jev_at": a.jev_at,
                       "predictor": self.controller.predictor}
                self.log.write(json.dumps(row) + "\n")
                self.log.flush()

    async def models(self, request: Request) -> Response:
        if not self.authorized(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        r = await self.http.get(f"{self.upstream}/models", headers=self._up_headers({}))
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    async def admin_policy(self, request: Request) -> Response:
        if not self.authorized(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        body = await request.json()
        try:
            self.controller.set_policy(body.get("policy", self.controller.policy), body.get("notice"),
                                       body.get("predictor"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if "reject_waiting" in body:
            self.reject_waiting = float(body["reject_waiting"])
        if "admit_rate" in body:
            self.admit_rate = float(body["admit_rate"])
        if "admit_burst" in body:
            self.admit_burst = float(body["admit_burst"])
        if body.get("snapshot"):
            self.clf.snapshot()
        if body.get("reset") == "warm":
            self.clf.restore()
        elif body.get("reset"):
            self.clf.reset()
        self.refill()
        self.admitted.clear()
        return JSONResponse(self.controller.stats())

    def stats(self) -> dict:
        return {"gateway": {**self.counts, "deferred_s": round(self.counts["deferred_s"], 2),
                            "serial_wait_s": round(self.counts["serial_wait_s"], 3),
                            "oracle_prompts": len(self.oracle),
                            "reject_waiting": self.reject_waiting, "admit_rate": self.admit_rate,
                            "admit_burst": self.admit_burst},
                "controller": self.controller.stats(), "pressure": self.router.pressure_stats(),
                "replicas": self.router.stats(),
                "jev": self.jevq.stats(), "classifier": self.clf.stats()}


def create_app(gw: Gateway, *, state_path: str | Path | None = None, background: bool = True) -> Starlette:
    async def stats(request: Request) -> Response:
        return JSONResponse(gw.stats())

    async def pressure(request: Request) -> Response:
        try:
            since = float(request.query_params.get("since", 0))
        except ValueError:
            return JSONResponse({"error": "since must be a unix time"}, status_code=400)
        return JSONResponse({"interval": gw.monitor.interval, "columns": ["t", "p", "waiting", "running", "tpot_ms"],
                             "samples": gw.monitor.since(since),
                             "replicas": [{"url": r.url, "samples": r.monitor.since(since)}
                                          for r in gw.router.replicas]})

    async def health(request: Request) -> Response:
        return JSONResponse({"ok": True, "policy": gw.controller.policy})

    @asynccontextmanager
    async def lifespan(app: Starlette):
        tasks = []
        if background:
            tasks = [asyncio.create_task(r.monitor.run()) for r in gw.router.replicas]
            tasks.append(asyncio.create_task(gw.controller.run()))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            if state_path:
                gw.clf.save(state_path)
            if gw.clf.jev is not None:
                await gw.clf.jev.aclose()
            await gw.http.aclose()
            if gw.log is not None:
                gw.log.close()

    return Starlette(routes=[
        Route("/v1/completions", gw.completions, methods=["POST"]),
        Route("/v1/models", gw.models, methods=["GET"]),
        Route("/v1/stats", stats, methods=["GET"]),
        Route("/v1/pressure", pressure, methods=["GET"]),
        Route("/admin/policy", gw.admin_policy, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ], lifespan=lifespan)
