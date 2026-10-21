from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .classifier import BudgetClassifier


def create_app(clf: BudgetClassifier, *, state_path: str | Path | None = None, save_every: int = 100) -> Starlette:
    seen = {"n": 0}

    async def predict(request: Request) -> JSONResponse:
        body = await request.json()
        if not isinstance(body.get("request_id"), str) or not isinstance(body.get("prompt"), str):
            return JSONResponse({"error": "request_id and prompt are required strings"}, status_code=400)
        pred = await clf.predict(body["request_id"], body["prompt"], body.get("model"))
        return JSONResponse(pred.as_dict())

    async def observe(request: Request) -> JSONResponse:
        body = await request.json()
        if not isinstance(body.get("request_id"), str) or not isinstance(body.get("thinking_tokens"), int):
            return JSONResponse({"error": "request_id (string) and thinking_tokens (int) are required"}, status_code=400)
        out = clf.observe(body["request_id"], body["thinking_tokens"], finished=bool(body.get("finished", True)),
                          model=body.get("model"), prompt=body.get("prompt"))
        seen["n"] += 1
        if state_path and save_every and seen["n"] % save_every == 0:
            clf.save(state_path)
        return JSONResponse(out)

    async def pressure(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            clf.set_pressure(float(body["p"]))
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "p (number) is required"}, status_code=400)
        return JSONResponse({"pressure": clf.pressure, "shadow": clf.shadow})

    async def stats(request: Request) -> JSONResponse:
        return JSONResponse(clf.stats())

    async def save(request: Request) -> JSONResponse:
        if not state_path:
            return JSONResponse({"error": "server started without a state path"}, status_code=400)
        clf.save(state_path)
        return JSONResponse({"saved": str(state_path)})

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        if state_path:
            clf.save(state_path)
        if clf.jev is not None:
            await clf.jev.aclose()

    return Starlette(routes=[
        Route("/v1/predict", predict, methods=["POST"]),
        Route("/v1/observe", observe, methods=["POST"]),
        Route("/v1/pressure", pressure, methods=["POST"]),
        Route("/v1/stats", stats, methods=["GET"]),
        Route("/v1/save", save, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ], lifespan=lifespan)
