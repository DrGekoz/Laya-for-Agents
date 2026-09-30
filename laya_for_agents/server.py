"""The HTTP surface: a Jev ``/v1/systemone`` server backed by local Laya.

Deliberately the same route, the same request body and the same reply shape as
``laya-serve`` and TypeSafe Jev, so a client written against either works
unchanged -- ``hermes-jev-skills``, ``hs-jev``, ``typesafe-sdk``, or your own.

    POST /v1/systemone   one state, any number of typed questions
    GET  /health         what is resident, what it runs on, what it has done
    GET  /               a one-page summary for a human

Everything heavy (torch, the checkpoints) is imported lazily, so ``import
laya_for_agents.server`` stays cheap and touches no GPU.
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
from typing import Optional

from starlette.concurrency import run_in_threadpool

from . import protocol
from .engine import Busy, Engine
from .settings import Settings, load as load_settings

# Imported at module scope, not inside create_app, and this is not style.
# `from __future__ import annotations` makes every annotation a string, and
# FastAPI resolves those strings through the function's module globals. A
# function-local import would leave `"Request"` unresolvable, FastAPI would fall
# back to treating it as a query parameter, and every POST here would 422 with
# `{"loc": ["query", "request"], "msg": "Field required"}`. The try/except keeps
# the extras optional: importing this module without fastapi installed still
# works, and only actually serving raises.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:  # the server extras are optional: pip install -e ".[serve]"
    FastAPI = Request = JSONResponse = None  # type: ignore[assignment]

_log = logging.getLogger("laya_for_agents.server")


def create_app(settings: Optional[Settings] = None, engine: Optional[Engine] = None):
    """Build the ASGI app. Pass an ``engine`` to inject a pre-warmed one."""
    if FastAPI is None:
        raise RuntimeError('the server extras are missing; install with: pip install -e ".[serve]"')

    settings = settings or load_settings()
    engine = engine or Engine(settings)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        # The preload runs on a thread and is NOT awaited. Awaiting it means uvicorn
        # never binds the port or answers /health until every checkpoint is built --
        # measured at 92 s warm and 218 s cold on CPU. For that whole window the
        # service looks dead: a watchdog kills and restarts it in a loop, and a cron
        # health check reports it down. Serving immediately and reporting `loading` in
        # /health is the difference between "warming up" and "crashed".
        def _preload() -> None:
            try:
                described = engine.warm()
                _log.info("resident: %s on %s", ", ".join(described.get("loaded") or []) or "none",
                          described.get("device"))
            except Exception as error:  # noqa: BLE001 - serve anyway, load lazily
                _log.warning("warm-up failed, checkpoints will load on first use: %s", error)

        if settings.preload:
            threading.Thread(target=_preload, name="lfa-preload", daemon=True).start()
        yield

    app = FastAPI(
        title="Laya for Agents",
        summary="Local typed decisions over the TypeSafe Jev /v1/systemone wire protocol.",
        version="1.0.0",
        root_path=settings.root_path,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine

    def _authorised(request) -> bool:
        if not settings.api_key:
            return True
        header = request.headers.get("authorization") or ""
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), settings.api_key)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        """Refuse an oversized body before the worker ever sees it."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
            return JSONResponse({"detail": f"body over {settings.max_body_bytes} bytes"},
                                status_code=413)
        body = await request.body()
        if len(body) > settings.max_body_bytes:
            return JSONResponse({"detail": f"body over {settings.max_body_bytes} bytes"},
                                status_code=413)
        return await call_next(request)

    def _error(status: int, detail: str) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=status)

    @app.get("/health")
    async def health():
        """Always open, and cheap: it never touches the inference gate."""
        return JSONResponse(await run_in_threadpool(engine.health))

    @app.get("/")
    async def index():
        described = engine.describe()
        return JSONResponse({
            "service": "Laya for Agents",
            "what_it_is": "A local decision engine serving the TypeSafe Jev /v1/systemone wire protocol.",
            "endpoints": {"decide": "POST /v1/systemone", "health": "GET /health"},
            "question_types": list(protocol.QUESTION_TYPES),
            "loaded": described["loaded"],
            "device": described["device"],
            "auth": "bearer required" if settings.api_key else "open on loopback",
        })

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if not _authorised(request):
            return _error(401, "invalid or missing bearer token")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _error(400, "body is not valid JSON")
        try:
            checked = protocol.check_request(body, settings)
        except protocol.ProtocolError as error:
            return _error(error.status, error.detail)
        try:
            reply = await run_in_threadpool(engine.answer, checked)
        except protocol.ProtocolError as error:
            return _error(error.status, error.detail)
        except Busy as error:
            return _error(503, str(error))
        return JSONResponse(reply)

    return app


app = None  # populated by `laya-for-agents serve`, so importing never builds a Router


def main() -> None:  # pragma: no cover - convenience alias
    from .cli import main as cli_main

    raise SystemExit(cli_main(["serve"]))


__all__ = ["create_app", "app", "main", "Engine", "Settings", "Busy"]
