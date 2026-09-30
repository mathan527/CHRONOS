"""FastAPI surface: WebSocket + HTTP fallback + traces + health."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import ValidationError

from chronos.agent.session import AgentSession
from chronos.config import Settings
from chronos.protocol import Event, OutputMessage, OutputStatus, OutputType

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # also keeps ids safe as file names
VISUALIZER = Path(__file__).resolve().parent.parent / "trace" / "visualizer.html"
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_RE = re.compile(r"^[A-Za-z0-9_-]+\.(css|js|svg|png|ico)$")
ASSET_TYPES = {"css": "text/css; charset=utf-8", "js": "text/javascript; charset=utf-8",
               "svg": "image/svg+xml", "png": "image/png", "ico": "image/x-icon"}
SAMPLE_RE = re.compile(r"^[A-Za-z0-9_-]+\.png$")  # no path separators: cannot escape the folder

SessionFactory = Callable[[str], Awaitable[AgentSession]]


class SessionManager:
    """Owns the live sessions of one app instance (no module-level state)."""

    def __init__(self, factory: SessionFactory) -> None:
        self._factory = factory
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str, *, create: bool = True) -> AgentSession | None:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is None and create:
                s = await self._factory(session_id)
                s.start()
                self._sessions[session_id] = s
            return s

    @property
    def count(self) -> int:
        return len(self._sessions)

    async def close_all(self) -> None:
        async with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        await asyncio.gather(*(s.aclose() for s in sessions), return_exceptions=True)


def _error(session_id: str, epoch: int, text: str, **data: Any) -> OutputMessage:
    return OutputMessage(type=OutputType.ERROR, session_id=session_id, epoch=epoch,
                         status=OutputStatus.FAILED, text=text, data=data)


async def ingest(session: AgentSession, raw: str | dict[str, Any]) -> tuple[Event | None, OutputMessage | None]:
    """Validate client JSON into an Event and submit it. Never raises on bad input: returns a
    protocol-compliant ERROR message instead. Server-owned fields (epoch, receipt time) are
    overwritten: a client cannot forge the epoch, and its clock is not ours."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(data, dict):
            raise TypeError("event must be a JSON object")
        data.setdefault("session_id", session.session_id)
        data["epoch"] = 0
        data["ts_monotonic"] = time.perf_counter()
        event = Event.model_validate(data)
    except (ValueError, TypeError, ValidationError) as e:
        detail = e.errors(include_url=False, include_context=False, include_input=False) \
            if isinstance(e, ValidationError) else str(e)
        return None, _error(session.session_id, session.epochs.current(), "Invalid event.",
                            detail=detail)
    if event.session_id != session.session_id:
        return None, _error(session.session_id, session.epochs.current(),
                            "session_id does not match this connection.")
    return await session.submit(event), None


async def probe_ollama(settings: Settings,
                       transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    wanted = [settings.llm_model, settings.vision_model]
    info: dict[str, Any] = {"url": settings.ollama_url, "reachable": False,
                            "models": dict.fromkeys(wanted, False)}
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=1.0,
                                     transport=transport) as c:
            r = await c.get("/api/tags")
            r.raise_for_status()
            names = [str(m["name"]) for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return info
    info["reachable"] = True
    for w in wanted:
        info["models"][w] = any(
            n == w or n == f"{w}:latest" or (":" not in w and n.split(":")[0] == w)
            for n in names)
    return info


def create_app(settings: Settings | None = None, *, session_factory: SessionFactory | None = None,
               ollama_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = session_factory or (lambda sid: AgentSession.create(sid, settings))
    manager = SessionManager(factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.close_all()

    app = FastAPI(title="CHRONOS", version="0.1.0", lifespan=lifespan)
    app.state.manager, app.state.settings = manager, settings

    def check_id(session_id: str) -> str:
        if not SESSION_ID_RE.match(session_id):
            raise HTTPException(400, "session_id must match [A-Za-z0-9_-]{1,64}")
        return session_id

    # ------------------------------------------------------------------ websocket ----------
    @app.websocket("/ws/{session_id}")
    async def ws_endpoint(ws: WebSocket, session_id: str) -> None:
        if not SESSION_ID_RE.match(session_id):
            await ws.close(code=1008)
            return
        await ws.accept()
        session = await manager.get(session_id)
        assert session is not None
        out: asyncio.Queue[str] = asyncio.Queue()
        unsubscribe = session.subscribe(lambda m: out.put_nowait(m.model_dump_json()))
        unsubscribe_trace = None
        if ws.query_params.get("trace") in ("1", "true"):  # opt-in: trace rows as {"kind":"trace"}
            unsubscribe_trace = session.subscribe_trace(
                lambda row: out.put_nowait(json.dumps({"kind": "trace", "row": row}, default=str)))

        async def sender() -> None:
            while True:
                await ws.send_text(await out.get())

        send_task = asyncio.create_task(sender())
        try:
            while True:
                _, err = await ingest(session, await ws.receive_text())
                if err is not None:
                    out.put_nowait(err.model_dump_json())
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            if unsubscribe_trace is not None:
                unsubscribe_trace()
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)

    # ------------------------------------------------------------------ HTTP fallback ------
    @app.post("/sessions/{session_id}/events", status_code=202)
    async def post_event(session_id: str, body: Annotated[dict[str, Any], Body()],
                         wait: Annotated[bool, Query()] = False,
                         timeout: Annotated[float, Query(le=60.0)] = 15.0) -> JSONResponse:
        session = await manager.get(check_id(session_id))
        assert session is not None
        first = len(session.outputs)
        event, err = await ingest(session, body)
        if err is not None:
            return JSONResponse(err.model_dump(mode="json"), status_code=422)
        assert event is not None
        result: dict[str, Any] = {"accepted": True, "event_id": event.event_id,
                                  "epoch": event.epoch}
        if wait:
            try:
                await session.wait_idle(timeout)
                result["idle"] = True
            except TimeoutError:
                result["idle"] = False
            result["outputs"] = [m.model_dump(mode="json") for m in session.outputs[first:]]
        return JSONResponse(result, status_code=202)

    @app.get("/sessions/{session_id}/state")
    async def get_state(session_id: str) -> dict[str, Any]:
        session = await manager.get(check_id(session_id), create=False)
        if session is None:
            raise HTTPException(404, "no such session")
        return await session.describe()

    # -------------------------------------------------------------------- traces ----------
    def trace_path(session_id: str) -> Path:
        return Path(settings.trace_dir) / f"{check_id(session_id)}.jsonl"

    @app.get("/traces/{session_id}")
    async def get_trace(session_id: str) -> Response:
        path = trace_path(session_id)
        if not path.exists():
            raise HTTPException(404, "no trace for that session")
        return Response(await asyncio.to_thread(path.read_bytes),
                        media_type="application/x-ndjson")

    @app.get("/traces/{session_id}/view")
    async def view_trace(session_id: str) -> HTMLResponse:
        check_id(session_id)
        if not trace_path(session_id).exists():
            raise HTTPException(404, "no trace for that session")
        return HTMLResponse(await asyncio.to_thread(VISUALIZER.read_text, "utf-8"))

    # -------------------------------------------------------------------- health ----------
    # ----------------------------------------------------------------- demo client ----------
    # Locations can be overridden (Docker copies these folders elsewhere).
    web_index = Path(os.getenv("CHRONOS_WEB_DIR") or REPO_ROOT / "web") / "index.html"
    samples_dir = Path(os.getenv("CHRONOS_SAMPLES_DIR") or REPO_ROOT / "scenarios" / "images")

    web_dir = web_index.parent
    results_file = Path(os.getenv("CHRONOS_BENCH_DIR") or REPO_ROOT / "bench") / "results.json"

    def page(route: str, filename: str) -> None:
        async def handler() -> HTMLResponse:
            path = web_dir / filename
            if not path.is_file():
                raise HTTPException(404, f"web/{filename} not found")
            return HTMLResponse(await asyncio.to_thread(path.read_text, "utf-8"))
        app.add_api_route(route, handler, methods=["GET"], include_in_schema=False,
                          name=f"page_{filename}")

    page("/", "home.html")
    page("/demo", "index.html")
    page("/how-it-works", "how.html")
    page("/results", "results.html")
    page("/timeline", "timeline.html")

    @app.get("/api/results", include_in_schema=False)
    async def bench_results() -> Response:
        """The benchmark harness output, as written by `python -m bench.run`."""
        if not results_file.is_file():
            raise HTTPException(404, "no benchmark results; run: python -m bench.run")
        return Response(await asyncio.to_thread(results_file.read_bytes),
                        media_type="application/json", headers={"Cache-Control": "no-cache"})

    @app.get("/api/traces", include_in_schema=False)
    async def list_traces() -> list[dict[str, Any]]:
        """Sessions that have a trace file, newest first (feeds the Timeline page picker)."""
        def scan() -> list[dict[str, Any]]:
            rows = []
            for f in Path(settings.trace_dir).glob("*.jsonl"):
                if not SESSION_ID_RE.match(f.stem):
                    continue
                st = f.stat()
                with f.open("rb") as fh:
                    events = sum(1 for _ in fh)
                rows.append({"session_id": f.stem, "events": events, "modified": st.st_mtime})
            return sorted(rows, key=lambda r: r["modified"], reverse=True)
        return await asyncio.to_thread(scan)

    @app.get("/assets/{name}", include_in_schema=False)
    async def asset(name: str) -> Response:
        """The demo client's own CSS/JS/icons (a fixed whitelist of names and types)."""
        m = ASSET_RE.match(name)
        path = web_index.parent / name
        if not m or not path.is_file():
            raise HTTPException(404, "no such asset")
        return Response(await asyncio.to_thread(path.read_bytes),
                        media_type=ASSET_TYPES[m.group(1)],
                        headers={"Cache-Control": "no-cache"})

    @app.get("/samples/{name}", include_in_schema=False)
    async def sample_image(name: str) -> Response:
        if not SAMPLE_RE.match(name):
            raise HTTPException(400, "sample names look like panel_disconnected_cable.png")
        path = samples_dir / name
        if not path.is_file():
            raise HTTPException(404, "no such sample image")
        return Response(await asyncio.to_thread(path.read_bytes), media_type="image/png")

    @app.get("/health")
    async def health(_request: Request) -> dict[str, Any]:
        ollama = await probe_ollama(settings, ollama_transport)
        needs_ollama = settings.llm_mode == "ollama"
        ready = ollama["reachable"] and all(ollama["models"].values())
        return {"status": "degraded" if needs_ollama and not ready else "ok",
                "llm_mode": settings.llm_mode, "ollama": ollama,
                "sessions": manager.count}

    return app
