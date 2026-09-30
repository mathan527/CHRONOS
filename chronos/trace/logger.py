"""Structured JSONL tracing (structlog) -> traces/<session_id>.jsonl."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import structlog

# Canonical trace event names.
TRACE_EVENTS = frozenset({
    "event_received", "classified", "ack_sent", "epoch_bumped", "task_started",
    "task_cancelled", "stale_result_dropped", "tool_read", "write_blocked_duplicate",
    "write_committed", "response_sent", "write_blocked", "write_compensated",
    "task_finished", "intent_extracted", "plan_drafted", "plan_patched", "plan_committed", "plan_cancelled",
    "read_reused", "read_carried_over", "frame_described", "diagnosis_ready",
    "utterance", "frame_default_question", "frame_question",
})


class TraceLogger:
    """One per session. Lines are flushed immediately so a crash loses nothing."""

    def __init__(self, session_id: str, trace_dir: str | Path = "traces") -> None:
        self.session_id = session_id
        self.path = Path(trace_dir) / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self._subs: list[Callable[[dict[str, Any]], None]] = []
        self._fh: TextIO = open(self.path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - held open for the logger lifetime
        self._log = structlog.wrap_logger(
            structlog.WriteLogger(self._fh),
            processors=[structlog.processors.JSONRenderer(sort_keys=True, default=str)],
        )

    def emit(self, event: str, *, component: str, epoch: int, **fields: Any) -> None:
        now = time.perf_counter()
        base = {"session_id": self.session_id, "component": component, "epoch": epoch,
                "t_ms": round((now - self._t0) * 1000, 3), "mono_ms": round(now * 1000, 3)}
        self._log.info(event, **base, **fields)
        for cb in list(self._subs):  # live listeners (the demo UI's ?trace=1 stream)
            try:
                cb({"event": event, **base, **fields})
            except Exception:  # noqa: BLE001, S110 - a broken listener must never break tracing
                pass

    def subscribe(self, cb: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def bind(self, component: str) -> ComponentTrace:
        return ComponentTrace(self, component)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class ComponentTrace:
    """A TraceLogger view that pre-fills `component`."""

    def __init__(self, tracer: TraceLogger, component: str) -> None:
        self._tracer, self.component = tracer, component

    def emit(self, event: str, *, epoch: int, **fields: Any) -> None:
        self._tracer.emit(event, component=self.component, epoch=epoch, **fields)
