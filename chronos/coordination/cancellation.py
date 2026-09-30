"""Epoch-scoped task registry: bump cancels older work, stale results never get emitted."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from chronos.coordination.epoch import EpochManager
from chronos.trace.logger import ComponentTrace


@dataclass(eq=False)
class _Entry:
    task: asyncio.Task[Any]
    epoch: int
    name: str
    protected: bool
    tid: int  # unique per manager: lets the trace pair task_started with task_finished/cancelled
    t0: float = field(default_factory=time.perf_counter)


class CancellationManager:
    def __init__(self, epochs: EpochManager, trace: ComponentTrace | None = None) -> None:
        self._epochs = epochs
        self._trace = trace
        self._entries: set[_Entry] = set()
        self._seq = 0
        self.dropped_stale = 0
        epochs.add_listener(self._on_bump)

    # -- registration -------------------------------------------------------------------
    def register(self, epoch: int, task: asyncio.Task[Any], name: str = "",
                 protected: bool = False) -> asyncio.Task[Any]:
        """`protected` tasks (in-flight writes) are NOT cancelled on bump: killing a write
        mid-flight leaves the world in an unknown state. Their results still go through
        emit_if_current, and stale committed writes are compensated instead."""
        self._seq += 1
        entry = _Entry(task, epoch, name or task.get_name(), protected, self._seq)
        self._entries.add(entry)
        task.add_done_callback(lambda _t, e=entry: self._finished(e))
        if self._trace:
            self._trace.emit("task_started", epoch=epoch, task=entry.name, task_id=entry.tid,
                             protected=protected)
        return task

    def _finished(self, e: _Entry) -> None:
        self._entries.discard(e)
        if self._trace is None or e.task.cancelled():  # cancelled tasks were traced at cancel time
            return
        exc = e.task.exception()
        self._trace.emit("task_finished", epoch=e.epoch, task=e.name, task_id=e.tid,
                         ok=exc is None, error=type(exc).__name__ if exc else None,
                         duration_ms=round((time.perf_counter() - e.t0) * 1000, 3))

    def spawn(self, epoch: int, coro: Coroutine[Any, Any, Any], name: str = "",
              protected: bool = False) -> asyncio.Task[Any]:
        return self.register(epoch, asyncio.create_task(coro, name=name or None), name, protected)

    def active(self, epoch: int | None = None) -> list[asyncio.Task[Any]]:
        return [e.task for e in self._entries
                if not e.task.done() and (epoch is None or e.epoch == epoch)]

    # -- cancellation -------------------------------------------------------------------
    def _on_bump(self, _old: int, new: int, reason: str) -> None:
        self.cancel_older_than(new, reason)

    def cancel_older_than(self, epoch: int, reason: str = "epoch_bump") -> int:
        n = 0
        for e in list(self._entries):
            if e.epoch >= epoch or e.task.done():
                continue
            if e.protected:
                if self._trace:
                    self._trace.emit("task_protected", epoch=e.epoch, task=e.name,
                                     task_id=e.tid, superseded_by=epoch)
                continue
            e.task.cancel()
            n += 1
            if self._trace:
                self._trace.emit("task_cancelled", epoch=e.epoch, task=e.name, task_id=e.tid,
                                 superseded_by=epoch, reason=reason)
        return n

    async def join_cancelled(self) -> None:
        """Await all tasks that were asked to cancel (they finish with CancelledError)."""
        pending = [e.task for e in self._entries if e.task.cancelling()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # -- stale-result gate --------------------------------------------------------------
    def emit_if_current(self, epoch: int, result: Any,
                        sink: Callable[[Any], Any] | None = None, *, what: str = "") -> bool:
        """The single choke point for outputs. Returns True iff `result` was current (and, when
        a sink is given, was passed to it). Stale results are dropped and traced."""
        if self._epochs.is_current(epoch):
            if sink is not None:
                sink(result)
            return True
        self.dropped_stale += 1
        if self._trace:
            self._trace.emit("stale_result_dropped", epoch=epoch,
                             current_epoch=self._epochs.current(),
                             what=what or type(result).__name__)
        return False
