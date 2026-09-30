"""Unified event queue: INTERRUPT > TRANSCRIPT > CAMERA > TEXT, FIFO within a priority."""
from __future__ import annotations

import asyncio
import itertools
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from chronos.protocol import Event, EventType

PRIORITY: dict[EventType, int] = {
    EventType.INTERRUPT: 0,
    EventType.TRANSCRIPT_FINAL: 1,
    EventType.TRANSCRIPT_PARTIAL: 1,
    EventType.CAMERA_FRAME: 2,
    EventType.TEXT: 3,
}


@dataclass
class QueueMetrics:
    puts: int = 0
    gets: int = 0
    max_depth: int = 0
    put_by_type: Counter = field(default_factory=Counter)


class EventQueue:
    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue[tuple[int, int, Event]] = asyncio.PriorityQueue()
        self._seq = itertools.count()  # FIFO tiebreak within the same priority
        self.metrics = QueueMetrics()

    async def put(self, event: Event) -> None:
        await self._q.put((PRIORITY[event.type], next(self._seq), event))
        m = self.metrics
        m.puts += 1
        m.put_by_type[event.type] += 1
        m.max_depth = max(m.max_depth, self._q.qsize())

    async def get(self) -> Event:
        _, _, event = await self._q.get()
        self.metrics.gets += 1
        return event

    def drain(self) -> list[Event]:
        """Remove and return everything currently queued, in priority order (non-blocking)."""
        out: list[Event] = []
        while True:
            try:
                _, _, ev = self._q.get_nowait()
            except asyncio.QueueEmpty:
                return out
            self.metrics.gets += 1
            out.append(ev)

    def take_where(self, pred: Callable[[Event], bool]) -> list[Event]:
        """Remove and return the queued events matching `pred`, oldest first (arrival order);
        everything else keeps its exact position. Used to restore causal order when a
        lower-priority event (a camera frame) was sent before a higher-priority one."""
        kept: list[tuple[int, int, Event]] = []
        taken: list[tuple[int, int, Event]] = []
        while True:
            try:
                item = self._q.get_nowait()
            except asyncio.QueueEmpty:
                break
            (taken if pred(item[2]) else kept).append(item)
        for item in kept:
            self._q.put_nowait(item)  # same (priority, seq) key: order is preserved
        self.metrics.gets += len(taken)
        return [item[2] for item in sorted(taken, key=lambda i: i[1])]

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()
