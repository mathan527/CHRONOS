"""End-of-utterance detection for streaming partial transcripts.

Partials are cumulative hypotheses ("I want to" -> "I want to book" -> ...). Each new partial
resets a silence timer; only when the timer fires (default 700 ms of silence) or a FINAL arrives
is the utterance committed. A HESITATION is never committed: the buffer is held, and the next
partial simply replaces it and restarts the timer.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from chronos.perception.bargein import BargeInContext, tier1
from chronos.protocol import BargeInType

CommitFn = Callable[[str], Awaitable[None]]
HoldFn = Callable[[str], Awaitable[None]]


class EndOfUtteranceDetector:
    def __init__(self, on_commit: CommitFn, *, silence_ms: int = 700,
                 on_hold: HoldFn | None = None,
                 context: Callable[[], BargeInContext | None] | None = None) -> None:
        self._on_commit, self._on_hold = on_commit, on_hold
        self._silence_s = silence_ms / 1000
        self._context = context
        self._buffer = ""
        self._timer: asyncio.Task[None] | None = None
        self._gen = 0  # invalidates timers that were superseded

    @property
    def pending(self) -> str:
        return self._buffer

    @property
    def timer_active(self) -> bool:
        return self._timer is not None and not self._timer.done()

    def reset(self) -> None:
        """Forget any buffered partial (a FINAL transcript supersedes it). Synchronous."""
        self._cancel_timer()
        self._buffer = ""

    def _hesitant(self, text: str, final: bool) -> bool:
        d = tier1(text, final=final, context=self._context() if self._context else None)
        return d is not None and d.label is BargeInType.HESITATION

    def _cancel_timer(self) -> None:
        self._gen += 1
        if self._timer and not self._timer.done() and self._timer is not asyncio.current_task():
            self._timer.cancel()
        self._timer = None

    async def feed_partial(self, text: str) -> None:
        self._cancel_timer()
        self._buffer = text.strip()
        if not self._buffer:
            return
        gen = self._gen
        self._timer = asyncio.create_task(self._wait_then_decide(gen))

    async def feed_final(self, text: str) -> None:
        """A FINAL transcript ends the utterance immediately (no need to wait for silence)."""
        self._cancel_timer()
        self._buffer = ""
        text = text.strip()
        if not text:
            return
        if self._hesitant(text, final=True):
            if self._on_hold:
                await self._on_hold(text)
            return
        await self._on_commit(text)

    async def _wait_then_decide(self, gen: int) -> None:
        await asyncio.sleep(self._silence_s)
        if gen != self._gen:
            return
        text = self._buffer
        if self._hesitant(text, final=False):
            if self._on_hold:  # keep buffering; do NOT commit and do NOT clear
                await self._on_hold(text)
            return
        self._buffer = ""
        self._timer = None
        await self._on_commit(text)

    async def aclose(self) -> None:
        self._cancel_timer()
        self._buffer = ""
