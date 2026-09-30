"""Per-session monotonic epoch. Every interruption that changes intent creates a new one."""
from __future__ import annotations

from collections.abc import Callable

from chronos.trace.logger import ComponentTrace

BumpListener = Callable[[int, int, str], None]  # (old_epoch, new_epoch, reason)


class EpochManager:
    def __init__(self, session_id: str, trace: ComponentTrace | None = None,
                 start: int = 0) -> None:
        self.session_id = session_id
        self._epoch = start
        self._trace = trace
        self._listeners: list[BumpListener] = []

    def current(self) -> int:
        return self._epoch

    def is_current(self, epoch: int) -> bool:
        return epoch == self._epoch

    def add_listener(self, fn: BumpListener) -> None:
        self._listeners.append(fn)

    def bump(self, reason: str) -> int:
        """Synchronous (no await between increment and listeners), so no task can observe a
        half-bumped state on the single event loop."""
        old, self._epoch = self._epoch, self._epoch + 1
        if self._trace:
            self._trace.emit("epoch_bumped", epoch=self._epoch, old_epoch=old, reason=reason)
        for fn in self._listeners:
            fn(old, self._epoch, reason)
        return self._epoch
