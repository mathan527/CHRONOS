"""Immutable per-epoch state. A correction patches the previous snapshot instead of replanning."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from chronos.coordination.canonical import call_key


def freeze(obj: Any) -> Any:
    """Deep-copy into read-only structures (dict -> MappingProxyType, list -> tuple)."""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(v) for v in obj)
    return obj


def thaw(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [thaw(v) for v in obj]
    return obj


@dataclass(frozen=True)
class ReadEntry:
    """A cached speculative read. `deps` records the slot values it was computed from, so we
    can tell whether a later slot change invalidates it."""
    tool: str
    args: Mapping[str, Any]
    result: Mapping[str, Any]
    deps: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("args", "result", "deps"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @property
    def key(self) -> str:
        return call_key(self.tool, self.args)

    def valid_for(self, slots: Mapping[str, Any]) -> bool:
        return all(slots.get(k) == v for k, v in self.deps.items())


@dataclass(frozen=True)
class StateSnapshot:
    epoch: int
    intent: str | None = None
    slots: Mapping[str, Any] = field(default_factory=dict)
    plan: Mapping[str, Any] = field(default_factory=dict)
    read_cache: Mapping[str, ReadEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", freeze(self.slots))
        object.__setattr__(self, "plan", freeze(self.plan))
        object.__setattr__(self, "read_cache", MappingProxyType(dict(self.read_cache)))

    def patch(self, changes: Mapping[str, Any], *, epoch: int | None = None) -> StateSnapshot:
        """Return a new snapshot. Recognised keys in `changes`:
          intent      -> replaces intent
          slots       -> merged into slots; a value of None removes that slot
          plan        -> replaces plan
          cache_add   -> iterable of ReadEntry to add
        Cache entries whose recorded slot deps no longer hold are dropped; the rest are reused."""
        slots = thaw(self.slots)
        for k, v in (changes.get("slots") or {}).items():
            if v is None:
                slots.pop(k, None)
            else:
                slots[k] = v
        cache = {k: e for k, e in self.read_cache.items() if e.valid_for(slots)}
        for e in changes.get("cache_add") or ():
            cache[e.key] = e
        return StateSnapshot(
            epoch=self.epoch if epoch is None else epoch,
            intent=changes.get("intent", self.intent),
            slots=slots,
            plan=changes.get("plan", self.plan),
            read_cache=cache,
        )

    def cached(self, tool: str, args: Mapping[str, Any]) -> ReadEntry | None:
        return self.read_cache.get(call_key(tool, args))


class SnapshotStore:
    """History of snapshots for one session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._history: list[StateSnapshot] = [StateSnapshot(epoch=0)]

    def current(self) -> StateSnapshot:
        return self._history[-1]

    def apply(self, changes: Mapping[str, Any], *, epoch: int | None = None) -> StateSnapshot:
        snap = self.current().patch(changes, epoch=epoch)
        self._history.append(snap)
        return snap

    def history(self) -> tuple[StateSnapshot, ...]:
        return tuple(self._history)

    def at_epoch(self, epoch: int) -> StateSnapshot | None:
        """Last snapshot recorded in `epoch`."""
        for s in reversed(self._history):
            if s.epoch == epoch:
                return s
        return None
