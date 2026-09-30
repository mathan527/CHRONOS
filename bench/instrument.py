"""Measurement instruments shared by both agents.

`RecordingWorld` is the ground truth: it logs every call that actually reaches the world
(timestamps, arguments, result, whether it committed), whichever agent made it, so the
benchmark never has to trust an agent's own bookkeeping.

`WriteTap` records every write the agent *tried* to dispatch and how that attempt ended, which
is what lets us say "this stale write was stopped before it reached the world".
"""
from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any

from chronos.protocol import ToolResult
from chronos.tools.world import World

PRIMARY_WRITES = ("book_flight", "reserve_table", "set_navigation")  # what the user asked for
COMPENSATION_ID_ARG = {"cancel_booking": "booking_id", "cancel_reservation": "reservation_id",
                       "clear_navigation": "nav_id"}
RESULT_ID_KEY = {"book_flight": "booking_id", "reserve_table": "reservation_id",
                 "set_navigation": "nav_id"}

_NESTED: contextvars.ContextVar[bool] = contextvars.ContextVar("bench_world_nested", default=False)


@dataclass
class WorldCall:
    tool: str
    args: dict[str, Any]
    kind: str  # "read" | "write"
    t_enter: float
    t_done: float | None = None
    committed: bool = False  # a write that returned successfully
    result: dict[str, Any] | None = None
    error: str | None = None


class RecordingWorld(World):
    """A World that logs every top-level call. Calls the world makes to itself (e.g.
    reserve_table checking availability) are not logged, so counts mean 'tool calls'."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[WorldCall] = []

    async def _rec(self, kind: str, tool: str, args: dict[str, Any], fn: Any, *a: Any) -> Any:
        if _NESTED.get():
            return await fn(*a)
        call = WorldCall(tool, args, kind, time.perf_counter())
        self.calls.append(call)
        token = _NESTED.set(True)
        try:
            res = await fn(*a)
        except BaseException as e:
            call.error, call.t_done = type(e).__name__, time.perf_counter()
            raise
        finally:
            _NESTED.reset(token)
        call.t_done = time.perf_counter()
        call.result = res if isinstance(res, dict) else {"rows": len(res)}
        call.committed = kind == "write"
        return res

    # ---- writes ---------------------------------------------------------------------------
    async def book_flight(self, origin, dest, on, time, passenger, seat_pref):  # type: ignore[override]
        return await self._rec(
            "write", "book_flight", {"origin": origin, "dest": dest, "date": on, "time": time,
                                     "passenger": passenger, "seat_pref": seat_pref},
            super().book_flight, origin, dest, on, time, passenger, seat_pref)

    async def cancel_booking(self, booking_id):  # type: ignore[override]
        return await self._rec("write", "cancel_booking", {"booking_id": booking_id},
                               super().cancel_booking, booking_id)

    async def set_navigation(self, destination, via, origin):  # type: ignore[override]
        return await self._rec("write", "set_navigation",
                               {"destination": destination, "via": via, "origin": origin},
                               super().set_navigation, destination, via, origin)

    async def clear_navigation(self, nav_id):  # type: ignore[override]
        return await self._rec("write", "clear_navigation", {"nav_id": nav_id},
                               super().clear_navigation, nav_id)

    async def reserve_table(self, party_size, on, time, restaurant, name):  # type: ignore[override]
        return await self._rec(
            "write", "reserve_table", {"party_size": party_size, "date": on, "time": time,
                                       "restaurant": restaurant, "name": name},
            super().reserve_table, party_size, on, time, restaurant, name)

    async def cancel_reservation(self, reservation_id):  # type: ignore[override]
        return await self._rec("write", "cancel_reservation", {"reservation_id": reservation_id},
                               super().cancel_reservation, reservation_id)

    # ---- reads ----------------------------------------------------------------------------
    async def search_flights(self, origin, dest, on, time, limit=10):  # type: ignore[override]
        return await self._rec("read", "search_flights",
                               {"origin": origin, "dest": dest, "date": on, "time": time},
                               super().search_flights, origin, dest, on, time, limit)

    async def get_route(self, origin, destination, via):  # type: ignore[override]
        return await self._rec("read", "get_route",
                               {"origin": origin, "destination": destination, "via": via},
                               super().get_route, origin, destination, via)

    async def table_availability(self, party_size, on, time, restaurant):  # type: ignore[override]
        return await self._rec("read", "check_table_availability",
                               {"party_size": party_size, "date": on, "time": time,
                                "restaurant": restaurant},
                               super().table_availability, party_size, on, time, restaurant)

    async def get_booking(self, booking_id):  # type: ignore[override]
        return await self._rec("read", "get_booking", {"booking_id": booking_id},
                               super().get_booking, booking_id)

    # ---- ground-truth views ---------------------------------------------------------------
    def committed(self, kind: str | None = None) -> list[WorldCall]:
        return [c for c in self.calls if c.committed and (kind is None or c.kind == kind)]

    def primary_writes(self) -> list[WorldCall]:
        return [c for c in self.calls if c.committed and c.tool in PRIMARY_WRITES]

    def compensated_ids(self) -> set[str]:
        """Ids of entities that an explicit compensating call undid."""
        return {str(c.args[COMPENSATION_ID_ARG[c.tool]]) for c in self.calls
                if c.committed and c.tool in COMPENSATION_ID_ARG}

    def last_activity(self) -> float | None:
        stamps = [c.t_done for c in self.calls if c.t_done is not None]
        return max(stamps) if stamps else None


# ------------------------------------------------------------------------------ write tap ------
@dataclass
class Attempt:
    tool: str
    args: dict[str, Any]
    t_start: float
    t_end: float | None = None
    outcome: str = "pending"  # committed | committed_then_compensated | prevented | cancelled | failed


@dataclass
class WriteTap:
    attempts: list[Attempt] = field(default_factory=list)

    def begin(self, tool: str, args: dict[str, Any]) -> Attempt:
        a = Attempt(tool, dict(args), time.perf_counter())
        self.attempts.append(a)
        return a

    @staticmethod
    def end(a: Attempt, outcome: str) -> None:
        a.outcome, a.t_end = outcome, time.perf_counter()


def classify_result(res: ToolResult) -> str:
    """How a CHRONOS execute_write attempt ended."""
    if res.ok:
        return "committed"
    if res.error == "stale_after_commit":
        return "committed_then_compensated"
    if res.data.get("blocked"):
        return "prevented"
    return "failed"


def tap_chronos_session(session: Any, tap: WriteTap) -> None:
    """Wrap the session's executor so every write attempt is recorded. The planner looks the
    method up on the executor object at call time, so an instance attribute is enough."""
    executor = session.executor
    original = executor.execute_write

    async def wrapped(call, epoch, snapshot):
        spec = executor.registry.get(call.tool)
        try:
            args = spec.schema.model_validate(call.args).model_dump(mode="json")
        except Exception:  # noqa: BLE001 - invalid args: record raw, the executor rejects them
            args = dict(call.args)
        attempt = tap.begin(call.tool, args)
        try:
            res = await original(call, epoch, snapshot)
        except asyncio.CancelledError:
            tap.end(attempt, "cancelled")
            raise
        tap.end(attempt, classify_result(res))
        return res

    executor.execute_write = wrapped
