"""Write tools: mutate the world. Reachable only through ToolExecutor.execute_write."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chronos.tools.read_tools import _Norm
from chronos.tools.registry import Compensator, write_tool
from chronos.tools.world import World


class BookFlightArgs(_Norm):
    origin: str = "MAA"
    dest: str
    date: str
    time: str
    passenger: str = "Guest"
    seat_pref: str | None = None


class CancelBookingArgs(BaseModel):
    booking_id: str


class SetNavigationArgs(BaseModel):
    destination: str
    origin: str = "current location"
    via: str | None = None


class ClearNavigationArgs(BaseModel):
    nav_id: str


class ReserveTableArgs(_Norm):
    party_size: int = Field(ge=1, le=20)
    date: str | None = None
    time: str = "20:00"
    restaurant: str | None = None
    name: str = "Guest"


class CancelReservationArgs(BaseModel):
    reservation_id: str


@write_tool("book_flight", BookFlightArgs, Compensator(
    "cancel_booking", lambda _a, r: {"booking_id": r["booking_id"]}))
async def book_flight(world: World, a: BookFlightArgs) -> dict[str, Any]:
    return await world.book_flight(a.origin, a.dest, a.date, a.time, a.passenger, a.seat_pref)


@write_tool("cancel_booking", CancelBookingArgs)
async def cancel_booking(world: World, a: CancelBookingArgs) -> dict[str, Any]:
    return await world.cancel_booking(a.booking_id)


@write_tool("set_navigation", SetNavigationArgs, Compensator(
    "clear_navigation", lambda _a, r: {"nav_id": r["nav_id"]}))
async def set_navigation(world: World, a: SetNavigationArgs) -> dict[str, Any]:
    return await world.set_navigation(a.destination, a.via, a.origin)


@write_tool("clear_navigation", ClearNavigationArgs)
async def clear_navigation(world: World, a: ClearNavigationArgs) -> dict[str, Any]:
    return await world.clear_navigation(a.nav_id)


@write_tool("reserve_table", ReserveTableArgs, Compensator(
    "cancel_reservation", lambda _a, r: {"reservation_id": r["reservation_id"]}))
async def reserve_table(world: World, a: ReserveTableArgs) -> dict[str, Any]:
    return await world.reserve_table(a.party_size, a.date, a.time, a.restaurant, a.name)


@write_tool("cancel_reservation", CancelReservationArgs)
async def cancel_reservation(world: World, a: CancelReservationArgs) -> dict[str, Any]:
    return await world.cancel_reservation(a.reservation_id)
