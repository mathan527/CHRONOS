"""Read tools: side-effect free, safe to run speculatively."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from chronos.tools.registry import read_tool
from chronos.tools.world import World, normalize_city, normalize_date, normalize_time


class _Norm(BaseModel):
    @field_validator("origin", "dest", check_fields=False, mode="before")
    @classmethod
    def _city(cls, v: Any) -> Any:
        return normalize_city(v) if isinstance(v, str) else v

    @field_validator("date", check_fields=False, mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return normalize_date(v) if isinstance(v, str) else v

    @field_validator("time", check_fields=False, mode="before")
    @classmethod
    def _time(cls, v: Any) -> Any:
        return normalize_time(v) if isinstance(v, str) else v


class SearchFlightsArgs(_Norm):
    origin: str | None = None
    dest: str
    date: str | None = None
    time: str | None = None


class GetRouteArgs(BaseModel):
    destination: str
    origin: str = "current location"
    via: str | None = None


class CheckTableArgs(_Norm):
    party_size: int = Field(ge=1, le=20)
    date: str | None = None
    time: str = "20:00"
    restaurant: str | None = None


class GetBookingArgs(BaseModel):
    booking_id: str


class KbArgs(BaseModel):
    query: str = Field(min_length=1)
    observations: str = ""


@read_tool("search_flights", SearchFlightsArgs)
async def search_flights(world: World, a: SearchFlightsArgs) -> dict[str, Any]:
    flights = await world.search_flights(a.origin, a.dest, a.date, a.time)
    return {"flights": flights, "count": len(flights)}


@read_tool("get_route", GetRouteArgs)
async def get_route(world: World, a: GetRouteArgs) -> dict[str, Any]:
    return await world.get_route(a.origin, a.destination, a.via)


@read_tool("check_table_availability", CheckTableArgs)
async def check_table_availability(world: World, a: CheckTableArgs) -> dict[str, Any]:
    tables = await world.table_availability(a.party_size, a.date, a.time, a.restaurant)
    return {"tables": tables, "count": len(tables)}


@read_tool("get_booking", GetBookingArgs)
async def get_booking(world: World, a: GetBookingArgs) -> dict[str, Any]:
    return await world.get_booking(a.booking_id)


@read_tool("lookup_troubleshooting_kb", KbArgs)
async def lookup_troubleshooting_kb(world: World, a: KbArgs) -> dict[str, Any]:
    matches = world.kb_lookup(f"{a.query} {a.observations}")
    return {"matches": matches, "count": len(matches)}
