"""Mock world backing every tool: static catalogue + mutable state in SQLite (aiosqlite).

Latency is injected per tool (seedable) so interruptions genuinely land mid-task. All mutating
operations run under one asyncio.Lock inside a transaction, so world invariants hold no matter
how tool calls interleave. `check_invariants()` is the oracle the tests use.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

import aiosqlite


class ToolError(Exception):
    """A tool failed for a domain reason (unknown flight, sold out, ...). Not a crash."""


# --------------------------------------------------------------------------- normalisers ------
CITY_CODES = {
    "chennai": "MAA", "madras": "MAA", "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM",
    "bombay": "BOM", "bengaluru": "BLR", "bangalore": "BLR",
}
_TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap]m)?\s*$", re.IGNORECASE)


def normalize_city(value: str) -> str:
    v = value.strip()
    if re.fullmatch(r"[A-Za-z]{3}", v) and v.upper() in CITY_CODES.values():
        return v.upper()
    code = CITY_CODES.get(v.lower())
    if code is None:
        raise ValueError(f"unknown city: {value!r}")
    return code


def normalize_time(value: str) -> str:
    """'6pm' / '18:00' / '6:30 PM' -> 'HH:MM' (24h)."""
    m = _TIME_RE.match(value)
    if not m:
        raise ValueError(f"bad time: {value!r}")
    h, mi, mer = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if mer:
        if not 1 <= h <= 12:
            raise ValueError(f"bad time: {value!r}")
        h = h % 12 + (12 if mer == "pm" else 0)
    if h > 23 or mi > 59:
        raise ValueError(f"bad time: {value!r}")
    return f"{h:02d}:{mi:02d}"


def normalize_date(value: str) -> str:
    return date.fromisoformat(value.strip()).isoformat()


PLACES: dict[str, tuple[str, float, float]] = {
    "current location": ("Current Location", 13.0674, 80.2376),
    "chennai airport": ("Chennai Airport", 12.9941, 80.1709),
    "gas station": ("Indian Oil Fuel Station, Guindy", 13.0090, 80.2200),
    "home": ("Home", 13.0827, 80.2707),
    "office": ("Office, OMR", 12.9010, 80.2279),
    "marina beach": ("Marina Beach", 13.0500, 80.2824),
    "t nagar": ("T. Nagar", 13.0418, 80.2341),
    "central station": ("Chennai Central", 13.0827, 80.2755),
}
_PLACE_ALIASES = {
    "airport": "chennai airport", "the airport": "chennai airport", "maa airport": "chennai airport",
    "gas": "gas station", "petrol bunker": "gas station", "fuel station": "gas station",
    "petrol station": "gas station", "a gas station": "gas station", "the gas station": "gas station",
    "here": "current location", "my location": "current location",
}


def resolve_place(name: str) -> tuple[str, float, float]:
    k = re.sub(r"\s+", " ", name.strip().lower())
    k = _PLACE_ALIASES.get(k, k)
    if k not in PLACES:
        raise ToolError(f"unknown_place: {name!r}")
    return PLACES[k]


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


KB: list[dict[str, Any]] = [
    {"id": "loose_power_cable", "title": "Loose or damaged power cable",
     "keywords": ["power", "cable", "cord", "plug", "loose", "off", "dead", "intermittent",
                  "flicker", "led", "socket", "connector", "wire", "wires", "frayed"],
     "causes": "Power cable not seated, or connector/insulation damaged.",
     "steps": ["Switch the machine off and unplug it.", "Inspect cable and connector for damage.",
               "Reseat the cable firmly at both ends.", "Replace the cable if frayed."]},
    {"id": "tripped_breaker", "title": "Tripped circuit breaker",
     "keywords": ["breaker", "trip", "tripped", "fuse", "no power", "dead", "panel", "switch",
                  "overload", "shut", "shutdown", "electrical"],
     "causes": "Overload or short caused the breaker to trip.",
     "steps": ["Locate the breaker panel.", "Unplug heavy loads on the circuit.",
               "Reset the breaker fully off then on.", "If it trips again, call an electrician."]},
    {"id": "overheating_fan", "title": "Overheating due to failed cooling fan",
     "keywords": ["hot", "heat", "overheat", "overheating", "fan", "cooling", "vent", "noise",
                  "grinding", "temperature", "thermal", "smoke", "shutdown", "dust", "blocked"],
     "causes": "Fan seized/stopped or vents blocked, so the machine overheats and throttles.",
     "steps": ["Power off and let the machine cool for 15 minutes.", "Clear dust from vents.",
               "Check the fan spins freely; replace it if it does not.",
               "Restart and monitor temperature."]},
    {"id": "clogged_filter", "title": "Clogged air/oil filter",
     "keywords": ["filter", "clog", "clogged", "pressure", "flow", "slow", "weak", "dirty",
                  "oil", "leak", "leaking", "fluid"],
     "causes": "Filter saturated, restricting flow.",
     "steps": ["Depressurise the machine.", "Remove and inspect the filter.",
               "Clean or replace it.", "Check fluid levels."]},
    {"id": "worn_belt", "title": "Worn or slipping drive belt",
     "keywords": ["belt", "slip", "slipping", "squeal", "squeak", "vibration", "vibrating",
                  "motor", "spin", "rotating", "worn", "cracked"],
     "causes": "Belt worn or loose, losing drive to the motor.",
     "steps": ["Power off and lock out.", "Inspect belt for cracks and glazing.",
               "Adjust tension or replace the belt."]},
]
_GENERIC_KB = {"id": "no_match", "title": "No specific match", "keywords": [],
               "causes": "Symptoms did not match a known fault.",
               "steps": ["Power off safely.", "Check power supply and visible damage.",
                         "Escalate to a field technician."], "score": 0}

_SCHEMA = """
CREATE TABLE flights (
    flight_id TEXT PRIMARY KEY, origin TEXT NOT NULL, dest TEXT NOT NULL, date TEXT NOT NULL,
    time TEXT NOT NULL, price INTEGER NOT NULL, capacity INTEGER NOT NULL,
    seats_left INTEGER NOT NULL CHECK (seats_left >= 0));
CREATE TABLE bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, flight_id TEXT NOT NULL REFERENCES flights,
    passenger TEXT NOT NULL, seat_pref TEXT, status TEXT NOT NULL);
CREATE TABLE charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id INTEGER NOT NULL UNIQUE REFERENCES bookings, amount INTEGER NOT NULL,
    status TEXT NOT NULL);
CREATE TABLE rest_tables (
    table_id INTEGER PRIMARY KEY, restaurant TEXT NOT NULL, capacity INTEGER NOT NULL);
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, table_id INTEGER NOT NULL REFERENCES rest_tables,
    restaurant TEXT NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL, party_size INTEGER NOT NULL,
    name TEXT NOT NULL, status TEXT NOT NULL);
CREATE UNIQUE INDEX uq_active_res ON reservations(table_id, date, time) WHERE status='ACTIVE';
CREATE TABLE navigation (
    id INTEGER PRIMARY KEY AUTOINCREMENT, destination TEXT NOT NULL, via TEXT, origin TEXT NOT NULL,
    route_json TEXT NOT NULL, status TEXT NOT NULL);
"""
_ROUTES = [("MAA", "DEL", 5200), ("DEL", "MAA", 5300), ("MAA", "BOM", 4400), ("BOM", "MAA", 4300),
           ("MAA", "BLR", 2600), ("BLR", "MAA", 2500), ("DEL", "BOM", 4900), ("BOM", "DEL", 5000)]
_TIMES = ["06:00", "08:00", "12:30", "18:00", "20:00"]
_RESTAURANTS = {"Saffron Garden": [2, 2, 4, 4, 6], "Marina Bites": [2, 4, 4, 8],
                "Spice Route": [2, 2, 2, 4, 6, 10]}


def _fmt(prefix: str, n: int) -> str:
    return f"{prefix}-{n:04d}"


def _pk(prefix: str, ident: str) -> int:
    m = re.fullmatch(rf"{prefix}-(\d+)", ident.strip())
    if not m:
        raise ToolError(f"bad_id: {ident!r}")
    return int(m.group(1))


class World:
    def __init__(self, db: aiosqlite.Connection, *, seed: int = 0,
                 latency_ms: tuple[int, int] = (100, 800),
                 tool_latency_ms: dict[str, tuple[int, int]] | None = None,
                 today: date | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._db = db
        self._rng = random.Random(seed)
        self._lat, self._tool_lat = latency_ms, dict(tool_latency_ms or {})
        self._sleep = sleep
        self.today = today or date.today()  # noqa: DTZ011 - local calendar day
        self._wlock = asyncio.Lock()

    @classmethod
    async def create(cls, path: str = ":memory:", *, seed: int = 0, **kw: Any) -> World:
        db = await aiosqlite.connect(path)
        await db.executescript(_SCHEMA)
        w = cls(db, seed=seed, **kw)
        await w._seed()
        return w

    async def close(self) -> None:
        await self._db.close()

    async def _seed(self) -> None:
        rows = []
        for o, d, base in _ROUTES:
            for day in range(31):
                dt = (self.today + timedelta(days=day)).isoformat()
                for t in _TIMES:
                    cap = self._rng.randint(3, 9)
                    price = base + self._rng.randint(-300, 600)
                    fid = f"{o}{d}-{dt.replace('-', '')}-{t.replace(':', '')}"
                    rows.append((fid, o, d, dt, t, price, cap, cap))
        await self._db.executemany("INSERT INTO flights VALUES (?,?,?,?,?,?,?,?)", rows)
        tid = 1
        for name, caps in _RESTAURANTS.items():
            for c in caps:
                await self._db.execute("INSERT INTO rest_tables VALUES (?,?,?)", (tid, name, c))
                tid += 1
        await self._db.commit()

    # -- latency ------------------------------------------------------------------------
    async def delay(self, tool: str) -> float:
        lo, hi = self._tool_lat.get(tool, self._lat)
        secs = self._rng.uniform(lo, hi) / 1000
        await self._sleep(secs)
        return secs * 1000

    # -- helpers ------------------------------------------------------------------------
    async def _all(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(sql, params) as cur:
            return await cur.fetchall()

    async def _one(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        rows = await self._all(sql, params)
        return rows[0] if rows else None

    # -- reads --------------------------------------------------------------------------
    async def search_flights(self, origin: str | None, dest: str, on: str | None,
                             time: str | None, limit: int = 10) -> list[dict[str, Any]]:
        sql, p = "SELECT * FROM flights WHERE dest=? AND seats_left>0", [dest]
        for col, val in (("origin", origin), ("date", on), ("time", time)):
            if val:
                sql += f" AND {col}=?"
                p.append(val)
        sql += " ORDER BY date, time LIMIT ?"
        return [dict(r) for r in await self._all(sql, (*p, limit))]

    async def get_route(self, origin: str, destination: str, via: str | None) -> dict[str, Any]:
        pts = [resolve_place(origin)] + ([resolve_place(via)] if via else []) + [
            resolve_place(destination)]
        km = sum(_haversine_km(a[1:], b[1:]) for a, b in pairwise(pts)) * 1.3
        return {"origin": pts[0][0], "destination": pts[-1][0], "via": via and pts[1][0],
                "distance_km": round(km, 1), "eta_min": max(1, round(km / 35 * 60)),
                "waypoints": [{"name": n, "lat": la, "lon": lo} for n, la, lo in pts]}

    async def table_availability(self, party_size: int, on: str | None, time: str,
                                 restaurant: str | None) -> list[dict[str, Any]]:
        on = on or self.today.isoformat()
        sql = ("SELECT t.table_id, t.restaurant, t.capacity FROM rest_tables t WHERE t.capacity>=? "
               "AND NOT EXISTS (SELECT 1 FROM reservations r WHERE r.table_id=t.table_id "
               "AND r.date=? AND r.time=? AND r.status='ACTIVE')")
        p: list[Any] = [party_size, on, time]
        if restaurant:
            sql += " AND lower(t.restaurant)=lower(?)"
            p.append(restaurant)
        sql += " ORDER BY t.capacity, t.table_id"
        return [{**dict(r), "date": on, "time": time} for r in await self._all(sql, tuple(p))]

    async def get_booking(self, booking_id: str) -> dict[str, Any]:
        r = await self._one(
            "SELECT b.id, b.flight_id, b.passenger, b.seat_pref, b.status, f.origin, f.dest, "
            "f.date, f.time, c.amount, c.status AS charge_status FROM bookings b "
            "JOIN flights f ON f.flight_id=b.flight_id LEFT JOIN charges c ON c.booking_id=b.id "
            "WHERE b.id=?", (_pk("BK", booking_id),))
        if not r:
            raise ToolError(f"no_such_booking: {booking_id}")
        d = dict(r)
        d["booking_id"] = _fmt("BK", d.pop("id"))
        return d

    @staticmethod
    def kb_lookup(query: str, limit: int = 3) -> list[dict[str, Any]]:
        toks = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for e in KB:
            s = sum(1 for k in e["keywords"] if k in toks or any(k in t for t in toks if len(t) > 3))
            if s:
                scored.append({**e, "score": s})
        scored.sort(key=lambda e: -e["score"])
        return scored[:limit] or [_GENERIC_KB]

    # -- writes (single lock + transaction => invariants hold under interleaving) -------
    async def _tx(self, fn: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        async with self._wlock:
            try:
                out = await fn()
                await self._db.commit()
                return out
            except BaseException:
                await asyncio.shield(self._db.rollback())
                raise

    async def book_flight(self, origin: str, dest: str, on: str, time: str, passenger: str,
                          seat_pref: str | None) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            f = await self._one("SELECT * FROM flights WHERE origin=? AND dest=? AND date=? "
                                "AND time=?", (origin, dest, on, time))
            if not f:
                raise ToolError(f"no_such_flight: {origin}->{dest} {on} {time}")
            cur = await self._db.execute(
                "UPDATE flights SET seats_left=seats_left-1 WHERE flight_id=? AND seats_left>0",
                (f["flight_id"],))
            if cur.rowcount != 1:
                raise ToolError(f"sold_out: {f['flight_id']}")
            cur = await self._db.execute(
                "INSERT INTO bookings (flight_id,passenger,seat_pref,status) VALUES (?,?,?,'BOOKED')",
                (f["flight_id"], passenger, seat_pref))
            bid = cur.lastrowid
            cur = await self._db.execute(
                "INSERT INTO charges (booking_id,amount,status) VALUES (?,?,'CHARGED')",
                (bid, f["price"]))
            return {"booking_id": _fmt("BK", bid), "charge_id": _fmt("CH", cur.lastrowid),
                    "flight_id": f["flight_id"], "origin": origin, "dest": dest, "date": on,
                    "time": time, "amount": f["price"], "passenger": passenger}
        return await self._tx(go)

    async def cancel_booking(self, booking_id: str) -> dict[str, Any]:
        bid = _pk("BK", booking_id)

        async def go() -> dict[str, Any]:
            b = await self._one("SELECT * FROM bookings WHERE id=?", (bid,))
            if not b:
                raise ToolError(f"no_such_booking: {booking_id}")
            if b["status"] == "CANCELLED":
                return {"booking_id": booking_id, "already_cancelled": True}
            await self._db.execute("UPDATE bookings SET status='CANCELLED' WHERE id=?", (bid,))
            await self._db.execute(
                "UPDATE flights SET seats_left=seats_left+1 WHERE flight_id=?", (b["flight_id"],))
            await self._db.execute("UPDATE charges SET status='REFUNDED' WHERE booking_id=?", (bid,))
            return {"booking_id": booking_id, "already_cancelled": False, "refunded": True}
        return await self._tx(go)

    async def set_navigation(self, destination: str, via: str | None,
                             origin: str) -> dict[str, Any]:
        route = await self.get_route(origin, destination, via)

        async def go() -> dict[str, Any]:
            await self._db.execute(
                "UPDATE navigation SET status='SUPERSEDED' WHERE status='ACTIVE'")
            cur = await self._db.execute(
                "INSERT INTO navigation (destination,via,origin,route_json,status) "
                "VALUES (?,?,?,?,'ACTIVE')", (destination, via, origin, json.dumps(route)))
            return {"nav_id": _fmt("NAV", cur.lastrowid), "route": route}
        return await self._tx(go)

    async def clear_navigation(self, nav_id: str) -> dict[str, Any]:
        n = _pk("NAV", nav_id)

        async def go() -> dict[str, Any]:
            cur = await self._db.execute(
                "UPDATE navigation SET status='CLEARED' WHERE id=? AND status='ACTIVE'", (n,))
            if cur.rowcount == 0 and not await self._one("SELECT 1 FROM navigation WHERE id=?", (n,)):
                raise ToolError(f"no_such_navigation: {nav_id}")
            return {"nav_id": nav_id, "cleared": cur.rowcount == 1}
        return await self._tx(go)

    async def reserve_table(self, party_size: int, on: str | None, time: str,
                            restaurant: str | None, name: str) -> dict[str, Any]:
        on = on or self.today.isoformat()

        async def go() -> dict[str, Any]:
            free = await self.table_availability(party_size, on, time, restaurant)
            if not free:
                raise ToolError("no_table_available")
            t = free[0]
            cur = await self._db.execute(
                "INSERT INTO reservations (table_id,restaurant,date,time,party_size,name,status) "
                "VALUES (?,?,?,?,?,?,'ACTIVE')",
                (t["table_id"], t["restaurant"], on, time, party_size, name))
            return {"reservation_id": _fmt("RS", cur.lastrowid), "restaurant": t["restaurant"],
                    "table_id": t["table_id"], "date": on, "time": time, "party_size": party_size,
                    "name": name}
        return await self._tx(go)

    async def cancel_reservation(self, reservation_id: str) -> dict[str, Any]:
        r = _pk("RS", reservation_id)

        async def go() -> dict[str, Any]:
            cur = await self._db.execute(
                "UPDATE reservations SET status='CANCELLED' WHERE id=? AND status='ACTIVE'", (r,))
            if cur.rowcount == 0 and not await self._one(
                    "SELECT 1 FROM reservations WHERE id=?", (r,)):
                raise ToolError(f"no_such_reservation: {reservation_id}")
            return {"reservation_id": reservation_id, "cancelled": cur.rowcount == 1}
        return await self._tx(go)

    # -- test / demo support ------------------------------------------------------------
    async def set_seats(self, flight_id: str, seats_left: int) -> None:
        await self._db.execute("UPDATE flights SET seats_left=?, capacity=MAX(capacity,?) "
                               "WHERE flight_id=?", (seats_left, seats_left, flight_id))
        await self._db.commit()

    async def active_bookings(self) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._all("SELECT * FROM bookings WHERE status='BOOKED'")]

    async def charges(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return [dict(r) for r in await self._all("SELECT * FROM charges WHERE status=?", (status,))]
        return [dict(r) for r in await self._all("SELECT * FROM charges")]

    async def active_reservations(self) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._all("SELECT * FROM reservations WHERE status='ACTIVE'")]

    async def active_navigation(self) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._all("SELECT * FROM navigation WHERE status='ACTIVE'")]

    async def check_invariants(self) -> list[str]:
        """Return human-readable violations; empty list == consistent world."""
        bad: list[str] = []
        for r in await self._all(
                "SELECT f.flight_id, f.capacity, f.seats_left, COUNT(b.id) AS n FROM flights f "
                "LEFT JOIN bookings b ON b.flight_id=f.flight_id AND b.status='BOOKED' "
                "GROUP BY f.flight_id HAVING f.seats_left<0 OR f.seats_left+COUNT(b.id)!=f.capacity"):
            bad.append(f"seat accounting broken on {r['flight_id']}")
        for r in await self._all(
                "SELECT b.id, b.status, COUNT(c.id) AS n, "
                "SUM(c.status='CHARGED') AS charged, SUM(c.status='REFUNDED') AS refunded "
                "FROM bookings b LEFT JOIN charges c ON c.booking_id=b.id GROUP BY b.id"):
            if r["n"] != 1:
                bad.append(f"booking {r['id']} has {r['n']} charges (want exactly 1)")
            elif r["status"] == "BOOKED" and not r["charged"]:
                bad.append(f"active booking {r['id']} not charged")
            elif r["status"] == "CANCELLED" and not r["refunded"]:
                bad.append(f"cancelled booking {r['id']} not refunded")
        if len(await self.active_navigation()) > 1:
            bad.append("more than one active navigation")
        return bad
