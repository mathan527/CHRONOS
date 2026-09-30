"""Client-facing wording for results. Unlike acks, these MAY say something is done, but only
ever after the underlying write has really committed (they are built from ToolResults)."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from chronos.fastpath.ack import CITY_NAMES, fmt_time
from chronos.perception.intent import Intent
from chronos.protocol import ToolResult

_MISSING_PROMPTS = {
    "dest": "Where would you like to fly to?",
    "date": "Which day would you like to travel?",
    "destination": "Where would you like to go?",
    "party_size": "For how many people?",
    "booking_id": "Which booking? Please tell me the booking ID.",
    "stop": "Which place would you like to stop at?",
}

_BLOCKED = {
    "no_flights": "I couldn't find any flights for that. Would you like a different day?",
    "no_flight_at_time": "There's no flight at that time. Would you like another time?",
    "search_failed": "I couldn't search flights just now. Please try again.",
    "route_failed": "I couldn't find a route to that place. Could you name it differently?",
    "no_table": "No table is free at that time. Would you like a different time or size?",
    "booking_not_found": "I couldn't find that booking. Could you check the booking ID?",
    "already_cancelled": "That booking is already cancelled.",
    "change_booking_not_supported": "I can't change an existing booking yet; I can cancel it "
                                    "and book a new one.",
    "no_write_for_intent": "I'm not sure how to do that yet.",
}


def missing_prompt(missing: tuple[str, ...] | list[str]) -> str:
    return _MISSING_PROMPTS.get(missing[0], "Could you give me a bit more detail?") \
        if missing else "Could you give me a bit more detail?"


def blocked_text(note: str) -> str:
    return _BLOCKED.get(note, "Sorry, I couldn't do that.")


def _city(code: str) -> str:
    return CITY_NAMES.get(code, code)


def write_result_text(res: ToolResult) -> str:
    d: Mapping[str, Any] = res.data
    if res.tool == "book_flight":
        return (f"Booked {_city(d['origin'])} to {_city(d['dest'])} on "
                f"{fmt_date_plain(d['date'])} at {fmt_time(d['time'])}. "
                f"Booking {d['booking_id']}, INR {d['amount']}.")
    if res.tool == "set_navigation":
        r = d.get("route", {})
        via = f" via {r['via']}" if r.get("via") else ""
        return (f"Navigation started to {r.get('destination', 'your destination')}{via}. "
                f"About {r.get('eta_min', '?')} minutes.")
    if res.tool == "reserve_table":
        return (f"Reserved a table for {d['party_size']} at {d['restaurant']} on "
                f"{fmt_date_plain(d['date'])} at {fmt_time(d['time'])}. "
                f"Reservation {d['reservation_id']}.")
    if res.tool == "cancel_booking":
        return f"Cancelled booking {d['booking_id']}; the charge was refunded."
    return f"{res.tool} completed."


def fmt_date_plain(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%a} {d.day} {d:%b}"


def compensation_text(res: ToolResult) -> str:
    d = res.data
    if res.tool == "cancel_booking":
        return f"I cancelled the earlier booking {d.get('booking_id', '')} and refunded it."
    if res.tool == "clear_navigation":
        return "I cleared the earlier route."
    if res.tool == "cancel_reservation":
        return f"I cancelled the earlier table reservation {d.get('reservation_id', '')}."
    return "I undid the earlier action."


def diagnosis_text(d: Mapping[str, Any]) -> str:
    steps = " ".join(f"{i}. {s}" for i, s in enumerate(d.get("steps", []), 1))
    return f"{d['diagnosis']} Try this: {steps}" if steps else str(d["diagnosis"])


def answer_question(intent: Intent | None, status: str, slots: Mapping[str, Any],
                    write_args: Mapping[str, Any] | None) -> str:
    """Deterministic answer to 'what are you doing?'-style questions from the plan state."""
    if intent is None:
        return "I'm not working on anything right now."
    working = "I've finished" if status == "done" else "I'm working on"
    if intent is Intent.BOOK_FLIGHT:
        w = write_args or {}
        when = f" at {fmt_time(w['time'])}" if w.get("time") else ""
        day = f" on {fmt_date_plain(slots['date'])}" if slots.get("date") else ""
        return f"{working} a flight to {_city(str(slots.get('dest', 'your destination')))}{day}{when}."
    if intent is Intent.NAVIGATE:
        via = f" via {slots['via']}" if slots.get("via") else ""
        return f"{working} the route to {slots.get('destination', 'your destination')}{via}."
    if intent is Intent.RESERVE_TABLE:
        return f"{working} a table for {slots.get('party_size', 'your group')}."
    return f"{working} that request."
