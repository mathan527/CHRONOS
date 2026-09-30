"""Fast path: deterministic, template-based acknowledgments. No LLM, no I/O, pure functions.

Hard rule: an ack is sent BEFORE the work is done, so its wording must never imply completion
("Got it, switching to 6 pm…", never "Booked!"). tests/test_fastpath.py enforces a banned-word
list over every template.
"""
from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from typing import Any

from chronos.perception.intent import Intent
from chronos.protocol import BargeInType, OutputMessage, OutputStatus, OutputType

I, B = Intent, BargeInType

CITY_NAMES = {"DEL": "Delhi", "MAA": "Chennai", "BOM": "Mumbai", "BLR": "Bengaluru"}

# (intent, barge-in type) -> template. barge=None is the first acknowledgment of a new request.
# A `None` intent is the generic fallback for that barge-in type.
TEMPLATES: dict[tuple[Intent | None, BargeInType | None], str] = {
    (I.BOOK_FLIGHT, None): "On it — looking up flights to {dest}…",
    (I.CHANGE_BOOKING, None): "Okay — pulling up that booking…",
    (I.CANCEL_BOOKING, None): "Okay — checking that booking first…",
    (I.NAVIGATE, None): "On it — working out the route to {destination}…",
    (I.ADD_STOP, None): "Got it — looking for a {stop} along your route…",
    (I.RESERVE_TABLE, None): "On it — checking tables for {party_size}…",
    (I.TROUBLESHOOT, None): "Let me take a look — checking what the camera sees…",
    (I.SMALLTALK, None): "Hi! How can I help?",
    (None, None): "Got it — working on that…",
    (None, B.CORRECTION): "Got it, switching to {change}…",
    (I.NAVIGATE, B.GOAL_CHANGE): "Okay, changing plans — {via} first, then {destination}…",
    (I.ADD_STOP, B.GOAL_CHANGE): "Okay, changing plans — {stop} first…",
    (I.BOOK_FLIGHT, B.GOAL_CHANGE): "Okay, changing plans — looking up flights to {dest}…",
    (I.RESERVE_TABLE, B.GOAL_CHANGE): "Okay, changing plans — checking tables for {party_size}…",
    (I.TROUBLESHOOT, B.GOAL_CHANGE): "Okay, changing plans — taking a look at the machine…",
    (None, B.GOAL_CHANGE): "Okay, changing plans — one moment…",
    (None, B.ADDITION): "Sure — adding {change}…",
    (None, B.CANCEL): "Okay, stopping that.",
    (None, B.CLARIFICATION_QUESTION): "Good question — one moment…",
}
# Used instead of the primary template when one of its slots is unknown.
ALT_TEMPLATES: dict[tuple[Intent | None, BargeInType | None], str] = {
    (I.BOOK_FLIGHT, None): "On it — looking up flights…",
    (I.NAVIGATE, None): "On it — working out the route…",
    (I.ADD_STOP, None): "Got it — looking for a stop along your route…",
    (I.RESERVE_TABLE, None): "On it — checking table availability…",
}
CANCEL_WITH_UNDO = "Okay, stopping that — I'll undo anything already in progress."
CORRECTION_FALLBACK = "Got it, updating that…"
ADDITION_FALLBACK = "Sure — adding that…"

ALL_TEMPLATES: tuple[str, ...] = (
    *TEMPLATES.values(), *ALT_TEMPLATES.values(), CANCEL_WITH_UNDO, CORRECTION_FALLBACK,
    ADDITION_FALLBACK)

# Words that claim or imply the work is finished. No template or rendered ack may contain them.
COMPLETION_WORDS = ("booked", "done", "confirmed", "reserved", "set", "complete", "completed",
                    "finished", "placed", "sorted", "cancelled", "canceled", "successfully",
                    "ready", "all set", "success")
_COMPLETION_RE = re.compile(r"\b(?:" + "|".join(COMPLETION_WORDS) + r")\b", re.IGNORECASE)


def contains_completion_claim(text: str) -> bool:
    return bool(_COMPLETION_RE.search(text))


def fmt_time(hhmm: str) -> str:
    h, m = (int(x) for x in hhmm.split(":"))
    suffix = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"


def fmt_date(iso: str, today: date) -> str:
    d = date.fromisoformat(iso)
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    if d == today + timedelta(days=7):
        return "next week"
    return f"{d:%a} {d.day} {d:%b}"


def describe_slot(key: str, value: Any, today: date) -> str | None:
    """Human phrase for one changed slot, or None if it has no natural phrasing."""
    if key == "time":
        return fmt_time(str(value))
    if key == "date":
        return fmt_date(str(value), today)
    if key == "party_size":
        return f"a table for {value}"
    if key == "seat_pref":
        return f"a {value} seat"
    if key in ("dest", "origin"):
        return CITY_NAMES.get(str(value), str(value))
    if key in ("destination", "via", "stop", "restaurant"):
        return str(value)
    return None


class _Safe(dict):  # unknown placeholders render as a neutral word, never raise
    def __missing__(self, key: str) -> str:
        return "that"


class AckBuilder:
    def __init__(self, today: Callable[[], date] | None = None) -> None:
        self._today = today or date.today

    @staticmethod
    def _values(slots: Mapping[str, Any]) -> _Safe:
        v = _Safe()
        for k, val in slots.items():
            v[k] = CITY_NAMES.get(str(val), str(val)) if k in ("dest", "origin") else val
        return v

    def build(self, intent: Intent | None, barge: BargeInType | None = None,
              slots: Mapping[str, Any] | None = None, *,
              changed: Mapping[str, Any] | None = None, has_writes: bool = False) -> str | None:
        """Ack text for an (intent, barge-in) situation, or None when the agent must stay quiet
        (BACKCHANNEL / HESITATION never get an ack: they must not interrupt)."""
        if barge in (B.BACKCHANNEL, B.HESITATION):
            return None
        if barge is B.CANCEL and has_writes:
            return CANCEL_WITH_UNDO
        slots = dict(slots or {})
        today = self._today()
        if barge in (B.CORRECTION, B.ADDITION):
            phrases = [p for k, v in (changed or {}).items()
                       if (p := describe_slot(k, v, today))]
            if not phrases:
                return CORRECTION_FALLBACK if barge is B.CORRECTION else ADDITION_FALLBACK
            return TEMPLATES[(None, barge)].format_map(_Safe(change=" and ".join(phrases)))
        key = (intent, barge)
        template = TEMPLATES.get(key) or TEMPLATES[(None, barge)]
        needed = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        if needed - slots.keys():  # a slot we would have to invent: use the neutral variant
            template = ALT_TEMPLATES.get(key) or TEMPLATES[(None, barge)]
        return template.format_map(self._values(slots))

    @staticmethod
    def message(session_id: str, epoch: int, text: str) -> OutputMessage:
        return OutputMessage(type=OutputType.ACK, session_id=session_id, epoch=epoch,
                             status=OutputStatus.PENDING, text=text)
