"""Intent + slot extraction. Rules first (microseconds); LLM only when no rule fires.

`parse_rules` is the shared, pure parser: the barge-in classifier reuses it to tell a parameter
tweak ("make it 6pm") from a new objective ("gas station first").
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chronos.slowpath.llm import LLM, LLMError, LLMInvalidJSON, LLMTimeout
from chronos.tools.world import CITY_CODES, normalize_city, normalize_time
from chronos.trace.logger import ComponentTrace


class Intent(str, Enum):
    BOOK_FLIGHT = "book_flight"
    CHANGE_BOOKING = "change_booking"
    CANCEL_BOOKING = "cancel_booking"
    NAVIGATE = "navigate"
    ADD_STOP = "add_stop"
    RESERVE_TABLE = "reserve_table"
    TROUBLESHOOT = "troubleshoot"
    SMALLTALK = "smalltalk"


# --------------------------------------------------------------------- slot schemas ---------
class _Slots(BaseModel):
    model_config = ConfigDict(extra="ignore")


class BookFlightSlots(_Slots):
    origin: str | None = None
    dest: str | None = None
    date: str | None = None
    time: str | None = None
    passenger: str | None = None
    seat_pref: str | None = None


class ChangeBookingSlots(_Slots):
    booking_id: str | None = None
    dest: str | None = None
    date: str | None = None
    time: str | None = None


class CancelBookingSlots(_Slots):
    booking_id: str | None = None


class NavigateSlots(_Slots):
    destination: str | None = None
    via: str | None = None


class AddStopSlots(_Slots):
    stop: str | None = None


class ReserveTableSlots(_Slots):
    party_size: int | None = Field(default=None, ge=1, le=20)
    date: str | None = None
    time: str | None = None
    restaurant: str | None = None


class TroubleshootSlots(_Slots):
    symptom: str | None = None


class SmalltalkSlots(_Slots):
    pass


SLOT_MODELS: dict[Intent, type[_Slots]] = {
    Intent.BOOK_FLIGHT: BookFlightSlots, Intent.CHANGE_BOOKING: ChangeBookingSlots,
    Intent.CANCEL_BOOKING: CancelBookingSlots, Intent.NAVIGATE: NavigateSlots,
    Intent.ADD_STOP: AddStopSlots, Intent.RESERVE_TABLE: ReserveTableSlots,
    Intent.TROUBLESHOOT: TroubleshootSlots, Intent.SMALLTALK: SmalltalkSlots,
}
REQUIRED: dict[Intent, tuple[str, ...]] = {
    Intent.BOOK_FLIGHT: ("dest",), Intent.NAVIGATE: ("destination",), Intent.ADD_STOP: ("stop",),
    Intent.RESERVE_TABLE: ("party_size",),
}


class IntentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: Intent
    slots: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    source: Literal["rules", "llm", "fallback"] = "rules"
    missing: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


# ------------------------------------------------------------------------ text utils --------
_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_NUMW = "|".join(_NUMS)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}
_CITY_RE = "|".join(sorted((re.escape(c) for c in CITY_CODES), key=len, reverse=True))
_RESTAURANTS = ("saffron garden", "marina bites", "spice route")
_PLACE_SYN = {"petrol bunker": "gas station", "petrol pump": "gas station",
              "fuel station": "gas station", "petrol station": "gas station",
              "petrol": "gas station", "gas": "gas station", "fuel": "gas station",
              "diesel": "gas station"}
_BARE_NUMBER = re.compile(rf"^(\d{{1,2}}|{_NUMW})(?: (?:people|persons|pax|please))?$")
_STOPWORDS_LEAD = re.compile(r"^(?:the|a|an|some|any|one|quick|nearest|nearby|closest)\s+")


def norm(text: str) -> str:
    t = text.lower().replace("…", "...").replace("’", "'")
    return re.sub(r"\s+", " ", t).strip()


def clean_phrase(s: str) -> str:
    s = re.split(r"\b(?:via|through|and then|then|please|now|first|instead|only|na|yaar)\b", s)[0]
    s = re.sub(r"[^\w\s'&.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-")
    while (n := _STOPWORDS_LEAD.sub("", s)) != s:
        s = n
    return _PLACE_SYN.get(s, s)


def find_time(text: str) -> str | None:
    t = text.lower().replace(".", "")
    if re.search(r"\bnoon\b", t):
        return "12:00"
    if re.search(r"\bmidnight\b", t):
        return "00:00"
    m = re.search(rf"\b(\d{{1,2}}|{_NUMW})(?::(\d{{2}}))?\s*(am|pm)\b", t)
    if m:
        h = m.group(1)
        h = str(_NUMS.get(h, h))
        try:
            return normalize_time(f"{h}:{m.group(2) or '00'} {m.group(3)}")
        except ValueError:
            return None
    m = re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", t)
    if m:
        try:
            return normalize_time(m.group(0))
        except ValueError:
            return None
    return None


def resolve_date(text: str, today: date) -> str | None:
    t = text.lower()
    if "day after tomorrow" in t:
        return (today + timedelta(days=2)).isoformat()
    if re.search(r"\b(tomorrow|tomorow|tommorow|tmrw|tmr)\b", t):
        return (today + timedelta(days=1)).isoformat()
    if re.search(r"\b(today|tonight)\b", t):
        return today.isoformat()
    if re.search(r"\bnext week\b", t):
        return (today + timedelta(days=7)).isoformat()
    m = re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError:
            return None
    m = re.search(rf"\b({'|'.join(_WEEKDAYS)})\b", t)
    if m:
        delta = (_WEEKDAYS.index(m.group(1)) - today.weekday()) % 7 or 7
        return (today + timedelta(days=delta)).isoformat()
    months = "|".join(_MONTHS)
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months})\b", t) or None
    d_m = (m.group(1), m.group(2)) if m else None
    if not d_m:
        m2 = re.search(rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t)
        d_m = (m2.group(2), m2.group(1)) if m2 else None
    if d_m:
        try:
            d = date(today.year, _MONTHS[d_m[1]], int(d_m[0]))
            if d < today:
                d = d.replace(year=today.year + 1)
            return d.isoformat()
        except ValueError:
            return None
    return None


def find_party(text: str) -> int | None:
    t = text.lower()
    m = re.search(rf"\b(?:for|party of|table of|group of)\s+(\d{{1,2}}|{_NUMW})\b(?!\s*(?::|am\b|pm\b))", t)
    if not m:
        m = re.search(rf"\b(\d{{1,2}}|{_NUMW})\s+(?:people|persons|pax|guests|of us|members)\b", t)
    if not m:  # "make it six" / "change it to 5" (but never "make it 6pm" or "make it 7:30")
        m = re.search(rf"\b(?:make it|change it to|make that)\s+(?:for\s+)?(\d{{1,2}}|{_NUMW})\b"
                      rf"(?!\s*(?::|am\b|pm\b|\d))", t)
    if not m:
        return None
    v = m.group(1)
    n = _NUMS.get(v) or int(v)
    return n if 1 <= n <= 20 else None


def find_seat_pref(text: str) -> str | None:
    m = re.search(r"\b(window|aisle|middle)\b(?:\s+seat)?", text.lower())
    return m.group(1) if m else None


# --------------------------------------------------------------------- intent rules ---------
_PLACE_NOUNS = (r"gas station|petrol|fuel|diesel|atm|pharmacy|chemist|restaurant|coffee|bank|"
                r"hospital|mall|home|office|airport|station|hotel|market|bunker")
_R = {
    "cancel_booking": re.compile(r"\bcancel\b.*\b(booking|flight|ticket|reservation|trip|bk-\d+)"),
    "change_booking": re.compile(
        r"\b(change|reschedule|modify|move|shift|switch|postpone|prepone)\b.*"
        r"\b(booking|flight|reservation|ticket|trip)\b"),
    "reserve_table": re.compile(r"\btable\b|\breservation\b|\breserve\b"),
    "book_flight": re.compile(
        r"\b(book|reserve|get|need|want|buy|find|search|check)\b.*\b(flight|ticket|tickets|fly)\b"
        r"|\bfly(?:ing)?\s+to\b|\bflights?\s+(?:to|from|for|on)\b"),
    "add_stop": re.compile(
        rf"\bstop(?:ping)?\s+(?:at|by|for|near)\b|\badd\s+(?:a\s+)?stop\b|\bon the way\b"
        rf"|\bneed\s+(?:some\s+|to get\s+)?(?:petrol|gas|fuel|diesel)\b"
        rf"|\b(?:{_PLACE_NOUNS})\b.*\bfirst\b|\bfirst\b.*\b(?:{_PLACE_NOUNS})\b"),
    "navigate": re.compile(
        r"\b(navigate|navigation|directions?|route)\b|\b(take me|go|head|drive|bring me)\b"
        r"(?:\s+me)?\s+(?:to|towards?)\b"),
    "troubleshoot": re.compile(
        r"what'?s wrong|whats wrong|not working|doesn'?t work|isn'?t working|\bbroken\b|"
        r"stopped working|malfunction|diagnos|troubleshoot|\bfault\b|overheat|\bsmok(?:e|ing)\b|"
        r"making (?:a |some )?(?:noise|sound)|won'?t (?:start|turn on|boot)|\bmachine\b|"
        r"\bequipment\b|error code"),
    "smalltalk": re.compile(
        r"^(?:hi|hello|hey|hii+|namaste|good (?:morning|afternoon|evening|night)|thanks?|"
        r"thank you|bye|goodbye|how are you|who are you|what can you do)\b"),
}
_ORDER = ["cancel_booking", "change_booking", "reserve_table", "book_flight", "add_stop",
          "navigate", "troubleshoot", "smalltalk"]


def detect_intent(body: str) -> Intent | None:
    for name in _ORDER:
        if _R[name].search(body):
            return Intent(name)
    return None


@dataclass(frozen=True)
class Parsed:
    intent: Intent | None
    slots: dict[str, Any] = field(default_factory=dict)


def find_slots(text: str, intent: Intent | None, today: date) -> dict[str, Any]:
    """Slots relevant to `intent` (or the generic time/date/party/seat set when None)."""
    t = norm(text)
    s: dict[str, Any] = {}
    time_ = find_time(t)
    date_ = resolve_date(t, today)
    party = find_party(t)
    seat = find_seat_pref(t)
    generic = intent is None
    if intent in (Intent.BOOK_FLIGHT, Intent.CHANGE_BOOKING) or generic:
        if (m := re.search(rf"\bto\s+({_CITY_RE})\b", t)) or (
                intent is Intent.BOOK_FLIGHT and (m := re.search(rf"\bfor\s+({_CITY_RE})\b", t))):
            s["dest"] = normalize_city(m.group(1))
        if intent is Intent.BOOK_FLIGHT and (m := re.search(rf"\bfrom\s+({_CITY_RE})\b", t)):
            s["origin"] = normalize_city(m.group(1))
        if intent is Intent.BOOK_FLIGHT and seat:
            s["seat_pref"] = seat
        if generic and seat:
            s["seat_pref"] = seat
    if intent in (Intent.BOOK_FLIGHT, Intent.CHANGE_BOOKING, Intent.RESERVE_TABLE) or generic:
        if time_:
            s["time"] = time_
        if date_:
            s["date"] = date_
    if intent is Intent.RESERVE_TABLE or generic:
        if party:
            s["party_size"] = party
        if intent is Intent.RESERVE_TABLE and not party and (m := _BARE_NUMBER.match(t.strip(" .!?"))):
            s["party_size"] = _NUMS.get(m.group(1)) or int(m.group(1))  # answer to "for how many?"
        for r in _RESTAURANTS:
            if r in t:
                s["restaurant"] = r.title()
    if intent in (Intent.CHANGE_BOOKING, Intent.CANCEL_BOOKING) and (
            m := re.search(r"\bbk-(\d+)\b", t)):
        s["booking_id"] = f"BK-{int(m.group(1)):04d}"
    if intent is Intent.NAVIGATE:
        m = re.search(r"\b(?:navigate|drive|go|head|take me|bring me|directions?|route)\b"
                      r"(?:\s+me)?\s+(?:to|towards?)\s+(.+)", t) or re.search(
            r"\b(?:navigate|navigation|directions?|route)\s+(?:to\s+)?(.+)", t)
        if m and (d := clean_phrase(m.group(1))):
            s["destination"] = d.title()
        m = re.search(r"\b(?:via|through|passing|stop(?:ping)? (?:at|by))\s+(.+)", t)
        if m and (v := clean_phrase(m.group(1))):
            s["via"] = v.title()
    if intent is Intent.ADD_STOP:
        stop = None
        if m := re.search(r"\bneed\s+(?:some\s+|to get\s+)?(petrol|gas|fuel|diesel)\b", t):
            stop = "gas station"
        elif (m := re.search(r"\bstop(?:ping)?\s+(?:at|by|for|near)\s+(.+)", t)) or (m := re.search(r"\badd\s+(?:a\s+)?stop(?:\s+(?:at|for|by))?\s*(.*)", t)) or (m := re.search(r"^(?:actually|no|ok|okay|so|wait|hey)?[, ]*(?:let'?s |i need to |"
                            r"we need to |go to |first go to )?(.+?)\s+first\b", t)) or (m := re.search(r"\bvia\s+(.+)", t)):
            stop = clean_phrase(m.group(1))
        if stop:
            s["stop"] = stop
    if intent is Intent.TROUBLESHOOT:
        s["symptom"] = text.strip()
    return s


def parse_rules(text: str, today: date) -> Parsed:
    t = norm(text)
    intent = detect_intent(t)
    return Parsed(intent, find_slots(t, intent, today))


# ------------------------------------------------------------------------- extractor --------
_SYSTEM = (
    "You extract the user's intent for a voice assistant. Reply with ONLY a JSON object: "
    '{"intent": one of [book_flight, change_booking, cancel_booking, navigate, add_stop, '
    'reserve_table, troubleshoot, smalltalk], "slots": {...}, "confidence": 0..1}. '
    "Slots: book_flight{origin,dest,date,time,seat_pref}; change_booking{booking_id,date,time,"
    "dest}; cancel_booking{booking_id}; navigate{destination,via}; add_stop{stop}; "
    "reserve_table{party_size,date,time,restaurant}; troubleshoot{symptom}. Dates are ISO "
    "YYYY-MM-DD, times 24h HH:MM. Omit slots the user did not state.")


class IntentExtractor:
    def __init__(self, llm: LLM | None = None, *, today: date | None = None,
                 timeout_s: float = 4.0, rules_threshold: float = 0.75,
                 trace: ComponentTrace | None = None) -> None:
        self.llm, self.timeout_s, self.threshold = llm, timeout_s, rules_threshold
        self._today = today
        self._trace = trace

    def today(self) -> date:
        return self._today or date.today()  # noqa: DTZ011 - local calendar day

    def _finish(self, r: IntentResult, t0: float, epoch: int, text: str) -> IntentResult:
        r = r.model_copy(update={"latency_ms": (time.perf_counter() - t0) * 1000})
        if self._trace:
            self._trace.emit("intent_extracted", epoch=epoch, text=text, intent=r.intent.value,
                             source=r.source, confidence=r.confidence, slots=r.slots)
        return r

    def from_rules(self, text: str) -> IntentResult | None:
        p = parse_rules(text, self.today())
        if p.intent is None:
            return None
        missing = [k for k in REQUIRED.get(p.intent, ()) if k not in p.slots]
        conf = 0.9 if not missing else 0.8
        return IntentResult(intent=p.intent, slots=p.slots, confidence=conf, source="rules",
                            missing=missing)

    def slot_updates(self, text: str, intent: Intent | None,
                     awaiting: tuple[str, ...] = ()) -> dict[str, Any]:
        """Slots stated in `text` for the current task's intent, for an incremental patch
        (e.g. 'make it 6pm' -> {'time': '18:00'}). `awaiting` lists slots the agent just asked
        for, which lets a bare answer ('Delhi', 'Chennai Airport') count as that slot."""
        s = find_slots(text, intent, self.today())
        t = norm(text).strip(" .!?")
        if "dest" in awaiting and "dest" not in s and re.fullmatch(_CITY_RE, t):
            s["dest"] = normalize_city(t)
        if ("destination" in awaiting and "destination" not in s and len(t.split()) <= 5
                and (d := clean_phrase(t))):
            s["destination"] = d.title()
        return s

    async def extract(self, text: str, *, epoch: int = 0) -> IntentResult:
        t0 = time.perf_counter()
        ruled = self.from_rules(text)
        if ruled is not None and ruled.confidence >= self.threshold:
            return self._finish(ruled, t0, epoch, text)
        if self.llm is not None:
            for _attempt in range(2):  # one retry, and only for malformed output
                try:
                    raw = await self.llm.chat_json(
                        "intent", _SYSTEM, json.dumps({"utterance": text}),
                        timeout=self.timeout_s)
                    return self._finish(self._validate(raw), t0, epoch, text)
                except (LLMInvalidJSON, ValidationError, KeyError, TypeError, ValueError):
                    continue
                except (TimeoutError, LLMTimeout, LLMError):
                    break
        fallback = ruled or IntentResult(intent=Intent.SMALLTALK, confidence=0.2,
                                         source="fallback")
        return self._finish(fallback.model_copy(update={"source": "fallback"}), t0, epoch, text)

    @staticmethod
    def _validate(raw: dict[str, Any]) -> IntentResult:
        intent = Intent(raw["intent"])
        slots = SLOT_MODELS[intent].model_validate(raw.get("slots") or {}).model_dump(
            exclude_none=True)
        conf = min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0)
        missing = [k for k in REQUIRED.get(intent, ()) if k not in slots]
        return IntentResult(intent=intent, slots=slots, confidence=conf, source="llm",
                            missing=missing)
