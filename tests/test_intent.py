import json
from datetime import date

import httpx
import pytest

from chronos.perception.intent import Intent, IntentExtractor, parse_rules, resolve_date
from chronos.slowpath.llm import LLMInvalidJSON, LLMTimeout, MockLLM, OllamaLLM

TODAY = date(2026, 10, 1)  # a Thursday


def ex(llm=None) -> IntentExtractor:
    return IntentExtractor(llm, today=TODAY)


@pytest.mark.parametrize("text,intent,slots", [
    ("Book a flight to Delhi tomorrow", Intent.BOOK_FLIGHT, {"dest": "DEL", "date": "2026-10-02"}),
    ("book a flight from Chennai to Mumbai next week at 6pm", Intent.BOOK_FLIGHT,
     {"origin": "MAA", "dest": "BOM", "date": "2026-10-08", "time": "18:00"}),
    ("I need a ticket to Bengaluru on Monday with a window seat", Intent.BOOK_FLIGHT,
     {"dest": "BLR", "date": "2026-10-05", "seat_pref": "window"}),
    ("Navigate to Chennai Airport", Intent.NAVIGATE, {"destination": "Chennai Airport"}),
    ("take me to the airport via the gas station", Intent.NAVIGATE,
     {"destination": "Airport", "via": "Gas Station"}),
    ("actually, gas station first", Intent.ADD_STOP, {"stop": "gas station"}),
    ("add a stop at the pharmacy", Intent.ADD_STOP, {"stop": "pharmacy"}),
    ("I need petrol", Intent.ADD_STOP, {"stop": "gas station"}),
    ("book a table for 4", Intent.RESERVE_TABLE, {"party_size": 4}),
    ("reserve a table for six at Saffron Garden tomorrow at 8 pm", Intent.RESERVE_TABLE,
     {"party_size": 6, "restaurant": "Saffron Garden", "date": "2026-10-02", "time": "20:00"}),
    ("cancel my flight booking BK-0007", Intent.CANCEL_BOOKING, {"booking_id": "BK-0007"}),
    ("reschedule my booking to 12 October", Intent.CHANGE_BOOKING, {"date": "2026-10-12"}),
    ("what's wrong with this machine?", Intent.TROUBLESHOOT, {}),
    ("hello there", Intent.SMALLTALK, {}),
])
async def test_rules_extract(text, intent, slots):
    r = await ex().extract(text)
    assert r.intent is intent and r.source == "rules" and r.confidence >= 0.8
    for k, v in slots.items():
        assert r.slots[k] == v, (k, r.slots)


async def test_missing_required_slots_reported():
    r = await ex().extract("book a table")
    assert r.intent is Intent.RESERVE_TABLE and r.missing == ["party_size"]
    r = await ex().extract("book a flight tomorrow")
    assert r.missing == ["dest"] and r.confidence < 0.9


def test_slot_only_updates_for_corrections():
    e = ex()
    assert e.slot_updates("make it 6pm", Intent.BOOK_FLIGHT) == {"time": "18:00"}
    assert e.slot_updates("actually, next week instead", Intent.BOOK_FLIGHT) == {
        "date": "2026-10-08"}
    assert e.slot_updates("actually 5 people", Intent.RESERVE_TABLE) == {"party_size": 5}
    assert e.slot_updates("and add a window seat", Intent.BOOK_FLIGHT) == {"seat_pref": "window"}
    # a table party-size must not be mistaken for a time and vice versa
    assert e.slot_updates("for 4 at 8pm", Intent.RESERVE_TABLE) == {
        "party_size": 4, "time": "20:00"}


def test_date_resolution():
    assert resolve_date("day after tomorrow", TODAY) == "2026-10-03"
    assert resolve_date("on friday", TODAY) == "2026-10-02"
    assert resolve_date("on thursday", TODAY) == "2026-10-08"  # strictly future
    assert resolve_date("5th november", TODAY) == "2026-11-05"
    assert resolve_date("3 january", TODAY) == "2027-01-03"  # rolls to next year
    assert resolve_date("2026-12-25", TODAY) == "2026-12-25"
    assert resolve_date("i may go", TODAY) is None


def test_parse_rules_is_pure_and_fast_path_unknown():
    assert parse_rules("blah blah", TODAY).intent is None


# ----------------------------------------------------------------- LLM fallback ------------
async def test_llm_used_only_when_rules_cannot_parse():
    llm = MockLLM()
    r = await ex(llm).extract("Book a flight to Delhi tomorrow")
    assert r.source == "rules" and llm.calls == []
    r = await ex(llm).extract("I'd like to see the hills next month, any tickets?")
    assert r.source == "llm" and r.intent is Intent.BOOK_FLIGHT and len(llm.calls) == 1


async def test_llm_output_validated_against_slot_schema():
    llm = MockLLM(handlers={"intent": lambda _p: {
        "intent": "reserve_table", "slots": {"party_size": 4, "bogus": 1}, "confidence": 0.9}})
    r = await ex(llm).extract("xyzzy")
    assert r.source == "llm" and r.slots == {"party_size": 4} and r.missing == []


async def test_invalid_output_retries_once_then_succeeds():
    outs = iter([{"intent": "not_an_intent"}, {"intent": "navigate", "slots": {
        "destination": "Home"}, "confidence": 0.8}])
    llm = MockLLM(handlers={"intent": lambda _p: next(outs)})
    r = await ex(llm).extract("xyzzy")
    assert len(llm.calls) == 2 and r.source == "llm" and r.intent is Intent.NAVIGATE


async def test_invalid_output_twice_falls_back():
    llm = MockLLM(handlers={"intent": lambda _p: {"intent": "not_an_intent"}})
    r = await ex(llm).extract("xyzzy")
    assert len(llm.calls) == 2  # exactly one retry
    assert r.source == "fallback" and r.intent is Intent.SMALLTALK and r.confidence <= 0.3


@pytest.mark.parametrize("bad", [{"slots": {}}, {"intent": "book_flight", "slots": {
    "seat_pref": {"nested": 1}}}, {"intent": "reserve_table", "slots": {"party_size": 99}}])
async def test_schema_violations_fall_back(bad):
    llm = MockLLM(handlers={"intent": lambda _p: bad})
    assert (await ex(llm).extract("xyzzy")).source == "fallback"


async def test_llm_timeout_does_not_retry_and_falls_back():
    llm = MockLLM(latency_s=1.0)
    r = await IntentExtractor(llm, today=TODAY, timeout_s=0.05).extract("xyzzy")
    assert r.source == "fallback" and len(llm.calls) == 1


async def test_no_llm_falls_back():
    assert (await ex().extract("xyzzy")).source == "fallback"


# --------------------------------------------------------------------- OllamaLLM ------------
def ollama(handler, **kw) -> OllamaLLM:
    return OllamaLLM(model="llama3.2:3b", transport=httpx.MockTransport(handler), **kw)


async def test_ollama_request_shape_and_parse():
    seen = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"message": {"content": json.dumps({"ok": 1})}})

    out = await ollama(h).chat_json("t", "sys", "usr")
    assert out == {"ok": 1}
    assert seen["url"].endswith("/api/chat")
    b = seen["body"]
    assert b["model"] == "llama3.2:3b" and b["format"] == "json" and b["stream"] is False
    assert [m["role"] for m in b["messages"]] == ["system", "user"]


async def test_ollama_typed_failures():
    def bad_json(_r):
        return httpx.Response(200, json={"message": {"content": "not json"}})

    def not_obj(_r):
        return httpx.Response(200, json={"message": {"content": "[1,2]"}})

    def http500(_r):
        return httpx.Response(500)

    def conn(_r):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMInvalidJSON):
        await ollama(bad_json).chat_json("t", "s", "u")
    with pytest.raises(LLMInvalidJSON):
        await ollama(not_obj).chat_json("t", "s", "u")
    from chronos.slowpath.llm import LLMError
    with pytest.raises(LLMError):
        await ollama(http500).chat_json("t", "s", "u")
    with pytest.raises(LLMError):
        await ollama(conn).chat_json("t", "s", "u")


async def test_ollama_hard_timeout():
    import asyncio

    async def slow(_r):
        await asyncio.sleep(2)
        return httpx.Response(200, json={"message": {"content": "{}"}})

    with pytest.raises(LLMTimeout):
        await ollama(slow, timeout_s=0.05).chat_json("t", "s", "u")
    with pytest.raises(LLMTimeout):
        await ollama(slow, timeout_s=5).chat_json("t", "s", "u", timeout=0.05)
