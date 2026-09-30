"""The recording world, the tap, and the baseline: the pieces the comparison stands on."""
import asyncio
from datetime import date

import pytest
from helpers import IMAGES

from bench.baseline_agent import BaselineAgent
from bench.instrument import RecordingWorld, WriteTap, tap_chronos_session
from bench.metrics import check_state, snapshot_world, stale_stats
from bench.scenarios import Expect
from chronos.agent.session import AgentSession
from chronos.protocol import Event, EventType, OutputType
from chronos.slowpath.llm import MockLLM
from chronos.slowpath.vision import MockVision
from chronos.tools.world import ToolError

TODAY = date(2026, 10, 1)
TOMORROW, NEXT_WEEK = "2026-10-02", "2026-10-08"


# ------------------------------------------------------------------------ RecordingWorld -----
async def test_recording_world_logs_top_level_calls_only():
    w = await RecordingWorld.create(":memory:", latency_ms=(1, 2), today=TODAY)
    await w.reserve_table(4, None, "20:00", None, "G")  # checks availability internally
    await w.set_navigation("chennai airport", None, "current location")  # routes internally
    kinds = [(c.kind, c.tool) for c in w.calls]
    assert kinds == [("write", "reserve_table"), ("write", "set_navigation")]
    assert all(c.committed and c.t_done >= c.t_enter for c in w.calls)
    await w.search_flights(None, "DEL", TOMORROW, None)
    assert w.calls[-1].kind == "read" and not w.calls[-1].committed
    assert [c.tool for c in w.primary_writes()] == ["reserve_table", "set_navigation"]
    await w.close()


async def test_recording_world_records_failures_and_compensations():
    w = await RecordingWorld.create(":memory:", latency_ms=(1, 2), today=TODAY)
    with pytest.raises(ToolError):
        await w.book_flight("MAA", "DEL", "2031-01-01", "06:00", "G", None)
    failed = w.calls[-1]
    assert failed.error == "ToolError" and not failed.committed
    assert w.primary_writes() == []  # a write that failed is not ground truth
    r = await w.book_flight("MAA", "DEL", TOMORROW, "06:00", "G", None)
    await w.cancel_booking(r["booking_id"])
    assert w.compensated_ids() == {r["booking_id"]}
    assert w.last_activity() is not None
    await w.close()


async def test_recording_world_concurrent_tasks_do_not_hide_each_others_calls():
    w = await RecordingWorld.create(":memory:", latency_ms=(1, 2), today=TODAY)
    await asyncio.gather(*(w.search_flights(None, "DEL", TOMORROW, None) for _ in range(5)),
                         w.reserve_table(2, None, "19:00", None, "A"))
    assert sum(1 for c in w.calls if c.tool == "search_flights") == 5
    assert sum(1 for c in w.calls if c.tool == "reserve_table") == 1
    await w.close()


# ---------------------------------------------------------------------------- the baseline ----
async def baseline(filter_noise=True, silence_ms=120):
    w = await RecordingWorld.create(":memory:", seed=3, latency_ms=(30, 50), today=TODAY)
    tap = WriteTap()
    a = BaselineAgent("b", world=w, llm=MockLLM(), vision=MockVision(), today=TODAY,
                      silence_ms=silence_ms, filter_noise=filter_noise, tap=tap)
    times: list[float] = []
    a.subscribe(lambda _m: times.append(asyncio.get_running_loop().time()))
    a.start()
    return a, w, tap


async def say(agent, text):
    await agent.submit(Event(session_id="b", type=EventType.TRANSCRIPT_FINAL,
                             payload={"text": text}))


async def finish(agent, world):
    await agent.wait_idle(10)
    return await snapshot_world(world)


async def test_baseline_is_half_duplex_it_only_speaks_after_acting():
    a, w, _ = await baseline()
    await say(a, "Book a flight to Delhi tomorrow")
    await asyncio.sleep(0.02)  # it is busy searching: nothing has been said
    assert a.outputs == []
    st = await finish(a, w)
    assert [m.type for m in a.outputs] == [OutputType.ACTION_RESULT]  # no ack, no progress
    assert len([b for b in st["bookings"] if b["status"] == "BOOKED"]) == 1
    await a.aclose()
    await w.close()


async def test_baseline_restart_after_the_write_committed_double_books():
    """The failure CHRONOS exists to prevent: the user changes their mind after it booked."""
    a, w, tap = await baseline()
    await say(a, "Book a flight to Delhi tomorrow")
    await a.wait_idle(10)
    await say(a, "Actually, next week instead")
    st = await finish(a, w)
    live = [b for b in st["bookings"] if b["status"] == "BOOKED"]
    assert sorted(b["date"] for b in live) == [TOMORROW, NEXT_WEEK]  # both stand
    ok, why = check_state(Expect(bookings=({"dest": "DEL", "date": NEXT_WEEK},)), st, None)
    assert not ok and "2 live bookings" in why[0]
    assert sum(1 for c in st["charges"] if c["status"] == "CHARGED") == 2  # charged twice
    stats = stale_stats(tap, w, Expect(bookings=({"dest": "DEL", "date": NEXT_WEEK},)), st)
    assert stats["started"] == 1 and stats["standing"] == 1 and stats["prevented"] == 0
    await a.aclose()
    await w.close()


async def test_baseline_restart_during_the_read_never_starts_the_stale_write():
    a, w, tap = await baseline()
    await say(a, "Book a flight to Delhi tomorrow")
    await asyncio.sleep(0.01)  # still searching
    await say(a, "Actually, next week instead")
    st = await finish(a, w)
    assert len([b for b in st["bookings"] if b["status"] == "BOOKED"]) == 1
    expect = Expect(bookings=({"dest": "DEL", "date": NEXT_WEEK},))
    assert check_state(expect, st, None)[0]
    assert stale_stats(tap, w, expect, st)["started"] == 0
    await a.aclose()
    await w.close()


async def test_baseline_restart_while_the_write_dispatches_cancels_it_cleanly():
    a, w, tap = await baseline()
    await say(a, "Navigate to Chennai Airport")
    while not any(t.tool == "set_navigation" for t in tap.attempts):  # write is dispatching
        await asyncio.sleep(0.002)
    await say(a, "Actually, gas station first")
    st = await finish(a, w)
    expect = Expect(navigation=({"destination": "chennai airport", "via": "gas station"},))
    assert check_state(expect, st, None)[0]
    stale = [t for t in tap.attempts if t.args.get("via") is None]
    assert stale and stale[0].outcome in ("cancelled", "committed")  # cancelled unless it won the race
    await a.aclose()
    await w.close()


async def test_baseline_cancel_undoes_only_what_it_remembers():
    a, w, _ = await baseline()
    await say(a, "Book a flight to Delhi tomorrow")
    await a.wait_idle(10)
    await say(a, "cancel that")
    st = await finish(a, w)
    assert [b for b in st["bookings"] if b["status"] == "BOOKED"] == []
    assert [c["status"] for c in st["charges"]] == ["REFUNDED"]
    assert "undid" in a.outputs[-1].text
    await a.aclose()
    await w.close()


async def test_generous_baseline_ignores_backchannels_strict_baseline_restarts_and_double_books():
    for filter_noise, expected_live in ((True, 1), (False, 2)):
        a, w, _ = await baseline(filter_noise=filter_noise)
        await say(a, "Book a flight to Delhi tomorrow")
        await a.wait_idle(10)
        await say(a, "okay")
        st = await finish(a, w)
        assert len([b for b in st["bookings"] if b["status"] == "BOOKED"]) == expected_live
        await a.aclose()
        await w.close()


async def test_baseline_endpoints_partials_on_silence():
    a, w, _ = await baseline(silence_ms=80)
    await a.submit(Event(session_id="b", type=EventType.TRANSCRIPT_PARTIAL,
                         payload={"text": "book a table for 4"}))
    st = await finish(a, w)
    assert len([r for r in st["reservations"] if r["status"] == "ACTIVE"]) == 1
    await a.aclose()
    await w.close()


async def test_baseline_diagnoses_sequentially_with_the_same_llm_code():
    a, w, _ = await baseline()
    await a.submit(Event(session_id="b", type=EventType.CAMERA_FRAME, payload={
        "path": str(IMAGES / "breaker_tripped.png"), "text": "What's wrong with this machine?"}))
    await finish(a, w)
    diag = a.outputs[-1].data["diagnosis"]
    assert diag["kb_ids"][0] == "tripped_breaker" and diag["grounded"] and diag["source"] == "llm"
    await a.aclose()
    await w.close()


# ------------------------------------------------------------------ CHRONOS under the tap ----
async def test_tap_records_chronos_write_outcomes(fast_settings):
    w = await RecordingWorld.create(":memory:", seed=3, latency_ms=(30, 50), today=TODAY)
    s = await AgentSession.create("t1", fast_settings, world=w, today=TODAY)
    tap = WriteTap()
    tap_chronos_session(s, tap)
    s.start()
    ev = lambda t: Event(session_id="t1", type=EventType.TRANSCRIPT_FINAL, payload={"text": t})
    await s.submit(ev("Book a flight to Delhi tomorrow"))
    await s.wait_idle(10)
    await s.submit(ev("Actually, next week instead"))
    await s.wait_idle(10)
    st = await snapshot_world(w)
    expect = Expect(bookings=({"dest": "DEL", "date": NEXT_WEEK},))
    assert check_state(expect, st, None)[0]
    stats = stale_stats(tap, w, expect, st)
    assert stats == {"started": 1, "prevented": 0, "compensated": 1, "standing": 0,
                     "superseded": 0}  # committed before the change, then undone
    assert [a.outcome for a in tap.attempts] == ["committed", "committed"]
    await s.aclose()


async def test_tap_records_a_prevented_stale_write_when_the_interrupt_lands_mid_dispatch(
        fast_settings):
    w = await RecordingWorld.create(":memory:", seed=3, latency_ms=(60, 80), today=TODAY)
    s = await AgentSession.create("t2", fast_settings, world=w, today=TODAY)
    tap = WriteTap()
    tap_chronos_session(s, tap)
    s.start()
    ev = lambda t: Event(session_id="t2", type=EventType.TRANSCRIPT_FINAL, payload={"text": t})
    await s.submit(ev("Navigate to Chennai Airport"))
    while not tap.attempts:  # the write is now in its dispatch window
        await asyncio.sleep(0.002)
    await s.submit(ev("Actually, gas station first"))
    await s.wait_idle(10)
    st = await snapshot_world(w)
    expect = Expect(navigation=({"destination": "chennai airport", "via": "gas station"},))
    assert check_state(expect, st, None)[0]
    stats = stale_stats(tap, w, expect, st)
    assert stats["started"] == 1 and stats["prevented"] == 1 and stats["standing"] == 0
    assert [a.outcome for a in tap.attempts] == ["prevented", "committed"]
    await s.aclose()
