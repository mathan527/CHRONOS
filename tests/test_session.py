import asyncio
import json
import time

import pytest
from helpers import (
    IMAGES,
    Driver,
    assert_ack_precedes_slow_work,
    assert_acks_fast,
    assert_no_stale_output,
    assert_world_consistent,
)

from chronos.protocol import Event, EventType
from chronos.protocol import OutputStatus as S
from chronos.protocol import OutputType as T
from chronos.slowpath.llm import MockLLM
from chronos.slowpath.planner import PlannerPolicy


# ----------------------------------------------------------------- loop behaviour ------------
async def test_loop_never_blocks_on_a_slow_llm(open_session):
    """An LLM intent lookup takes 0.6 s; the loop keeps acking other events meanwhile."""
    s = await open_session(llm=MockLLM(latency_s=0.6))
    d = Driver(s)
    await d.say("how do I get a refund?")  # no rule matches -> slow LLM intent fallback
    await asyncio.sleep(0.05)
    assert s.cancellation.active(), "the slow path should be in flight"
    t0 = time.monotonic()
    await d.say("hello there")  # arrives while the first request is still being understood
    while len(d.acks()) < 2:
        await asyncio.sleep(0.005)
    assert (time.monotonic() - t0) * 1000 < 300
    assert s.cancellation.active(), "...and the first request is STILL running"
    await d.idle(timeout=10)
    assert_acks_fast(d, 2)
    assert_no_stale_output(d)
    assert len(d.of(T.RESPONSE)) == 2  # both requests were eventually answered


async def test_backchannels_are_traced_and_ignored_while_busy(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await asyncio.sleep(0.02)
    await d.say("okay")
    await d.text("uh-huh")
    await d.idle()
    assert_acks_fast(d, 1)  # no ack for a backchannel
    assert s.metrics.ignored["backchannel"] == 2 and not d.events("epoch_bumped")
    labels = [r["label"] for r in d.events("classified")]
    assert labels.count("backchannel") == 2
    state = await assert_world_consistent(d)
    assert len(state["world"]["bookings"]) == 1  # the booking was not disturbed
    assert_no_stale_output(d)


async def test_a_question_is_answered_without_touching_the_task(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await asyncio.sleep(0.02)
    await d.say("what time does it leave?")
    await d.idle()
    assert [a.text for a in d.acks()][1] == "Good question — one moment…"
    answers = [m for m in d.of(T.RESPONSE)]
    assert len(answers) == 1 and "flight to Delhi" in answers[0].text
    assert s.epochs.current() == 1
    state = await assert_world_consistent(d)
    assert len(state["world"]["bookings"]) == 1
    assert_no_stale_output(d)


async def test_interrupt_events_jump_the_queue(open_session):
    s = await open_session(start=False)  # queue everything first, then let the loop run
    d = Driver(s)
    await d.text("hello there")
    await d.text("hello again")
    await d.interrupt(text="stop")  # INTERRUPT outranks TEXT
    s.start()
    await d.idle()
    rows = d.rows()
    ignored = next(i for i, r in enumerate(rows) if r["event"] == "ignored")
    first_ack = next(i for i, r in enumerate(rows) if r["event"] == "ack_sent")
    assert ignored < first_ack  # the interrupt was handled before the earlier text events


async def test_explicit_interrupt_cancel_stops_work_in_flight(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await asyncio.sleep(0.02)
    await d.interrupt(action="cancel")
    state = await d.idle() or await d.state()
    assert state["world"]["bookings"] == []
    assert await d.ledger("book_flight", "COMMITTED") == []
    assert d.outs[-1].status is S.CANCELLED and d.outs[-1].text == "Okay, I've stopped that."
    assert_no_stale_output(d)
    await assert_world_consistent(d)


async def test_speech_onset_interrupt_signal_changes_nothing(open_session):
    s = await open_session()
    d = Driver(s)
    await d.interrupt(kind="speech_start")
    await d.idle()
    assert d.outs == [] and s.epochs.current() == 1 and d.events("interrupt_signal")


async def test_bad_camera_payload_reports_an_error_and_the_loop_survives(open_session):
    s = await open_session()
    d = Driver(s)
    await d.send(EventType.CAMERA_FRAME, {"path": "does_not_exist.png"})
    await d.say("hello there")
    await d.idle()
    assert d.of(T.ERROR) and d.of(T.ERROR)[0].status is S.FAILED
    assert any(m.type is T.RESPONSE for m in d.outs)  # still alive


async def test_frame_sent_before_a_question_is_used_even_though_transcripts_outrank_frames(
        open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "breaker_tripped.png")
    await d.say("What's wrong with this machine?")  # queued right behind the frame
    await d.idle()
    diag = d.of(T.RESPONSE)[0].data["diagnosis"]
    assert diag["grounded"] is True and diag["kb_ids"][0] == "tripped_breaker"


# --------------------------------------------------------------- slot filling / confirm -----
async def test_missing_slot_is_asked_for_and_the_answer_fills_it_without_interrupting(
        open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("book a table")
    await d.idle()
    (ask,) = d.of(T.RESPONSE)
    assert ask.status is S.PENDING and ask.text == "For how many people?"
    assert ask.data == {"missing": ["party_size"]}
    await d.say("four")
    state = await d.idle() or await d.state()
    assert s.epochs.current() == 1  # a slot fill is not an interruption
    assert [a.text for a in d.acks()][1] == "Sure — adding a table for 4…"
    assert len(state["world"]["reservations"]) == 1
    assert state["world"]["reservations"][0]["party_size"] == 4
    assert_no_stale_output(d)


async def test_bare_city_answers_a_missing_destination(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("book a flight tomorrow")
    await d.idle()
    assert d.of(T.RESPONSE)[0].text == "Where would you like to fly to?"
    await d.say("Delhi")
    state = await d.idle() or await d.state()
    assert len(state["world"]["bookings"]) == 1
    assert "20261002" in state["world"]["bookings"][0]["flight_id"]
    assert "DEL" in state["world"]["bookings"][0]["flight_id"]


async def test_okay_is_not_taken_as_the_answer_to_a_slot_question(open_session):
    d2 = Driver(await open_session("s2"))
    await d2.say("book a table")
    await d2.idle()
    await d2.say("okay")
    await d2.idle()
    assert d2.s.metrics.ignored["backchannel"] == 1
    assert (await d2.state())["world"]["reservations"] == []


async def test_explicit_confirmation_flow(open_session):
    s = await open_session(policy=PlannerPolicy(confirm_writes=frozenset({"book_flight"})))
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await d.idle()
    (ask,) = d.of(T.RESPONSE)
    assert ask.status is S.PENDING and "yes" in ask.text.lower()
    assert (await d.state())["world"]["bookings"] == []  # nothing written before the "yes"
    await d.say("yes")
    await d.idle()
    state = await assert_world_consistent(d)
    assert len(state["world"]["bookings"]) == 1
    assert d.acks()[1].text == "Okay — going ahead…"
    assert s.epochs.current() == 1
    assert_no_stale_output(d)


# ----------------------------------------------------------------------- undo window --------
async def test_a_stray_stop_long_after_completion_does_not_undo_old_work(open_session):
    s = await open_session(policy=PlannerPolicy(undo_window_s=0.0))
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await d.idle()
    await asyncio.sleep(0.02)
    await d.say("stop")
    await d.idle()
    assert s.metrics.ignored["cancel_nothing"] == 1
    state = await assert_world_consistent(d)
    assert len(state["world"]["bookings"]) == 1 and s.epochs.current() == 1


async def test_a_new_request_after_a_finished_goal_leaves_it_alone(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await d.idle()
    await d.say("Book a table for 3")
    await d.idle()
    state = await assert_world_consistent(d)
    assert len(state["world"]["bookings"]) == 1 and len(state["world"]["reservations"]) == 1
    assert s.epochs.current() == 1  # no bump: nothing was superseded
    assert_no_stale_output(d)


# --------------------------------------------------------------------- introspection -------
async def test_describe_is_json_and_reports_metrics(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await d.idle()
    state = json.loads(json.dumps(await s.describe()))
    assert state["epoch"] == 1 and state["goal"] == {
        "intent": "book_flight", "status": "done", "writes": 1}
    assert state["plan"]["status"] == "done" and state["metrics"]["stale_emitted"] == 0
    assert state["metrics"]["ack_ms"]["n"] == 1 and state["ledger"][0]["status"] == "COMMITTED"
    assert state["outputs"] and state["queue"]["size"] == 0


async def test_submit_rejects_events_for_another_session(open_session):
    s = await open_session()
    with pytest.raises(ValueError):
        await s.submit(Event(session_id="someone-else", type=EventType.TEXT, payload={}))


async def test_subscribers_receive_every_output_and_a_broken_one_is_harmless(open_session):
    s = await open_session()
    got = []
    unsub = s.subscribe(got.append)
    s.subscribe(lambda _m: 1 / 0)
    d = Driver(s)
    await d.say("hello there")
    await d.idle()
    assert [m.type for m in got] == [T.ACK, T.RESPONSE]
    unsub()
    await d.say("hello again")
    await d.idle()
    assert len(got) == 2 and len(d.outs) == 4
    assert_ack_precedes_slow_work(d)


async def test_a_question_after_a_finished_goal_is_a_new_request_not_a_question_about_it(
        open_session):
    """Regression found by driving the real demo client: after a reservation completed,
    'What's wrong with this machine?' was answered as 'I've finished a table for 4'."""
    s = await open_session()
    d = Driver(s)
    await d.say("Book a table for 4")
    await d.idle()
    await d.frame(IMAGES / "breaker_tripped.png", text="What is wrong with this machine?")
    await d.idle()
    diag = d.of(T.RESPONSE)[-1].data["diagnosis"]
    assert diag["grounded"] is True and diag["kb_ids"][0] == "tripped_breaker"
    state = await assert_world_consistent(d)
    assert len(state["world"]["reservations"]) == 1  # the reservation was left alone
    assert s.epochs.current() == 1
    assert_no_stale_output(d)
