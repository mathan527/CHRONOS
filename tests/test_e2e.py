"""End-to-end: the four demo use cases from CLAUDE.md, through the real session loop.

Mock LLM, real asyncio, real (30-80 ms) tool latency so interruptions land mid-task. Every test
asserts: ack < 300 ms after each utterance, exactly one committed write per final intent, no
stale-epoch output ever emitted, and the final world state.
"""
import asyncio

from helpers import (
    IMAGES,
    Driver,
    assert_ack_precedes_slow_work,
    assert_acks_fast,
    assert_no_stale_output,
    assert_world_consistent,
)

from chronos.protocol import OutputStatus as S
from chronos.protocol import OutputType as T


async def finish(d: Driver, acks: int):
    """Common closing assertions for every scenario."""
    await d.idle()
    assert_acks_fast(d, acks)
    assert_no_stale_output(d)
    assert_ack_precedes_slow_work(d)
    return await assert_world_consistent(d)


# ============================================================ 1. in-car navigation ===========
async def test_use_case_1_in_car_gas_station_first_mid_route_calculation(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Navigate to Chennai Airport")
    await asyncio.sleep(0.01)  # the route is still being calculated
    await d.say("Actually, gas station first")
    state = await finish(d, acks=2)

    assert [a.text for a in d.acks()] == [
        "On it — working out the route to Chennai Airport…",
        "Okay, changing plans — gas station first…"]
    assert [a.epoch for a in d.acks()] == [1, 2]
    # epoch 1 never got anything past its ack: the old route task was cancelled
    assert [m.type for m in d.outs if m.epoch == 1] == [T.ACK]
    assert any(c["epoch"] == 1 and c["task"] in ("read:get_route", "slow:new:new")
               for c in d.events("task_cancelled"))
    # epoch 2 planned the new route and ran it
    assert [m.type for m in d.outs if m.epoch == 2] == [T.ACK, T.PROGRESS, T.ACTION_RESULT]
    progress = d.of(T.PROGRESS)[0]
    assert progress.status is S.COMMITTED and "gas station" in progress.text
    assert [(r["old_epoch"], r["epoch"], r["reason"]) for r in d.events("epoch_bumped")] == [
        (1, 2, "goal_change")]
    # exactly one set_navigation ever committed, in epoch 2
    live = await d.ledger("set_navigation", "COMMITTED")
    assert len(live) == 1 and live[0]["epoch"] == 2
    assert len(await d.ledger("set_navigation")) == 1  # the epoch-1 write was never even claimed
    assert len(state["world"]["navigation"]) == 1
    assert "Guindy" in state["world"]["navigation"][0]["route_json"]  # routed via the gas station


async def test_use_case_1_variant_interrupt_lands_while_the_write_is_dispatching(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Navigate to Chennai Airport")
    while not d.of(T.PROGRESS):  # wait until the plan is committed and the write is in flight
        await asyncio.sleep(0.005)
    await d.say("Actually, gas station first")
    state = await finish(d, acks=2)
    assert [m.epoch for m in d.of(T.ACTION_RESULT)] == [2]  # nothing from epoch 1 was emitted
    assert len(await d.ledger("set_navigation", "COMMITTED")) == 1
    assert len(state["world"]["navigation"]) == 1


# ================================================================ 2. customer support =========
async def test_use_case_2a_correction_before_the_booking_is_written(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow")
    await asyncio.sleep(0.02)  # flights are still being searched
    await d.say("Actually, next week instead")
    state = await finish(d, acks=2)

    assert [a.text for a in d.acks()] == ["On it — looking up flights to Delhi…",
                                          "Got it, switching to next week…"]
    assert [m.type for m in d.outs if m.epoch == 1] == [T.ACK]
    assert len(state["world"]["bookings"]) == 1  # no double booking
    assert "20261008" in state["world"]["bookings"][0]["flight_id"]  # next week, not tomorrow
    assert [c["status"] for c in state["world"]["charges"]] == ["CHARGED"]  # no double charge
    assert len(await d.ledger("book_flight")) == 1
    assert len(await d.ledger("book_flight", "COMMITTED")) == 1


async def test_use_case_2b_correction_after_the_booking_was_committed(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow")
    await d.idle()
    first = await d.state()
    assert len(first["world"]["bookings"]) == 1
    assert "20261002" in first["world"]["bookings"][0]["flight_id"]  # tomorrow
    assert d.of(T.ACTION_RESULT)[0].status is S.DONE

    await d.say("Actually, next week instead")
    state = await finish(d, acks=2)

    # compensating action: the first booking was cancelled and refunded, one new one exists
    assert len(state["world"]["bookings"]) == 1
    assert "20261008" in state["world"]["bookings"][0]["flight_id"]
    assert sorted(c["status"] for c in state["world"]["charges"]) == ["CHARGED", "REFUNDED"]
    assert sorted(r["status"] for r in await d.ledger("book_flight")) == [
        "COMMITTED", "COMPENSATED"]
    assert len(await d.ledger("book_flight", "COMMITTED")) == 1  # one live write per final intent
    e2 = [m for m in d.outs if m.epoch == 2]
    assert [m.type for m in e2] == [T.ACK, T.ACTION_RESULT, T.PROGRESS, T.ACTION_RESULT]
    assert [m.status for m in e2 if m.type is T.ACTION_RESULT] == [S.CANCELLED, S.DONE]
    assert "cancelled the earlier booking" in e2[1].text
    assert d.events("write_compensated")


async def test_use_case_2_rapid_fire_corrections_leave_exactly_one_booking(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 8pm")
    await asyncio.sleep(0.01)
    await d.say("make it 6pm")
    await asyncio.sleep(0.01)
    await d.say("make it 12:30 pm")
    state = await finish(d, acks=3)
    assert s.epochs.current() == 3
    assert len(state["world"]["bookings"]) == 1
    assert state["world"]["bookings"][0]["flight_id"].endswith("1230")
    assert len(await d.ledger("book_flight", "COMMITTED")) == 1
    assert [c["status"] for c in state["world"]["charges"] if c["status"] == "CHARGED"] == [
        "CHARGED"]


async def test_use_case_2_cancel_right_after_booking_undoes_it(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow at 6pm")
    await d.idle()
    await d.say("cancel that")
    state = await finish(d, acks=2)
    assert d.acks()[1].text == "Okay, stopping that — I'll undo anything already in progress."
    assert state["world"]["bookings"] == []
    assert [c["status"] for c in state["world"]["charges"]] == ["REFUNDED"]
    assert d.outs[-1].status is S.CANCELLED


# ============================================================ 3. field troubleshooting =======
async def test_use_case_3_camera_frame_and_question(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "panel_disconnected_cable.png", text="What's wrong with this machine?")
    state = await finish(d, acks=1)

    assert d.acks()[0].text == "Let me take a look — checking what the camera sees…"
    (resp,) = d.of(T.RESPONSE)
    assert resp.status is S.DONE and "cable" in resp.text.lower()
    diag = resp.data["diagnosis"]
    assert diag["grounded"] is True and diag["kb_ids"][0] == "loose_power_cable"
    assert "cable" in diag["observed"].lower()  # the answer is grounded in what the camera saw
    assert s.vision.calls == ["panel_disconnected_cable.png"]
    assert await d.ledger("book_flight") == [] and state["world"]["bookings"] == []
    assert s.epochs.current() == 1 and not d.events("epoch_bumped")  # reads only: no epoch churn


async def test_use_case_3_vision_starts_speculatively_on_the_partial_transcript(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "machine_overheating_fan.png")
    await d.partial("what's wrong with this machine")
    await asyncio.sleep(0.05)  # the user is still "speaking"
    assert s.vision.calls == ["machine_overheating_fan.png"]
    described_before_final = len(d.events("frame_described"))
    await d.say("What's wrong with this machine?")
    await finish(d, acks=1)
    assert described_before_final == 1 and s.vision.calls == ["machine_overheating_fan.png"]
    (resp,) = d.of(T.RESPONSE)
    assert resp.data["diagnosis"]["kb_ids"][0] == "overheating_fan"
    rows = d.rows()
    described = next(i for i, r in enumerate(rows) if r["event"] == "frame_described")
    acked = next(i for i, r in enumerate(rows) if r["event"] == "ack_sent")
    assert described < acked  # vision was already done when the final arrived


# =============================================================== 4. accessibility ============
async def test_use_case_4_hesitation_then_self_correction_one_reservation(open_session):
    s = await open_session()
    d = Driver(s)
    await d.partial("I want to boo…")
    await d.partial("I want to boo… um…")
    await d.say("actually, book a table for 4")
    state = await finish(d, acks=1)

    assert d.outs[0].type is T.ACK and d.acks()[0].text == "On it — checking tables for 4…"
    assert len(state["world"]["reservations"]) == 1
    assert state["world"]["reservations"][0]["party_size"] == 4
    assert len(await d.ledger("reserve_table", "COMMITTED")) == 1
    assert not d.events("epoch_bumped")  # nothing had been started, so nothing to supersede
    rows = d.rows()
    first_tool = next(i for i, r in enumerate(rows) if r["event"] == "tool_read")
    ack_idx = next(i for i, r in enumerate(rows) if r["event"] == "ack_sent")
    assert ack_idx < first_tool  # hesitation triggered no tool call at all


async def test_use_case_4_a_held_hesitation_is_never_acted_on(open_session):
    s = await open_session()
    d = Driver(s)
    await d.partial("I want to boo…")
    await asyncio.sleep(0.4)  # far longer than the end-of-utterance silence (120 ms)
    await d.idle()
    assert d.outs == [] and s.epochs.current() == 1
    assert d.events("hesitation_held") and not d.events("tool_read")
    await d.say("actually, book a table for 4")
    state = await finish(d, acks=1)
    assert len(state["world"]["reservations"]) == 1


async def test_use_case_4_silence_commits_the_utterance_and_the_late_final_is_ignored(
        open_session):
    s = await open_session()
    d = Driver(s)
    await d.partial("actually, book a table for 4")  # no FINAL: end-of-utterance by silence
    await d.idle()
    assert len(d.acks()) == 1
    await d.say("actually, book a table for 4")  # the ASR's late final for the same words
    state = await finish(d, acks=1)
    assert s.metrics.ignored["duplicate_final"] == 1
    assert len(state["world"]["reservations"]) == 1
    assert len(await d.ledger("reserve_table")) == 1


async def test_use_case_4_correcting_a_reservation_updates_the_task_incrementally(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a table for 4")
    await d.idle()
    await d.say("make it six")
    state = await finish(d, acks=2)
    assert d.acks()[1].text == "Got it, switching to a table for 6…"
    assert len(state["world"]["reservations"]) == 1
    assert state["world"]["reservations"][0]["party_size"] == 6
    assert sorted(r["status"] for r in await d.ledger("reserve_table")) == [
        "COMMITTED", "COMPENSATED"]
