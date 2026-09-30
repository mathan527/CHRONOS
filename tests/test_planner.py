import asyncio
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

import chronos.tools  # noqa: F401  (registers tools)
from chronos.coordination.cancellation import CancellationManager
from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger, LedgerStatus
from chronos.coordination.snapshot import SnapshotStore
from chronos.perception.bargein import BargeInClassifier
from chronos.perception.intent import Intent, IntentExtractor
from chronos.protocol import BargeInType as B
from chronos.protocol import PlanStatus
from chronos.slowpath.llm import MockLLM
from chronos.slowpath.planner import PlannerPolicy, SpeculativePlanner
from chronos.slowpath.vision import MockVision
from chronos.tools.registry import ToolExecutor
from chronos.tools.world import World
from chronos.trace.logger import TraceLogger

TODAY = date(2026, 10, 1)
TOMORROW = (TODAY + timedelta(days=1)).isoformat()
NEXT_WEEK = (TODAY + timedelta(days=7)).isoformat()
IMAGES = Path(__file__).parent.parent / "scenarios" / "images"


class Env:
    pass


@pytest.fixture
async def env(tmp_path):
    e = Env()
    e.sleeps = []

    async def fake_sleep(s: float) -> None:
        e.sleeps.append(s)
        await asyncio.sleep(0.02)

    e.tracer = TraceLogger("s", tmp_path)
    e.world = await World.create(seed=7, today=TODAY, sleep=fake_sleep)
    e.ledger = await IdempotencyLedger.open(trace=e.tracer.bind("coordination"))
    e.epochs = EpochManager("s", e.tracer.bind("coordination"))
    e.cancel = CancellationManager(e.epochs, e.tracer.bind("coordination"))
    e.snaps = SnapshotStore("s")
    e.executor = ToolExecutor("s", e.world, e.epochs, e.ledger, trace=e.tracer.bind("tools"))
    e.extractor = IntentExtractor(MockLLM(), today=TODAY)
    e.llm, e.vision = MockLLM(), MockVision()
    e.new_planner = lambda **kw: SpeculativePlanner(
        "s", epochs=e.epochs, cancellation=e.cancel, snapshots=e.snaps, executor=e.executor,
        extractor=e.extractor, llm=kw.pop("llm", e.llm), vision=kw.pop("vision", e.vision),
        trace=e.tracer.bind("slow"), **kw)
    e.p = e.new_planner()
    e.rows = lambda: [json.loads(x) for x in e.tracer.path.read_text().splitlines()]
    e.events = lambda name: [r for r in e.rows() if r["event"] == name]
    e.reads = lambda tool: [r for r in e.events("tool_read") if r["tool"] == tool
                            and not r["cached"]]
    yield e
    await e.p.aclose()
    e.tracer.close()
    await e.world.close()
    await e.ledger.close()


async def settle(t: float = 0.1) -> None:
    await asyncio.sleep(t)


# ------------------------------------------------------------ speculation / drafting ----------
async def test_partial_starts_speculative_read_and_never_commits(env):
    plan = await env.p.on_partial("book a flight to Delhi")
    assert plan.status is PlanStatus.DRAFT and plan.intent is Intent.BOOK_FLIGHT
    assert plan.missing == ("date",) and plan.writes == ()
    assert [c.tool for c in plan.reads] == ["search_flights"]
    assert plan.reads[0].args["dest"] == "DEL" and plan.epoch == 0
    await settle()  # the read ran without any final transcript
    assert len(env.reads("search_flights")) == 1
    assert env.snaps.current().cached("search_flights", plan.reads[0].args) is not None
    assert env.snaps.current().plan["status"] == "draft"
    assert await env.ledger.committed("s") == []
    # a write attempted with the draft snapshot is refused by the executor's guard
    call = env.executor.registry.make_call(
        "book_flight", {"dest": "DEL", "date": TOMORROW, "time": "18:00"}, 0)
    r = await env.executor.execute_write(call, 0, env.snaps.current())
    assert r.data["reason"] == "not_committed"


async def test_partials_refine_the_draft_without_relaunching_identical_reads(env):
    await env.p.on_partial("book a flight to Delhi")
    await settle()
    await env.p.on_partial("book a flight to Delhi tomorrow")  # date arrives: new read args
    await env.p.on_partial("book a flight to Delhi tomorrow")  # identical again: nothing new
    await settle()
    assert len(env.reads("search_flights")) == 2  # (dest) and (dest, date) once each
    assert dict(env.snaps.current().slots) == {"dest": "DEL", "date": TOMORROW}


async def test_gibberish_partial_creates_no_plan(env):
    assert await env.p.on_partial("I want to boo...") is None
    assert env.p.goal is None and env.sleeps == []


# ------------------------------------------------------------------------ commit ------------
async def test_final_commits_infers_time_and_executes_once(env):
    await env.p.on_partial("book a flight to Delhi tomorrow")
    await settle()
    out = await env.p.on_final("book a flight to Delhi tomorrow")
    assert out.action == "committed" and out.plan.status is PlanStatus.COMMITTED
    (write,) = out.plan.writes
    assert write.tool == "book_flight" and write.args["time"] == "06:00"
    assert out.plan.inferred == ("time",)
    assert env.snaps.current().plan["status"] == "committed"
    assert len(env.reads("search_flights")) == 1  # the speculative read was reused, not repeated

    res = await env.p.execute(out.plan)
    assert res.ok and len(res.writes) == 1
    assert len(await env.world.active_bookings()) == 1
    assert env.p.goal.status == "done" and len(env.p.goal.write_keys) == 1
    assert env.p.plan.status is PlanStatus.DONE and env.snaps.current().plan["status"] == "done"
    assert await env.world.check_invariants() == []
    again = await env.p.execute(out.plan)  # a DONE goal's old plan cannot run again
    assert len(await env.world.active_bookings()) == 1 and again.ok in (True, False)


async def test_explicit_time_is_used_and_needs_a_matching_flight(env):
    out = await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    assert out.committed and out.plan.writes[0].args["time"] == "18:00" and out.plan.inferred == ()
    env2 = await env.p.on_final("book a flight to Delhi tomorrow at 7:17pm")
    assert env2.action in ("committed", "busy")  # first goal is still committed => busy


async def test_no_flight_at_requested_time_blocks(env):
    out = await env.p.on_final("book a flight to Delhi tomorrow at 7:17pm")
    assert out.action == "blocked" and out.note == "no_flight_at_time"
    assert out.plan.status is PlanStatus.DRAFT and (await env.p.execute(out.plan)).note == \
        "not_committed"


async def test_needs_slots_does_not_commit(env):
    out = await env.p.on_final("book a table")
    assert out.action == "needs_slots" and out.note == "party_size"
    assert out.plan.status is PlanStatus.DRAFT
    assert (await env.p.execute(out.plan)).note == "not_committed"
    assert await env.world.active_reservations() == []
    out = await env.p.on_final("book a flight tomorrow")
    assert out.action == "needs_slots" and out.note == "dest"


async def test_blocked_reasons(env):
    out = await env.p.on_final("navigate to Narnia")
    assert out.action == "blocked" and out.note == "route_failed"
    p2 = env.new_planner()
    env.p.goal = None
    out = await p2.on_final("cancel my booking BK-0042")
    assert out.action == "blocked" and out.note == "booking_not_found"


async def test_confirmation_policy_holds_the_write_until_confirm(env):
    p = env.new_planner(policy=PlannerPolicy(confirm_writes=frozenset({"book_flight"})))
    out = await p.on_final("book a flight to Delhi tomorrow at 6pm")
    assert out.action == "needs_confirmation" and out.note == "book_flight"
    assert out.plan.status is PlanStatus.DRAFT and env.snaps.current().plan["status"] == "draft"
    assert (await p.execute(out.plan)).note == "not_committed"
    assert await env.world.active_bookings() == []
    out = await p.confirm()
    assert out.committed and (await p.execute(out.plan)).ok
    assert len(await env.world.active_bookings()) == 1


async def test_commit_is_abandoned_if_an_interruption_lands_while_reading(env):
    t = asyncio.create_task(env.p.on_final("book a flight to Delhi tomorrow"))
    await asyncio.sleep(0.005)  # reads are in flight
    env.epochs.bump("something else")
    out = await t
    assert out.action == "stale" and out.plan.status is PlanStatus.DRAFT
    assert env.snaps.current().plan["status"] != "committed"
    assert env.events("stale_result_dropped")
    assert await env.ledger.committed("s") == []


# --------------------------------------------------------------------- corrections ----------
async def test_time_correction_reuses_speculative_read_and_bumps_epoch(env):
    await env.p.on_partial("book a flight to Delhi tomorrow at 8pm")
    await settle()
    out = await env.p.handle_bargein(B.CORRECTION, "make it 6pm", final=False)
    assert out.action == "corrected" and out.epoch == 1 and env.epochs.current() == 1
    assert out.updates == {"time": "18:00"}
    assert out.reads_reused == 1 and out.reads_started == 0 and out.replanned == []
    assert dict(env.snaps.current().slots) == {"dest": "DEL", "date": TOMORROW, "time": "18:00"}
    assert out.plan.epoch == 1 and out.plan.reads[0].epoch == 1
    assert len(env.reads("search_flights")) == 1
    done = await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    assert done.committed and done.plan.writes[0].args["time"] == "18:00"


async def test_date_correction_replans_only_the_affected_read(env):
    await env.p.on_partial("book a flight to Delhi tomorrow")
    await settle()
    out = await env.p.handle_bargein(B.CORRECTION, "actually, next week instead", final=False)
    assert out.updates == {"date": NEXT_WEEK}
    assert out.reads_reused == 0 and out.replanned == ["search_flights"]
    await settle()
    assert len(env.reads("search_flights")) == 2
    # the obsolete (date=tomorrow) entry was invalidated by the patch; the new one is cached
    cached_dates = {e.args["date"] for e in env.snaps.current().read_cache.values()}
    assert cached_dates == {NEXT_WEEK}


async def test_inflight_read_is_cancelled_at_bump_but_its_work_is_shared(env):
    await env.p.on_partial("book a flight to Delhi tomorrow")  # read starts, still running
    old_tasks = env.cancel.active(epoch=0)
    assert old_tasks
    out = await env.p.handle_bargein(B.CORRECTION, "make it 6pm", final=False)
    await env.cancel.join_cancelled()
    assert all(t.cancelled() for t in old_tasks)  # the epoch-0 task really was cancelled
    cancelled = env.events("task_cancelled")
    assert any(c["task"] == "read:search_flights" and c["epoch"] == 0 for c in cancelled)
    assert out.reads_started == 1  # re-requested in epoch 1...
    await settle()
    assert len(env.sleeps) == 1  # ...but the world was only hit once: the read was shared


async def test_correction_after_committed_booking_cancels_it_and_rebooks(env):
    """Headline scenario: 8pm is booked, the user says make it 6pm."""
    first = await env.p.on_final("book a flight to Delhi tomorrow at 8pm")
    assert (await env.p.execute(first.plan)).ok
    (b0,) = await env.world.active_bookings()
    assert b0["flight_id"].endswith("2000")

    out = await env.p.handle_bargein(B.CORRECTION, "make it 6pm", final=True)
    assert out.epoch == 1 and out.commit.committed
    assert len(out.compensated) == 1 and out.compensated[0].ok
    assert (await env.p.execute(out.plan)).ok

    active = await env.world.active_bookings()
    assert len(active) == 1 and active[0]["flight_id"].endswith("1800")
    assert len(await env.world.charges("CHARGED")) == 1
    assert len(await env.world.charges("REFUNDED")) == 1
    assert await env.world.check_invariants() == []
    assert [e["reason"] for e in env.events("epoch_bumped")] == ["correction"]
    assert env.events("write_compensated")
    assert len(env.p.goal.write_keys) == 1  # only the live booking is tracked now


async def test_no_double_booking_when_correction_races_the_write(env):
    plan = (await env.p.on_final("book a flight to Delhi tomorrow at 8pm")).plan
    exec_task = asyncio.create_task(env.p.execute(plan))
    await asyncio.sleep(0.005)  # write is dispatching
    out = await env.p.handle_bargein(B.CORRECTION, "make it 6pm", final=True)
    old = await exec_task
    assert not old.ok and old.stale  # fenced (or compensated): never leaves an 8pm booking
    assert (await env.p.execute(out.plan)).ok
    active = await env.world.active_bookings()
    assert len(active) == 1 and active[0]["flight_id"].endswith("1800")
    assert await env.world.check_invariants() == []
    assert any(r["event"] == "task_protected" for r in env.rows())


async def test_unparseable_correction_still_bumps_but_changes_nothing(env):
    await env.p.on_partial("book a flight to Delhi tomorrow")
    before = dict(env.snaps.current().slots)
    out = await env.p.handle_bargein(B.CORRECTION, "the other one", final=True)
    assert out.epoch == 1 and out.note == "no_slot_change_understood" and out.commit is None
    assert dict(env.snaps.current().slots) == before


# ------------------------------------------------------------------- goal change ------------
async def test_goal_change_mid_route_calculation(env):
    """In-car: navigate -> (route still calculating) -> gas station first."""
    await env.p.on_partial("navigate to Chennai Airport")  # epoch-0 route read in flight
    out = await env.p.handle_bargein(B.GOAL_CHANGE, "actually, gas station first")
    assert out.epoch == 1 and out.action == "goal_changed" and out.commit.committed
    assert out.plan.intent is Intent.NAVIGATE
    assert out.plan.slots["destination"] == "Chennai Airport" and out.plan.slots["via"]
    await env.cancel.join_cancelled()
    assert any(c["task"] == "read:get_route" and c["epoch"] == 0
               for c in env.events("task_cancelled"))
    assert (await env.p.execute(out.plan)).ok
    (nav,) = await env.world.active_navigation()
    assert "Guindy" in nav["route_json"]  # routed via the gas station
    committed_nav = [w for w in await env.ledger.committed("s") if w["tool"] == "set_navigation"]
    assert len(committed_nav) == 1 and committed_nav[0]["epoch"] == 1


async def test_goal_change_while_old_write_is_dispatching_yields_one_navigation(env):
    plan = (await env.p.on_final("navigate to Chennai Airport")).plan  # committed in epoch 0
    exec_task = asyncio.create_task(env.p.execute(plan))
    await asyncio.sleep(0.005)
    out = await env.p.handle_bargein(B.GOAL_CHANGE, "actually, gas station first")
    old = await exec_task
    assert not old.ok and old.stale
    assert (await env.p.execute(out.plan)).ok
    assert len(await env.world.active_navigation()) == 1
    assert [w["epoch"] for w in await env.ledger.committed("s")
            if w["tool"] == "set_navigation"] == [1]


async def test_goal_change_after_finished_goal_is_just_a_new_request(env):
    first = await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    await env.p.execute(first.plan)
    out = await env.p.handle_bargein(B.GOAL_CHANGE, "actually reserve a table for 3")
    assert env.epochs.current() == 0 and out.committed  # no bump, no undo
    assert out.compensated == []
    assert len(await env.world.active_bookings()) == 1  # the flight stays booked
    await env.p.execute(out.plan)
    assert len(await env.world.active_reservations()) == 1


async def test_unintelligible_goal_change_keeps_old_goal(env):
    first = await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    await env.p.execute(first.plan)
    env.p.goal.status = "committed"  # pretend still running
    out = await env.p.handle_bargein(B.GOAL_CHANGE, "xyzzy plugh")
    assert out.action == "clarify" and out.compensated == []
    assert len(await env.world.active_bookings()) == 1  # nothing was undone


# ------------------------------------------------------------------------- cancel ----------
async def test_cancel_undoes_committed_work_and_cancels_the_plan(env):
    out = await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    await env.p.execute(out.plan)
    c = await env.p.handle_bargein(B.CANCEL, "stop")
    assert c.action == "cancelled" and c.epoch == 1 and len(c.compensated) == 1
    assert c.plan.status is PlanStatus.CANCELLED and env.p.goal.status == "cancelled"
    assert env.snaps.current().plan["status"] == "cancelled"
    assert await env.world.active_bookings() == []
    assert len(await env.world.charges("REFUNDED")) == 1
    assert await env.world.check_invariants() == []
    assert (await env.p.execute(out.plan)).stale  # the old plan can never run again


async def test_cancel_a_draft_cancels_speculative_work_without_writes(env):
    await env.p.on_partial("book a flight to Delhi tomorrow")
    c = await env.p.handle_bargein(B.CANCEL, "never mind")
    await env.cancel.join_cancelled()
    assert c.action == "cancelled" and c.compensated == []
    assert env.events("task_cancelled")
    assert await env.ledger.committed("s") == []


async def test_cancel_with_nothing_to_cancel_is_a_noop(env):
    c = await env.p.handle_bargein(B.CANCEL, "stop")
    assert c.action == "noop" and env.epochs.current() == 0


# --------------------------------------------------------------- non-bumping labels ---------
async def test_addition_patches_incrementally_without_bumping(env):
    await env.p.on_partial("book a flight to Delhi tomorrow")
    await settle()
    out = await env.p.handle_bargein(B.ADDITION, "and add a window seat", final=False)
    assert out.action == "added" and env.epochs.current() == 0
    assert out.updates == {"seat_pref": "window"}
    assert out.reads_reused == 1 and out.reads_started == 0
    done = await env.p.on_final("book a flight to Delhi tomorrow")
    assert done.committed and done.plan.writes[0].args["seat_pref"] == "window"


async def test_addition_after_commit_never_creates_a_second_write(env):
    out = await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    await env.p.execute(out.plan)
    add = await env.p.handle_bargein(B.ADDITION, "and add a window seat")
    assert add.action == "addition_needs_replan" and add.updates == {"seat_pref": "window"}
    assert env.epochs.current() == 0 and len(await env.world.active_bookings()) == 1
    assert len([w for w in await env.ledger.committed("s") if w["tool"] == "book_flight"]) == 1


@pytest.mark.parametrize("label,action", [
    (B.BACKCHANNEL, "noop"), (B.HESITATION, "noop"), (B.CLARIFICATION_QUESTION, "answer_question")])
async def test_non_interrupting_labels_change_nothing(env, label, action):
    await env.p.on_partial("book a flight to Delhi tomorrow")
    before, goal, plan = dict(env.snaps.current().slots), env.p.goal, env.p.plan
    out = await env.p.handle_bargein(label, "okay")
    assert out.action == action and env.epochs.current() == 0
    assert env.p.goal is goal and env.p.plan is plan
    assert dict(env.snaps.current().slots) == before and not env.events("epoch_bumped")


async def test_accessibility_hesitation_then_self_correction_one_reservation(env):
    clf = BargeInClassifier(MockLLM(), today=TODAY)
    for text in ("I want to boo...", "um..."):
        r = await clf.classify(text, final=False)
        assert r.label is B.HESITATION
        await env.p.on_partial(text)  # rules find no intent: nothing is drafted
        assert (await env.p.handle_bargein(r.label, text, final=False)).action == "noop"
    assert env.sleeps == [] and env.epochs.current() == 0 and env.p.goal is None
    text = "actually, book a table for 4"
    r = await clf.classify(text, final=True)
    final = await env.p.handle_bargein(r.label, text, final=True)
    assert env.sleeps  # ...only the real request did its reads
    assert final.committed and final.plan.slots["party_size"] == 4
    assert env.epochs.current() == 0  # nothing to supersede => no bump
    assert (await env.p.execute(final.plan)).ok
    assert len(await env.world.active_reservations()) == 1
    assert len([w for w in await env.ledger.committed("s") if w["tool"] == "reserve_table"]) == 1


async def test_reservation_correction_updates_current_task_incrementally(env):
    await env.p.on_partial("book a table for 4")
    await settle()
    out = await env.p.handle_bargein(B.CORRECTION, "make it for six", final=True)
    assert out.updates == {"party_size": 6} and out.commit.committed
    assert (await env.p.execute(out.plan)).ok
    (res,) = await env.world.active_reservations()
    assert res["party_size"] == 6 and len(await env.world.active_reservations()) == 1


# --------------------------------------------------------------- vision + diagnosis ---------
@pytest.mark.parametrize("image,kb_id,word", [
    ("panel_disconnected_cable.png", "loose_power_cable", "cable"),
    ("machine_overheating_fan.png", "overheating_fan", "fan"),
    ("breaker_tripped.png", "tripped_breaker", "breaker")])
async def test_troubleshooting_is_grounded_in_the_camera_frame(env, image, kb_id, word):
    env.p.on_camera_frame({"path": str(IMAGES / image)})
    await env.p.on_partial("what's wrong with this machine")
    await settle()
    assert env.vision.calls == [image]  # vision started speculatively, before the final
    out = await env.p.on_final("What's wrong with this machine?")
    assert out.action == "diagnosis" and env.vision.calls == [image]  # not repeated
    d = out.data["diagnosis"]
    assert d["kb_ids"][0] == kb_id and d["grounded"] is True and d["source"] == "llm"
    assert word in d["observed"].lower() and word in d["diagnosis"].lower()
    assert d["steps"] and env.p.goal.status == "done"
    assert env.events("frame_described") and env.events("diagnosis_ready")


async def test_diagnosis_without_a_frame_is_not_claimed_grounded(env):
    out = await env.p.on_final("what's wrong with this machine, it is very hot and the fan is loud")
    d = out.data["diagnosis"]
    assert d["observed"] is None and d["grounded"] is False
    assert "no camera image" in d["diagnosis"].lower() and d["kb_ids"][0] == "overheating_fan"
    assert env.vision.calls == []


async def test_invalid_llm_diagnosis_falls_back_but_stays_grounded(env):
    bad = MockLLM(handlers={"diagnose": lambda _p: {"diagnosis": "", "steps": []}})
    p = env.new_planner(llm=bad)
    p.on_camera_frame({"path": str(IMAGES / "panel_disconnected_cable.png")})
    out = await p.on_final("what's wrong with this machine")
    d = out.data["diagnosis"]
    assert d["source"] == "fallback" and d["grounded"] and d["kb_ids"][0] == "loose_power_cable"


async def test_slow_vision_times_out_and_agent_still_answers(env):
    slow = MockVision(latency_s=5)
    p = env.new_planner(vision=slow, vision_timeout_s=0.05)
    p.on_camera_frame({"path": str(IMAGES / "breaker_tripped.png")})
    t0 = time.perf_counter()
    out = await p.on_final("what's wrong with this machine, the breaker keeps tripping")
    assert time.perf_counter() - t0 < 2
    d = out.data["diagnosis"]
    assert d["observed"] is None and d["grounded"] is False and d["kb_ids"][0] == "tripped_breaker"


async def test_latest_frame_wins(env):
    env.p.on_camera_frame({"path": str(IMAGES / "breaker_tripped.png")})
    env.p.on_camera_frame({"path": str(IMAGES / "machine_overheating_fan.png")})
    out = await env.p.on_final("what's wrong with this machine")
    assert env.vision.calls == ["machine_overheating_fan.png"]
    assert out.data["diagnosis"]["kb_ids"][0] == "overheating_fan"


# ---------------------------------------------------------------- other intents -------------
async def test_smalltalk_and_cancel_booking(env):
    assert (await env.p.on_final("hello there")).action == "smalltalk"
    booked = await env.world.book_flight("MAA", "DEL", TOMORROW, "18:00", "Asha", None)
    p = env.new_planner()
    out = await p.on_final(f"cancel my flight booking {booked['booking_id']}")
    assert out.committed and (await p.execute(out.plan)).ok
    assert await env.world.active_bookings() == []
    assert await env.world.check_invariants() == []


async def test_busy_when_a_committed_goal_is_running(env):
    await env.p.on_final("book a flight to Delhi tomorrow at 6pm")
    out = await env.p.on_final("navigate to the airport")
    assert out.action == "busy"


async def test_ledger_rows_reflect_compensation_states(env):
    first = await env.p.on_final("book a flight to Delhi tomorrow at 8pm")
    res = await env.p.execute(first.plan)
    key = res.writes[0].idempotency_key
    assert await env.ledger.status(key) == LedgerStatus.COMMITTED
    await env.p.handle_bargein(B.CANCEL, "cancel that")
    assert await env.ledger.status(key) == LedgerStatus.COMPENSATED


async def test_recommitting_never_duplicates_the_write_step(env):
    p = env.new_planner(policy=PlannerPolicy(confirm_writes=frozenset({"book_flight"})))
    await p.on_final("book a flight to Delhi tomorrow at 6pm")
    out = await p.confirm()
    assert [c.tool for c in out.plan.steps] == ["search_flights", "book_flight"]
    out2 = await p._try_commit()  # e.g. a retried commit
    assert [c.tool for c in out2.plan.steps] == ["search_flights", "book_flight"]
