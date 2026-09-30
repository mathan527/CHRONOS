"""Unit tests for the small pieces Phase 6 added underneath the session."""
import asyncio
import time
from datetime import date

import pytest

import chronos.tools  # noqa: F401  (registers tools)
from chronos.coordination.cancellation import CancellationManager
from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger
from chronos.coordination.snapshot import SnapshotStore
from chronos.events.queue import EventQueue
from chronos.perception.bargein import BargeInClassifier
from chronos.perception.intent import Intent, IntentExtractor, find_party
from chronos.protocol import BargeInType as B
from chronos.protocol import Event, EventType
from chronos.slowpath.llm import MockLLM
from chronos.slowpath.planner import PlannerPolicy, SpeculativePlanner
from chronos.tools.registry import ToolExecutor
from chronos.tools.world import World

TODAY = date(2026, 10, 1)


def ev(t: EventType, n: int = 0) -> Event:
    return Event(session_id="s", type=t, payload={"n": n})


# --------------------------------------------------------------------------- queue -----------
async def test_take_where_removes_matches_and_preserves_everyone_elses_order():
    q = EventQueue()
    for i, t in enumerate([EventType.TEXT, EventType.CAMERA_FRAME, EventType.TRANSCRIPT_FINAL,
                           EventType.CAMERA_FRAME, EventType.TEXT, EventType.INTERRUPT]):
        await q.put(ev(t, i))
    frames = q.take_where(lambda e: e.type is EventType.CAMERA_FRAME)
    assert [e.payload["n"] for e in frames] == [1, 3]  # arrival order
    rest = q.drain()
    assert [e.payload["n"] for e in rest] == [5, 2, 0, 4]  # priority, then FIFO, untouched
    assert q.take_where(lambda _e: True) == []


# ----------------------------------------------------------------------- classifier ----------
async def test_llm_can_be_disabled_for_an_idle_agent():
    llm = MockLLM()
    clf = BargeInClassifier(llm, today=TODAY)
    r = await clf.classify("the flight was nice last time", allow_llm=False)
    assert r.tier == 0 and r.reason == "fallback:llm_disabled" and llm.calls == []
    assert (await clf.classify("stop", allow_llm=False)).tier == 1  # rules still work


# ------------------------------------------------------------------------ intent -----------
def test_bare_answers_only_count_when_the_agent_asked():
    e = IntentExtractor(today=TODAY)
    assert e.slot_updates("Delhi", Intent.BOOK_FLIGHT) == {}
    assert e.slot_updates("Delhi", Intent.BOOK_FLIGHT, ("dest",)) == {"dest": "DEL"}
    assert e.slot_updates("Chennai Airport", Intent.NAVIGATE, ("destination",)) == {
        "destination": "Chennai Airport"}
    assert e.slot_updates("make it faster", Intent.NAVIGATE) == {}  # never invents a destination
    assert e.slot_updates("four", Intent.RESERVE_TABLE) == {"party_size": 4}
    assert e.slot_updates("four", Intent.BOOK_FLIGHT) == {}


def test_party_size_after_make_it_but_never_a_time():
    assert [find_party(t) for t in ("make it six", "change it to 5", "make it for six")] == [6, 5, 6]
    assert [find_party(t) for t in ("make it 6pm", "make it 7:30", "make it 6 pm please")] == [
        None, None, None]


# ------------------------------------------------------------------------ planner ----------
@pytest.fixture
async def rig(tmp_path):
    async def fast(_s: float) -> None:
        await asyncio.sleep(0.005)

    class R:
        pass

    r = R()
    r.world = await World.create(today=TODAY, sleep=fast)
    r.ledger = await IdempotencyLedger.open()
    r.epochs = EpochManager("s")
    r.cancel = CancellationManager(r.epochs)
    r.snaps = SnapshotStore("s")
    r.exec = ToolExecutor("s", r.world, r.epochs, r.ledger)
    r.mk = lambda **kw: SpeculativePlanner(
        "s", epochs=r.epochs, cancellation=r.cancel, snapshots=r.snaps, executor=r.exec,
        extractor=IntentExtractor(MockLLM(), today=TODAY), llm=MockLLM(), **kw)
    yield r
    await r.world.close()
    await r.ledger.close()


async def test_has_target_and_supersedes_by_goal_state(rig):
    p = rig.mk()
    for label in B:  # no goal at all: nothing to act on
        assert not p.has_target(label) and not p.supersedes(label)
    out = await p.on_final("book a flight to Delhi tomorrow at 6pm")  # committed, not executed
    assert out.committed
    assert p.has_target(B.CORRECTION) and p.supersedes(B.CORRECTION)
    assert p.has_target(B.ADDITION) and not p.supersedes(B.ADDITION)  # additions never bump
    assert p.has_target(B.CLARIFICATION_QUESTION) and not p.supersedes(B.CLARIFICATION_QUESTION)
    assert not p.has_target(B.BACKCHANNEL) and not p.has_target(B.HESITATION)
    await p.execute(out.plan)
    assert p.goal.status == "done"
    assert p.has_target(B.CANCEL) and p.supersedes(B.CANCEL)  # inside the undo window
    assert not p.has_target(B.GOAL_CHANGE)  # a new objective after completion is a new request


async def test_finished_goal_leaves_the_undo_window(rig):
    p = rig.mk(policy=PlannerPolicy(undo_window_s=0.05))
    out = await p.on_final("book a flight to Delhi tomorrow at 6pm")
    await p.execute(out.plan)
    assert p.supersedes(B.CORRECTION)
    await asyncio.sleep(0.08)
    for label in (B.CORRECTION, B.CANCEL, B.ADDITION, B.GOAL_CHANGE):
        assert not p.has_target(label)
    assert not p.has_target(B.CLARIFICATION_QUESTION)  # a question after completion is new work
    n = len(await rig.world.active_bookings())
    res = await p.handle_bargein(B.CANCEL, "stop")
    assert res.action == "noop" and len(await rig.world.active_bookings()) == n


async def test_pre_bumped_epoch_is_not_bumped_twice(rig):
    p = rig.mk()
    await p.on_partial("book a flight to Delhi tomorrow")
    assert rig.epochs.current() == 0
    rig.epochs.bump("correction")  # what the session does, synchronously, before the slow task
    out = await p.handle_bargein(B.CORRECTION, "make it 6pm", final=False, bumped=True)
    assert rig.epochs.current() == 1 and out.epoch == 1
    out2 = await p.handle_bargein(B.CORRECTION, "make it 7pm", final=False)  # not pre-bumped
    assert rig.epochs.current() == 2 and out2.epoch == 2


async def test_compensation_survives_a_later_epoch_bump(rig):
    """A cancelled slow task that was awaiting an undo must not abort the undo half-way."""
    p = rig.mk()
    out = await p.on_final("book a flight to Delhi tomorrow at 6pm")
    await p.execute(out.plan)
    assert len(await rig.world.active_bookings()) == 1
    task = asyncio.create_task(p.handle_bargein(B.CANCEL, "cancel that"))
    await asyncio.sleep(0.002)  # compensation is now running as a protected task
    task.cancel()  # the slow-path task itself is superseded...
    await asyncio.gather(task, return_exceptions=True)
    rig.epochs.bump("another interruption")
    for _ in range(50):  # ...but the protected compensation still completes
        if not await rig.world.active_bookings():
            break
        await asyncio.sleep(0.01)
    assert await rig.world.active_bookings() == []
    assert await rig.world.check_invariants() == []


async def test_clarification_and_noise_labels_never_bump(rig):
    p = rig.mk()
    await p.on_final("book a flight to Delhi tomorrow at 6pm")
    t0 = time.monotonic()
    for label in (B.BACKCHANNEL, B.HESITATION, B.CLARIFICATION_QUESTION):
        await p.handle_bargein(label, "whatever")
    assert rig.epochs.current() == 0 and time.monotonic() - t0 < 1
