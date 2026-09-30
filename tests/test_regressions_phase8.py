"""Bugs the benchmark's randomised scenarios and timing measurements exposed."""
import asyncio
import json
from datetime import date

from helpers import Driver, assert_no_stale_output, assert_world_consistent

from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger
from chronos.protocol import Event, EventType
from chronos.protocol import OutputType as T
from chronos.tools.registry import ToolExecutor
from chronos.tools.world import World
from chronos.trace.logger import TraceLogger

TODAY = date(2026, 10, 1)


async def test_a_stray_okay_in_the_middle_of_a_sentence_does_not_break_the_request(open_session):
    """Found by the benchmark: 'okay' arrived while the user was still speaking; it reset the
    'this partial stream is one request' flag, so the rest of the sentence was treated as an
    interruption, classified as hesitation and dropped. The route to 'Chennai' then failed."""
    s = await open_session()
    d = Driver(s)
    await d.partial("Navigate")
    await d.partial("Navigate to")
    await d.partial("Navigate to Chennai")
    await d.say("okay")  # a backchannel in the middle of the sentence
    await d.say("Navigate to Chennai Airport")
    await d.idle()
    assert s.metrics.ignored["backchannel"] == 1
    assert [r for r in d.events("ignored") if r["reason"] == "noise_during_speech"]
    state = await assert_world_consistent(d)
    assert len(state["world"]["navigation"]) == 1
    assert "chennai airport" in state["world"]["navigation"][0]["destination"].lower()
    assert [m for m in d.of(T.ACTION_RESULT)]
    assert_no_stale_output(d)


async def test_a_draft_that_is_only_waiting_for_the_user_accepts_a_fresh_request(open_session):
    """A blocked draft (no such place) is not running work: the user's next sentence is a new
    attempt, not something to ask the LLM to reinterpret as an interruption (which fell back to
    'hesitation' and dropped it)."""
    s = await open_session()
    d = Driver(s)
    await d.say("Navigate to Narnia")
    await d.idle()
    assert d.of(T.RESPONSE)[-1].data["reason"] == "route_failed"
    await d.say("Navigate to Chennai Airport")
    await d.idle()
    assert [c for c in s.llm.calls if c[0] == "bargein"] == []  # the LLM was never consulted
    state = await assert_world_consistent(d)
    assert len(state["world"]["navigation"]) == 1


async def test_executor_aclose_cancels_shared_reads_so_none_touches_a_closed_world():
    """Found by the benchmark: a speculative read still running at shutdown hit the closed
    database and logged 'Task exception was never retrieved'."""
    async def slow(_s: float) -> None:
        await asyncio.sleep(0.3)

    world = await World.create(":memory:", latency_ms=(1, 2), today=TODAY, sleep=slow)
    ledger = await IdempotencyLedger.open()
    epochs = EpochManager("x")
    ex = ToolExecutor("x", world, epochs, ledger)
    call = ex.registry.make_call("search_flights", {"dest": "DEL"}, epochs.current())
    caller = asyncio.create_task(ex.execute_read(call))
    await asyncio.sleep(0.02)  # the shared read is now in flight
    unhandled: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(lambda _l, ctx: unhandled.append(ctx))
    await ex.aclose()
    await world.close()  # what a shutdown does next
    await asyncio.gather(caller, return_exceptions=True)
    await asyncio.sleep(0.05)
    import gc
    gc.collect()
    assert unhandled == []
    await ledger.close()


def test_event_timestamps_have_sub_millisecond_resolution():
    """time.monotonic() ticks at ~15.6 ms on Windows, which made every ack latency read 0 and
    quantised the trace timeline. Timestamps now come from time.perf_counter()."""
    stamps = [Event(session_id="s", type=EventType.TEXT).ts_monotonic for _ in range(2000)]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) > 1000  # a 15.6 ms clock would give only a handful of distinct values


def test_trace_timestamps_are_not_quantised(tmp_path):
    t = TraceLogger("s", tmp_path)
    for i in range(300):
        t.emit("event_received", component="perception", epoch=1, i=i)
    t.close()
    ts = [json.loads(x)["t_ms"] for x in t.path.read_text(encoding="utf-8").splitlines()]
    assert ts == sorted(ts) and len(set(ts)) > 100


async def test_gas_station_first_after_the_route_was_set_changes_that_route(open_session):
    """Found by the benchmark's 'after' phase: once navigation had started, 'actually, gas
    station first' was treated as a brand-new request, lost the destination, and the old route
    stayed live with a 'where would you like to go?' prompt."""
    s = await open_session()
    d = Driver(s)
    await d.say("Navigate to Chennai Airport")
    await d.idle()
    assert len(d.of(T.ACTION_RESULT)) == 1 and s.planner.goal.status == "done"
    await d.say("Actually, gas station first")
    await d.idle()
    state = await assert_world_consistent(d)
    live = [n for n in state["world"]["navigation"]]
    assert len(live) == 1 and live[0]["via"] == "gas station"
    assert "chennai airport" in live[0]["destination"].lower()
    assert s.epochs.current() == 2
    assert sorted(r["status"] for r in await d.ledger("set_navigation")) == [
        "COMMITTED", "COMPENSATED"]
    assert not any(m.data.get("missing") for m in d.of(T.RESPONSE))  # it never asked "where to?"
    assert_no_stale_output(d)


async def test_has_target_treats_add_stop_after_navigation_as_a_change_but_not_other_goals():
    from chronos.perception.intent import Intent
    from chronos.protocol import BargeInType as B
    from chronos.slowpath.planner import Goal, SpeculativePlanner

    p = SpeculativePlanner.__new__(SpeculativePlanner)  # only has_target's inputs are needed
    p.policy = __import__("chronos.slowpath.planner", fromlist=["PlannerPolicy"]).PlannerPolicy()
    nav = Goal("g", Intent.NAVIGATE)
    nav.finish()
    p.goal = nav
    assert p.has_target(B.GOAL_CHANGE, Intent.ADD_STOP) and p.supersedes(B.GOAL_CHANGE, Intent.ADD_STOP)
    assert not p.has_target(B.GOAL_CHANGE, Intent.RESERVE_TABLE)  # a different task is a new request
    assert not p.has_target(B.GOAL_CHANGE)  # no hint: keep the conservative answer
    flight = Goal("f", Intent.BOOK_FLIGHT)
    flight.finish()
    p.goal = flight
    assert not p.has_target(B.GOAL_CHANGE, Intent.ADD_STOP)  # a stop makes no sense for a booking
