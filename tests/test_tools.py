import asyncio
import json
from datetime import date, timedelta

import pytest

import chronos.tools  # noqa: F401  (registers tools)
from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger, LedgerStatus
from chronos.coordination.snapshot import StateSnapshot
from chronos.protocol import PlanStatus
from chronos.tools.registry import REGISTRY, ToolExecutor, ToolKind
from chronos.tools.world import ToolError, World, normalize_time
from chronos.trace.logger import TraceLogger

TODAY = date(2026, 10, 1)
TOMORROW = (TODAY + timedelta(days=1)).isoformat()
NEXT_WEEK = (TODAY + timedelta(days=7)).isoformat()
COMMITTED = StateSnapshot(epoch=0, plan={"status": PlanStatus.COMMITTED.value})
DRAFT = StateSnapshot(epoch=0, plan={"status": PlanStatus.DRAFT.value})


class Env:
    def __init__(self, world, ledger, epochs, ex, tracer, sleeps):
        self.world, self.ledger, self.epochs, self.ex = world, ledger, epochs, ex
        self.tracer, self.sleeps = tracer, sleeps

    def trace_rows(self):
        return [json.loads(x) for x in self.tracer.path.read_text().splitlines()]

    def call(self, tool, epoch=None, **args):
        return REGISTRY.make_call(tool, args, self.epochs.current() if epoch is None else epoch)


@pytest.fixture
async def env(tmp_path):
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        await asyncio.sleep(0.02)  # short but real: lets interleavings happen

    tracer = TraceLogger("s", tmp_path)
    world = await World.create(seed=7, today=TODAY, sleep=fake_sleep)
    ledger = await IdempotencyLedger.open(trace=tracer.bind("coordination"))
    epochs = EpochManager("s", tracer.bind("coordination"))
    ex = ToolExecutor("s", world, epochs, ledger, trace=tracer.bind("tools"))
    yield Env(world, ledger, epochs, ex, tracer, sleeps)
    tracer.close()
    await world.close()
    await ledger.close()


BOOK_8PM = dict(dest="Delhi", date=TOMORROW, time="8pm", passenger="Asha")
BOOK_6PM = dict(dest="DEL", date=TOMORROW, time="6pm", passenger="Asha")


# ---------------------------------------------------------------- registry --------------------
def test_registry_contents():
    assert REGISTRY.names(ToolKind.READ) == sorted([
        "search_flights", "get_route", "check_table_availability", "get_booking",
        "lookup_troubleshooting_kb"])
    assert {"book_flight", "cancel_booking", "set_navigation", "reserve_table"} <= set(
        REGISTRY.names(ToolKind.WRITE))


def test_normalize_time():
    assert [normalize_time(x) for x in ("6pm", "18:00", "6:30 PM", "12am", "12pm", "07:05")] == [
        "18:00", "18:00", "18:30", "00:00", "12:00", "07:05"]
    with pytest.raises(ValueError):
        normalize_time("25:00")


async def test_kind_and_args_validation(env):
    r = await env.ex.execute_read(env.call("book_flight", dest="DEL"))
    assert not r.ok and r.error.startswith("not_a_read_tool")
    r = await env.ex.execute_write(env.call("search_flights", dest="DEL"), 0, COMMITTED)
    assert not r.ok and r.error.startswith("not_a_write_tool")
    r = await env.ex.execute_read(env.call("search_flights", dest="Atlantis"))
    assert not r.ok and r.error == "invalid_args"
    assert not (await env.ex.execute_read(env.call("nope"))).ok


# ------------------------------------------------------------- world invariants ---------------
async def test_seeded_latency_is_deterministic():
    async def run() -> list[float]:
        rec: list[float] = []

        async def s(x: float) -> None:
            rec.append(x)

        w = await World.create(seed=3, today=TODAY, sleep=s)
        for _ in range(5):
            await w.delay("get_route")
        await w.close()
        return rec

    a, b = await run(), await run()
    assert a == b and all(0.1 <= x <= 0.8 for x in a) and len(set(a)) > 1


async def test_no_booking_without_seat_under_concurrency(env):
    w = env.world
    f = (await w.search_flights("MAA", "DEL", TOMORROW, "20:00"))[0]
    await w.set_seats(f["flight_id"], 3)
    res = await asyncio.gather(*(
        w.book_flight("MAA", "DEL", TOMORROW, "20:00", f"p{i}", None) for i in range(20)),
        return_exceptions=True)
    ok = [r for r in res if isinstance(r, dict)]
    assert len(ok) == 3
    assert all(isinstance(r, ToolError) and "sold_out" in str(r) for r in res if r not in ok)
    assert len(await w.active_bookings()) == 3
    assert len(await w.charges("CHARGED")) == 3
    assert await w.check_invariants() == []


async def test_one_charge_per_booking_and_cancel_refunds_idempotently(env):
    w = env.world
    b = await w.book_flight("MAA", "DEL", TOMORROW, "20:00", "Asha", None)
    assert len(await w.charges()) == 1
    r1 = await w.cancel_booking(b["booking_id"])
    r2 = await w.cancel_booking(b["booking_id"])
    assert r1["already_cancelled"] is False and r2["already_cancelled"] is True
    assert [c["status"] for c in await w.charges()] == ["REFUNDED"]
    assert await w.check_invariants() == []
    assert (await w.get_booking(b["booking_id"]))["status"] == "CANCELLED"


async def test_failed_booking_leaves_no_partial_state(env):
    with pytest.raises(ToolError):
        await env.world.book_flight("MAA", "DEL", "2030-01-01", "20:00", "x", None)
    assert await env.world.charges() == [] and await env.world.check_invariants() == []


async def test_invariant_checker_detects_corruption(env):
    b = await env.world.book_flight("MAA", "DEL", TOMORROW, "20:00", "Asha", None)
    await env.world._db.execute("UPDATE charges SET status='REFUNDED'")  # corrupt on purpose
    await env.world._db.commit()
    assert any(b["booking_id"][3:].lstrip("0") in v for v in await env.world.check_invariants())


async def test_table_reservation_never_double_books_a_table(env):
    w = env.world
    res = await asyncio.gather(*(w.reserve_table(6, TOMORROW, "20:00", "Saffron Garden", "n")
                                 for _ in range(5)), return_exceptions=True)
    assert sum(isinstance(r, dict) for r in res) == 1  # only one table seats 6 there
    assert len(await w.active_reservations()) == 1


async def test_route_and_kb(env):
    r = await env.world.get_route("current location", "airport", "gas station")
    assert r["via"].startswith("Indian Oil") and len(r["waypoints"]) == 3 and r["eta_min"] > 0
    direct = await env.world.get_route("current location", "airport", None)
    assert direct["distance_km"] < r["distance_km"]
    with pytest.raises(ToolError):
        await env.world.get_route("here", "narnia", None)
    top = env.world.kb_lookup("machine is very hot, fan making grinding noise and it shut down")
    assert top[0]["id"] == "overheating_fan"
    assert env.world.kb_lookup("power cable is loose, led flickering")[0]["id"] == "loose_power_cable"
    assert env.world.kb_lookup("xyzzy")[0]["id"] == "no_match"


# ------------------------------------------------------------------- read tools --------------
async def test_read_cache_hits_and_shares_inflight(env):
    args = dict(dest="DEL", date=TOMORROW)
    n0 = len(env.sleeps)
    a, b = await asyncio.gather(env.ex.execute_read(env.call("search_flights", **args)),
                                env.ex.execute_read(env.call("search_flights", **args)))
    assert a.ok and b.ok and a.data == b.data
    assert len(env.sleeps) - n0 == 1 and sorted([a.cached, b.cached]) == [False, True]
    c = await env.ex.execute_read(env.call("search_flights", dest="Delhi", date=TOMORROW))
    assert c.cached  # different spelling, same canonical args
    assert len(env.sleeps) - n0 == 1
    d = await env.ex.execute_read(env.call("search_flights", dest="DEL", date=NEXT_WEEK))
    assert not d.cached


async def test_read_failure_not_cached_and_cancelled_caller_safe(env):
    r1 = await env.ex.execute_read(env.call("get_booking", booking_id="BK-0099"))
    r2 = await env.ex.execute_read(env.call("get_booking", booking_id="BK-0099"))
    assert not r1.ok and not r2.ok and not r2.cached
    t = asyncio.create_task(env.ex.execute_read(env.call("get_route", destination="airport")))
    await asyncio.sleep(0)
    t.cancel()
    later = await env.ex.execute_read(env.call("get_route", destination="airport"))
    assert later.ok


async def test_reads_are_traced_with_epoch(env):
    env.epochs.bump("x")
    await env.ex.execute_read(env.call("get_route", destination="airport"))
    row = next(r for r in env.trace_rows() if r["event"] == "tool_read")
    assert row["epoch"] == 1 and row["component"] == "tools" and row["tool"] == "get_route"


# ------------------------------------------------------------------ write guard --------------
async def test_write_blocked_not_committed(env):
    r = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, DRAFT)
    assert not r.ok and r.data["reason"] == "not_committed"
    assert await env.world.active_bookings() == []
    assert await env.ledger.committed("s") == []
    empty = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, StateSnapshot(0))
    assert empty.data["reason"] == "not_committed"


async def test_write_blocked_stale_epoch_precheck(env):
    call = env.call("book_flight", **BOOK_8PM)  # planned in epoch 0
    env.epochs.bump("correction")
    n_sleeps = len(env.sleeps)
    r = await env.ex.execute_write(call, 0, COMMITTED)
    assert not r.ok and r.data["reason"] == "stale_epoch"
    assert await env.world.active_bookings() == []
    assert len(env.sleeps) == n_sleeps  # rejected before any dispatch latency was spent
    assert await env.ledger.committed("s") == []
    ev = [x for x in env.trace_rows() if x["event"] == "write_blocked"]
    assert ev[0]["reason"] == "stale_epoch" and ev[0]["current_epoch"] == 1


async def test_fencing_blocks_write_when_epoch_changes_during_dispatch(env):
    t = asyncio.create_task(env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED))
    await asyncio.sleep(0.005)  # write is now past the pre-check, inside the latency window
    assert await env.ledger.status(
        env.ledger.make_key("s", "book_flight", {**BOOK_8PM, "origin": "MAA", "seat_pref": None,
                                                 "date": TOMORROW, "time": "20:00", "dest": "DEL"})
    ) == LedgerStatus.IN_FLIGHT
    env.epochs.bump("goal_change")
    r = await t
    assert not r.ok and r.data["reason"] == "stale_epoch"
    assert await env.world.active_bookings() == [] and await env.world.charges() == []
    assert await env.ledger.committed("s") == []
    # the claim was released (FAILED), so the same write is retryable in the new epoch
    r2 = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 1, COMMITTED)
    assert r2.ok


async def test_write_blocked_duplicate(env):
    r1 = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED)
    r2 = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED)
    assert r1.ok and not r2.ok and r2.data["reason"] == "duplicate"
    assert r2.data["existing"]["booking_id"] == r1.data["booking_id"]
    assert len(await env.world.active_bookings()) == 1
    assert len(await env.world.charges()) == 1
    assert "write_blocked_duplicate" in [x["event"] for x in env.trace_rows()]


async def test_equivalent_spellings_are_same_write(env):
    a = await env.ex.execute_write(env.call("book_flight", dest="Delhi", date=TOMORROW,
                                            time="8pm"), 0, COMMITTED)
    b = await env.ex.execute_write(env.call("book_flight", dest="DEL", date=TOMORROW,
                                            time="20:00"), 0, COMMITTED)
    assert a.ok and b.data["reason"] == "duplicate"


async def test_20_concurrent_identical_writes_one_booking(env):
    res = await asyncio.gather(*(
        env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED)
        for _ in range(20)))
    assert sum(r.ok for r in res) == 1
    assert all(r.data["reason"] == "duplicate" for r in res if not r.ok)
    assert len(await env.world.active_bookings()) == 1 and await env.world.check_invariants() == []


async def test_write_failure_is_reported_and_retryable(env):
    bad = dict(dest="DEL", date="2031-01-01", time="8pm")
    r = await env.ex.execute_write(env.call("book_flight", **bad), 0, COMMITTED)
    assert not r.ok and "no_such_flight" in r.error
    key = env.ledger.make_key("s", "book_flight", {"origin": "MAA", "dest": "DEL",
                                                   "date": "2031-01-01", "time": "20:00",
                                                   "passenger": "Guest", "seat_pref": None})
    assert await env.ledger.status(key) == LedgerStatus.FAILED


async def test_write_committed_clears_read_cache(env):
    args = dict(dest="DEL", date=TOMORROW, time="20:00")
    before = (await env.ex.execute_read(env.call("search_flights", **args))).data["flights"][0]
    await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED)
    after = await env.ex.execute_read(env.call("search_flights", **args))
    assert not after.cached and after.data["flights"][0]["seats_left"] == before["seats_left"] - 1


# ---------------------------------------------------------------- compensation ---------------
async def _commit(env, tool, **args):
    r = await env.ex.execute_write(env.call(tool, **args), env.epochs.current(), COMMITTED)
    assert r.ok, r
    return (await env.ledger.committed("s"))[-1]


async def test_compensate_book_flight(env):
    w = await _commit(env, "book_flight", **BOOK_8PM)
    r = await env.ex.compensate(w)
    assert r.ok and r.tool == "cancel_booking"
    assert await env.world.active_bookings() == []
    assert [c["status"] for c in await env.world.charges()] == ["REFUNDED"]
    assert await env.world.check_invariants() == []
    assert await env.ledger.status(w["key"]) == LedgerStatus.COMPENSATED
    again = await env.ex.compensate(w)  # idempotent
    assert again.ok and again.data.get("already_compensated")
    assert "write_compensated" in [x["event"] for x in env.trace_rows()]


async def test_compensate_set_navigation_and_reserve_table(env):
    nav = await _commit(env, "set_navigation", destination="airport")
    assert len(await env.world.active_navigation()) == 1
    assert (await env.ex.compensate(nav)).ok
    assert await env.world.active_navigation() == []

    rs = await _commit(env, "reserve_table", party_size=4, date=TOMORROW, time="8pm")
    assert len(await env.world.active_reservations()) == 1
    assert (await env.ex.compensate(rs)).ok
    assert await env.world.active_reservations() == []
    assert await env.world.check_invariants() == []
    # freed table can be booked again
    assert (await env.ex.execute_write(env.call("reserve_table", party_size=4, date=TOMORROW,
                                                time="8pm"), 0, COMMITTED)).ok


async def test_compensate_without_compensator_or_result(env):
    r = await env.ex.compensate({"key": "k", "tool": "cancel_booking", "args": {}, "epoch": 0,
                                 "result": {"x": 1}})
    assert not r.ok and r.error.startswith("no_compensator")
    r = await env.ex.compensate({"key": "k", "tool": "book_flight", "args": {}, "epoch": 0,
                                 "result": None})
    assert not r.ok


async def test_write_stale_after_commit_is_auto_compensated(env, monkeypatch):
    orig = env.world.book_flight

    async def bump_after_effect(*a, **k):
        out = await orig(*a, **k)
        env.epochs.bump("interrupt landed while the effect was landing")
        return out

    monkeypatch.setattr(env.world, "book_flight", bump_after_effect)
    r = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED)
    assert not r.ok and r.error == "stale_after_commit" and r.data["compensated"] is True
    assert await env.world.active_bookings() == []
    assert [c["status"] for c in await env.world.charges()] == ["REFUNDED"]
    assert await env.world.check_invariants() == []


async def test_compensate_stale_after_correction_yields_single_booking(env):
    """The headline scenario: 8pm committed in epoch 0, user corrects to 6pm."""
    first = await env.ex.execute_write(env.call("book_flight", **BOOK_8PM), 0, COMMITTED)
    assert first.ok
    env.epochs.bump("correction")
    comp = await env.ex.compensate_stale()
    assert len(comp) == 1 and comp[0].ok
    second = await env.ex.execute_write(env.call("book_flight", **BOOK_6PM), 1, COMMITTED)
    assert second.ok
    active = await env.world.active_bookings()
    assert len(active) == 1 and active[0]["flight_id"].endswith("1800")
    assert len(await env.world.charges("CHARGED")) == 1
    assert await env.world.check_invariants() == []
    assert await env.ex.compensate_stale() == []  # nothing left stale


async def test_compensate_stale_respects_keep_keys(env):
    w = await _commit(env, "book_flight", **BOOK_8PM)
    env.epochs.bump("addition-ish")
    assert await env.ex.compensate_stale(frozenset({w["key"]})) == []
    assert len(await env.world.active_bookings()) == 1
