import asyncio
import json

import pytest

from chronos.coordination.cancellation import CancellationManager
from chronos.coordination.canonical import idempotency_key
from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger, LedgerStatus
from chronos.coordination.snapshot import ReadEntry, SnapshotStore, StateSnapshot
from chronos.trace.logger import TraceLogger


def rows(tracer: TraceLogger) -> list[dict]:
    return [json.loads(line) for line in tracer.path.read_text().splitlines()]


# ---------------------------------------------------------------- epoch -----------------------
def test_epoch_bump_monotonic_and_traced(tmp_path):
    tr = TraceLogger("s", tmp_path)
    em = EpochManager("s", tr.bind("coordination"))
    assert em.current() == 0 and em.is_current(0)
    assert em.bump("correction") == 1
    assert em.bump("goal_change") == 2
    assert em.is_current(2) and not em.is_current(1)
    tr.close()
    bumps = [r for r in rows(tr) if r["event"] == "epoch_bumped"]
    assert [(b["old_epoch"], b["epoch"], b["reason"]) for b in bumps] == [
        (0, 1, "correction"), (1, 2, "goal_change")]


# ------------------------------------------------------------ cancellation --------------------
async def test_bump_cancels_older_running_tasks():
    em = EpochManager("s")
    cm = CancellationManager(em)
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(10)

    old = cm.spawn(0, slow(), "old")
    await started.wait()
    new_epoch = em.bump("goal_change")
    fresh = cm.spawn(new_epoch, asyncio.sleep(0.01, result="ok"), "fresh")
    await cm.join_cancelled()
    assert old.cancelled()
    assert await fresh == "ok"


async def test_cancel_only_older_epochs_and_traced(tmp_path):
    tr = TraceLogger("s", tmp_path)
    em = EpochManager("s", tr.bind("coordination"))
    cm = CancellationManager(em, tr.bind("coordination"))
    t0 = cm.spawn(0, asyncio.sleep(10), "e0")
    t1 = cm.spawn(1, asyncio.sleep(10), "e1")  # already ahead of the epoch we bump to
    em.bump("x")  # -> 1 : cancels e0 only
    await cm.join_cancelled()
    assert t0.cancelled() and not t1.done()
    em.bump("y")  # -> 2 : cancels e1
    await cm.join_cancelled()
    assert t1.cancelled()
    tr.close()
    cancelled = [r for r in rows(tr) if r["event"] == "task_cancelled"]
    assert [(c["task"], c["epoch"]) for c in cancelled] == [("e0", 0), ("e1", 1)]


async def test_protected_write_task_not_cancelled():
    em = EpochManager("s")
    cm = CancellationManager(em)
    write = cm.spawn(0, asyncio.sleep(0.05, result="written"), "write", protected=True)
    em.bump("correction")
    await cm.join_cancelled()
    assert await write == "written" and not write.cancelled()


async def test_stale_result_dropped_never_emitted(tmp_path):
    tr = TraceLogger("s", tmp_path)
    em = EpochManager("s", tr.bind("coordination"))
    cm = CancellationManager(em, tr.bind("coordination"))
    emitted: list = []

    async def work(epoch: int):
        await asyncio.sleep(0.05)  # finishes "after" the bump; ignores cancellation on purpose
        return f"result-e{epoch}"

    async def shielded_worker(epoch: int):
        # Simulates work that can't be interrupted (e.g. a thread) and then tries to emit.
        result = await asyncio.shield(asyncio.ensure_future(work(epoch)))
        cm.emit_if_current(epoch, result, emitted.append, what="plan")

    t = asyncio.ensure_future(shielded_worker(1))
    em.bump("a")  # epoch 1
    em.bump("b")  # epoch 2; worker started under epoch 1 is now stale
    await asyncio.gather(t)
    assert emitted == []
    assert cm.dropped_stale == 1
    # A current result does get through.
    assert cm.emit_if_current(2, "fresh", emitted.append) is True
    assert emitted == ["fresh"]
    tr.close()
    drops = [r for r in rows(tr) if r["event"] == "stale_result_dropped"]
    assert len(drops) == 1 and drops[0]["epoch"] == 1 and drops[0]["current_epoch"] == 2


# ---------------------------------------------------------------- snapshot --------------------
def test_patch_preserves_unchanged_slots_and_valid_cache():
    base = StateSnapshot(
        epoch=1, intent="book_flight",
        slots={"dest": "DEL", "date": "tomorrow", "time": "8pm"},
        read_cache={},
    )
    search = ReadEntry("search_flights", {"dest": "DEL"}, {"n": 3}, deps={"dest": "DEL"})
    with_time = ReadEntry("search_flights", {"dest": "DEL", "time": "8pm"}, {"n": 1},
                          deps={"dest": "DEL", "time": "8pm"})
    s1 = base.patch({"cache_add": [search, with_time]})
    assert len(s1.read_cache) == 2

    s2 = s1.patch({"slots": {"time": "6pm"}}, epoch=2)
    assert s2.epoch == 2 and s2.intent == "book_flight"
    assert dict(s2.slots) == {"dest": "DEL", "date": "tomorrow", "time": "6pm"}
    assert s2.cached("search_flights", {"dest": "DEL"}) == search  # still valid -> reused
    assert s2.cached("search_flights", {"dest": "DEL", "time": "8pm"}) is None  # invalidated
    # old snapshot untouched
    assert s1.slots["time"] == "8pm" and s1.epoch == 1 and len(s1.read_cache) == 2


def test_patch_removes_slot_with_none_and_replaces_plan():
    s = StateSnapshot(epoch=0, slots={"a": 1, "b": 2}, plan={"steps": ["x"]})
    p = s.patch({"slots": {"a": None}, "plan": {"steps": ["y"]}})
    assert dict(p.slots) == {"b": 2} and p.plan["steps"] == ("y",)


def test_snapshot_is_immutable():
    s = StateSnapshot(epoch=0, slots={"a": {"n": [1, 2]}})
    with pytest.raises(AttributeError):
        s.epoch = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        s.slots["a"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        s.slots["a"]["n"] = 0  # type: ignore[index]


def test_store_history_and_at_epoch():
    st = SnapshotStore("s")
    st.apply({"intent": "navigate", "slots": {"dest": "airport"}}, epoch=1)
    st.apply({"slots": {"via": "gas"}}, epoch=2)
    assert len(st.history()) == 3
    assert st.current().epoch == 2 and dict(st.current().slots) == {
        "dest": "airport", "via": "gas"}
    assert "via" not in st.at_epoch(1).slots
    assert st.at_epoch(7) is None


# ------------------------------------------------------------------ ledger --------------------
KEY = IdempotencyLedger.make_key("s", "book_flight", {"dest": "DEL", "date": "2026-10-01"})


def test_key_is_canonical_and_session_scoped():
    a = idempotency_key("s", "t", {"x": 1, "y": [1, 2]})
    assert a == idempotency_key("s", "t", {"y": [1, 2], "x": 1})
    assert a != idempotency_key("s2", "t", {"x": 1, "y": [1, 2]})
    assert idempotency_key("ab", "c", {}) != idempotency_key("a", "bc", {})
    assert len(a) == 64


async def test_50_concurrent_identical_writes_execute_exactly_once():
    ledger = await IdempotencyLedger.open()
    executed = 0

    async def attempt() -> bool:
        nonlocal executed
        async with ledger.guard(KEY, session_id="s", tool="book_flight", args={}) as ok:
            if ok:
                executed += 1
                await asyncio.sleep(0.05)  # in-flight window where the others pile up
                ok.set_result({"booking": "B1"})
                return True
        return False

    results = await asyncio.gather(*(attempt() for _ in range(50)))
    assert executed == 1 and sum(results) == 1
    assert await ledger.status(KEY) == LedgerStatus.COMMITTED
    assert await ledger.result(KEY) == {"booking": "B1"}
    await ledger.close()


async def test_duplicate_blocked_is_traced(tmp_path):
    tr = TraceLogger("s", tmp_path)
    ledger = await IdempotencyLedger.open(trace=tr.bind("coordination"))
    for _ in range(2):
        async with ledger.guard(KEY, session_id="s", tool="book_flight", epoch=1) as ok:
            if ok:
                ok.set_result({})
    tr.close()
    events = [r["event"] for r in rows(tr)]
    assert events == ["write_committed", "write_blocked_duplicate"]
    await ledger.close()


async def test_failed_write_can_retry_and_exception_propagates():
    ledger = await IdempotencyLedger.open()
    with pytest.raises(RuntimeError):
        async with ledger.guard(KEY, session_id="s", tool="t") as ok:
            assert ok
            raise RuntimeError("boom")
    assert await ledger.status(KEY) == LedgerStatus.FAILED
    async with ledger.guard(KEY, session_id="s", tool="t") as ok:
        assert ok  # retry allowed
    assert await ledger.status(KEY) == LedgerStatus.COMMITTED
    await ledger.close()


async def test_cancelled_write_marked_failed_not_stuck_in_flight():
    ledger = await IdempotencyLedger.open()

    async def w():
        async with ledger.guard(KEY, session_id="s", tool="t") as ok:
            assert ok
            await asyncio.sleep(10)

    t = asyncio.create_task(w())
    await asyncio.sleep(0.02)
    assert await ledger.status(KEY) == LedgerStatus.IN_FLIGHT
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert await ledger.status(KEY) == LedgerStatus.FAILED
    await ledger.close()


async def test_compensated_key_can_be_rewritten_and_committed_lists():
    ledger = await IdempotencyLedger.open()
    async with ledger.guard(KEY, session_id="s", tool="book_flight", args={"a": 1}, epoch=1) as ok:
        ok.set_result({"booking": "B1"})
    assert [c["tool"] for c in await ledger.committed("s")] == ["book_flight"]
    assert await ledger.mark_compensated(KEY) is True
    assert await ledger.mark_compensated(KEY) is False  # only COMMITTED -> COMPENSATED
    assert await ledger.committed("s") == []
    async with ledger.guard(KEY, session_id="s", tool="book_flight") as ok:
        assert ok
    await ledger.close()


async def test_ledger_survives_restart(tmp_path):
    db = str(tmp_path / "ledger.db")
    l1 = await IdempotencyLedger.open(db)
    async with l1.guard(KEY, session_id="s", tool="book_flight", args={"dest": "DEL"}) as ok:
        assert ok
        ok.set_result({"booking": "B1"})
    await l1.close()

    l2 = await IdempotencyLedger.open(db)  # "restart"
    assert await l2.status(KEY) == LedgerStatus.COMMITTED
    async with l2.guard(KEY, session_id="s", tool="book_flight") as ok:
        assert not ok and ok.existing == LedgerStatus.COMMITTED
    assert await l2.result(KEY) == {"booking": "B1"}
    await l2.close()


async def test_two_connections_same_file_still_exactly_once(tmp_path):
    db = str(tmp_path / "shared.db")
    a, b = await IdempotencyLedger.open(db), await IdempotencyLedger.open(db)
    executed = 0

    async def attempt(ledger: IdempotencyLedger) -> None:
        nonlocal executed
        async with ledger.guard(KEY, session_id="s", tool="t") as ok:
            if ok:
                executed += 1
                await asyncio.sleep(0.05)

    await asyncio.gather(*(attempt(a if i % 2 else b) for i in range(20)))
    assert executed == 1
    await a.close()
    await b.close()


def test_canonical_hash_same_for_frozen_and_plain_args():
    from chronos.coordination.snapshot import freeze
    args = {"dest": "DEL", "opts": {"seat": "window"}, "pax": [1, 2]}
    assert idempotency_key("s", "t", args) == idempotency_key("s", "t", freeze(args))
