"""The trace must carry enough to draw bars: unique task ids and an end for every task."""
import asyncio
import json

from chronos.coordination.cancellation import CancellationManager
from chronos.coordination.epoch import EpochManager
from chronos.trace.logger import TraceLogger


def rows(tracer: TraceLogger) -> list[dict]:
    return [json.loads(x) for x in tracer.path.read_text(encoding="utf-8").splitlines()]


def make(tmp_path):
    tracer = TraceLogger("s", tmp_path)
    epochs = EpochManager("s", tracer.bind("coordination"), start=1)
    return tracer, epochs, CancellationManager(epochs, tracer.bind("coordination"))


async def test_started_and_finished_are_paired_by_task_id(tmp_path):
    tracer, _e, cm = make(tmp_path)
    t = cm.spawn(1, asyncio.sleep(0.01, result="x"), "read:get_route")
    await t
    tracer.close()
    started, finished = [r for r in rows(tracer) if r["event"] in ("task_started", "task_finished")]
    assert started["event"] == "task_started" and finished["event"] == "task_finished"
    assert started["task_id"] == finished["task_id"] and started["task"] == "read:get_route"
    assert finished["ok"] is True and finished["error"] is None and finished["duration_ms"] >= 5
    assert finished["epoch"] == 1 and finished["t_ms"] >= started["t_ms"]


async def test_cancelled_task_is_closed_by_task_cancelled_not_task_finished(tmp_path):
    tracer, epochs, cm = make(tmp_path)
    t = cm.spawn(1, asyncio.sleep(10), "slow:new:new")
    epochs.bump("goal_change")
    await cm.join_cancelled()
    assert t.cancelled()
    tracer.close()
    evs = [(r["event"], r.get("task_id")) for r in rows(tracer) if r["event"].startswith("task_")]
    assert evs == [("task_started", 1), ("task_cancelled", 1)]  # no task_finished for it


async def test_failing_task_reports_ok_false_and_its_error(tmp_path):
    tracer, _e, cm = make(tmp_path)

    async def boom():
        raise RuntimeError("nope")

    t = cm.spawn(1, boom(), "write:book_flight")
    await asyncio.gather(t, return_exceptions=True)
    tracer.close()
    fin = next(r for r in rows(tracer) if r["event"] == "task_finished")
    assert fin["ok"] is False and fin["error"] == "RuntimeError"


async def test_task_ids_are_unique_and_protected_tasks_carry_the_id(tmp_path):
    tracer, epochs, cm = make(tmp_path)
    w = cm.spawn(1, asyncio.sleep(0.03), "write:x", protected=True)
    plain = [cm.spawn(1, asyncio.sleep(0.01), f"read:{i}") for i in range(3)]
    epochs.bump("correction")
    await asyncio.gather(w, *plain, return_exceptions=True)
    tracer.close()
    rs = rows(tracer)
    ids = [r["task_id"] for r in rs if r["event"] == "task_started"]
    assert len(ids) == len(set(ids)) == 4
    protected = next(r for r in rs if r["event"] == "task_protected")
    assert protected["task_id"] == ids[0] and protected["superseded_by"] == 2
    assert next(r for r in rs if r["event"] == "task_finished" and r["task_id"] == ids[0])["ok"]
