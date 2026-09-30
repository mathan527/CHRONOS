"""Runs the visualizer's DOM-free model script under Node (skipped when Node is not installed).

The model is what decides which spans are drawn cancelled, which lane a write marker lands in,
how overlapping bars stack, and what the summary chips say, so it is worth testing for real.
"""
import json
import re
import shutil
import subprocess

import pytest

from chronos.trace.export import TEMPLATE

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

RUNNER = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.match(/<script id="model">([\s\S]*?)<\/script>/)[1];
const api = vm.runInContext(src + "\n;({buildModel, parseJsonl, LANES})", vm.createContext({}));
const rows = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const jsonl = '{"a":1}\n\nnot json\n[1]\n{"b":2}\n';
console.log(JSON.stringify({ model: api.buildModel(rows), parsed: api.parseJsonl(jsonl),
                             empty: api.buildModel([]) }));
"""


def R(t, event, component, epoch, **kw):
    return {"t_ms": t, "event": event, "component": component, "epoch": epoch,
            "session_id": "s1", **kw}


# One scripted "goal change mid-route" session, with every kind of thing the chart must show.
TRACE = [
    R(0.0, "event_received", "perception", 1, type="transcript_final", text="Navigate"),
    R(1.0, "classified", "perception", 1, label="goal_change", latency_ms=0.07),
    R(2.0, "ack_sent", "fast", 1, text="On it", latency_ms=2.0),
    R(3.0, "task_started", "coordination", 1, task="slow:new:new", task_id=1, protected=False),
    R(3.5, "task_started", "coordination", 1, task="read:get_route", task_id=2, protected=False),
    R(40.0, "task_started", "coordination", 1, task="write:set_navigation", task_id=4, protected=True),
    R(50.0, "event_received", "perception", 1, type="transcript_final", text="Actually"),
    R(51.0, "epoch_bumped", "coordination", 2, old_epoch=1, reason="goal_change"),
    R(51.1, "task_cancelled", "coordination", 1, task="slow:new:new", task_id=1, superseded_by=2),
    R(51.2, "task_cancelled", "coordination", 1, task="read:get_route", task_id=2, superseded_by=2),
    R(51.3, "task_protected", "coordination", 1, task="write:set_navigation", task_id=4, superseded_by=2),
    R(52.0, "ack_sent", "fast", 2, text="Okay", latency_ms=1.5),
    R(53.0, "task_started", "coordination", 2, task="slow:bargein:goal_change", task_id=3, protected=False),
    R(54.0, "task_started", "coordination", 2, task="read:get_route", task_id=5, protected=False),
    R(90.0, "tool_read", "tools", 2, tool="get_route", cached=False, ok=True, latency_ms=30.0),
    R(91.0, "tool_read", "tools", 2, tool="get_route", cached=True, ok=True, latency_ms=0.0),
    R(95.0, "write_committed", "coordination", 2, tool="set_navigation", key="abc"),
    R(96.0, "task_finished", "coordination", 2, task="slow:bargein:goal_change", task_id=3, ok=True),
    R(97.0, "response_sent", "slow", 2, type="action_result", status="done", text="Navigation"),
    R(98.0, "write_blocked_duplicate", "coordination", 2, tool="set_navigation", key="abc"),
    R(99.0, "stale_result_dropped", "coordination", 1, current_epoch=2, what="action_result:done"),
    R(100.0, "write_compensated", "tools", 2, tool="book_flight"),
    R(101.0, "write_blocked", "tools", 2, tool="book_flight", reason="stale_epoch"),
    R(120.0, "task_finished", "coordination", 1, task="write:set_navigation", task_id=4, ok=True),
    R(121.0, "task_finished", "coordination", 2, task="orphan", task_id=99, ok=False),
    R(600.0, "ack_sent", "fast", 2, text="slow one", latency_ms=450.0),  # over the 300 ms budget
    R(140.0, "plan_committed", "slow", 2, steps=["get_route", "set_navigation"]),
]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("viz")
    (d / "run.js").write_text(RUNNER, encoding="utf-8")
    (d / "trace.json").write_text(json.dumps(TRACE), encoding="utf-8")
    r = subprocess.run([NODE, str(d / "run.js"), str(TEMPLATE), str(d / "trace.json")],
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def span(model, label, **want):
    hits = [s for s in model["spans"] if s["label"] == label
            and all(s.get(k) == v for k, v in want.items())]
    assert len(hits) == 1, (label, want, hits)
    return hits[0]


def test_javascript_in_the_template_parses_under_node():
    src = TEMPLATE.read_text(encoding="utf-8")
    for sid in ("model", "ui"):
        js = re.search(rf'<script id="{sid}">(.*?)</script>', src, re.DOTALL).group(1)
        r = subprocess.run([NODE, "--check", "-"], input=js.encode("utf-8"), capture_output=True)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")


def test_five_lanes_in_order(out):
    assert [(lane["id"], lane["label"]) for lane in out["model"]["lanes"]] == [
        ("perception", "Perception"), ("fast", "Fast path"), ("slow", "Slow path"),
        ("coordination", "Coordination"), ("tools", "Tools")]


def test_cancelled_tasks_are_marked_and_closed_at_the_cancel_time(out):
    m = out["model"]
    s = span(m, "slow:new:new")
    assert s["state"] == "cancelled" and s["lane"] == "slow" and s["epoch"] == 1
    assert s["start"] == 3.0 and s["end"] == 51.1 and s["supersededBy"] == 2
    r = span(m, "read:get_route", epoch=1)
    assert r["state"] == "cancelled" and r["lane"] == "tools" and r["end"] == 51.2
    # the epoch-2 task with the same name is a different span (paired by task_id)
    r2 = span(m, "read:get_route", epoch=2)
    assert r2["state"] == "running" and r2.get("open") is True and r2["end"] == m["tMax"]


def test_finished_protected_and_orphan_tasks(out):
    m = out["model"]
    done = span(m, "slow:bargein:goal_change")
    assert done["state"] == "done" and done["start"] == 53.0 and done["end"] == 96.0
    w = span(m, "write:set_navigation")
    assert w["protected"] is True and w["state"] == "done" and w["supersededBy"] == 2
    assert w["lane"] == "tools" and w["end"] == 120.0
    orphan = span(m, "orphan")
    assert orphan["state"] == "failed" and orphan["start"] == orphan["end"] == 121.0


def test_tool_reads_become_spans_or_cache_marks(out):
    m = out["model"]
    reads = [s for s in m["spans"] if s["kind"] == "read"]
    assert len(reads) == 1 and reads[0]["start"] == 60.0 and reads[0]["end"] == 90.0
    assert any(k["kind"] == "cachehit" and k["t"] == 91.0 for k in m["marks"])


def test_ack_spans_show_the_latency_and_flag_budget_overruns(out):
    m = out["model"]
    acks = sorted((s for s in m["spans"] if s["kind"] == "ack"), key=lambda s: s["start"])
    assert [round(s["end"] - s["start"], 3) for s in acks] == [2.0, 1.5, 450.0]  # bar = latency
    over = [s for s in acks if s.get("overBudget")]
    assert len(over) == 1 and over[0]["raw"]["end"]["latency_ms"] == 450.0
    assert m["summary"]["ackMax"] == 450.0 and m["summary"]["ackOverBudget"] == 1


def test_epoch_bump_is_a_labelled_vertical_line(out):
    (b,) = out["model"]["bumps"]
    assert (b["t"], b["old"], b["epoch"], b["reason"]) == (51.0, 1, 2, "goal_change")


def test_markers_land_in_the_right_lanes_with_the_right_kinds(out):
    marks = {(k["kind"], k["label"]): k for k in out["model"]["marks"]}
    assert marks[("duplicate", "write_blocked_duplicate")]["lane"] == "tools"  # not coordination
    assert marks[("commit", "write_committed")]["lane"] == "tools"
    assert marks[("comp", "write_compensated")]["lane"] == "tools"
    assert marks[("blocked", "write_blocked")]["lane"] == "tools"
    assert marks[("stale", "stale_result_dropped")]["lane"] == "coordination"
    assert marks[("stale", "stale_result_dropped")]["epoch"] == 1  # the epoch that was dropped
    assert marks[("protect", "task_protected")]["lane"] == "coordination"
    assert marks[("response", "response_sent")]["lane"] == "slow"
    assert marks[("plan", "plan_committed")]["lane"] == "slow"
    assert marks[("event", "event_received")]["lane"] == "perception"


def test_summary_counts(out):
    s = out["model"]["summary"]
    assert s["events"] == len(TRACE) and s["epochBumps"] == 1
    assert s["cancelledTasks"] == 2 and s["protectedTasks"] == 1
    assert s["staleDropped"] == 1 and s["duplicatesBlocked"] == 1
    assert s["writesBlocked"] == 1 and s["compensations"] == 1
    assert s["committed"] == {"set_navigation": 1}
    assert s["readsRun"] == 1 and s["readsCached"] == 1
    assert out["model"]["epochs"] == [1, 2] and out["model"]["sessionId"] == "s1"


def test_overlapping_spans_stack_into_rows_and_sequential_ones_share_a_row(out):
    slow = next(lane for lane in out["model"]["lanes"] if lane["id"] == "slow")
    tools = next(lane for lane in out["model"]["lanes"] if lane["id"] == "tools")
    assert slow["rows"] == 1  # the slow-path spans run one after another: one shared row
    assert tools["rows"] >= 3  # reads and the write overlap in time: they must stack
    for lane_id in ("slow", "tools"):
        spans = [s for s in out["model"]["spans"] if s["lane"] == lane_id]
        for a in spans:  # no two spans in the same row may overlap
            for b in spans:
                if a is not b and a["row"] == b["row"]:
                    assert a["end"] <= b["start"] or b["end"] <= a["start"], (a["label"], b["label"])


def test_jsonl_parser_skips_junk_and_counts_it(out):
    assert out["parsed"]["bad"] == 2 and out["parsed"]["rows"] == [{"a": 1}, {"b": 2}]


def test_empty_input_gives_an_empty_model(out):
    e = out["empty"]
    assert e["spans"] == [] and e["marks"] == [] and e["bumps"] == [] and e["tMax"] == 1
    assert e["summary"]["ackMax"] is None and e["sessionId"] is None
