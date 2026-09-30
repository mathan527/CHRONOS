"""Server support for the demo UI: the opt-in trace stream, the `utterance` trace row, assets."""
import asyncio
import json

from helpers import Driver
from starlette.testclient import TestClient
from test_api import make_app

from chronos.trace.logger import TraceLogger


def _send(ws, sid: str, type_: str, **payload) -> None:
    ws.send_text(json.dumps({"session_id": sid, "type": type_, "payload": payload}))


def _collect(ws, until_type: str, limit: int = 80) -> list[dict]:
    frames: list[dict] = []
    for _ in range(limit):
        frames.append(ws.receive_json())
        if frames[-1].get("type") == until_type:
            return frames
    raise AssertionError(f"never saw {until_type}: {[f.get('type') or f.get('kind') for f in frames]}")


def test_the_trace_stream_is_opt_in(fast_settings):
    with TestClient(make_app(fast_settings)) as c:
        with c.websocket_connect("/ws/plain") as ws:
            _send(ws, "plain", "transcript_final", text="Navigate to Chennai Airport")
            frames = _collect(ws, "action_result")
        assert all("kind" not in f for f in frames)  # nothing but protocol messages

        with c.websocket_connect("/ws/traced?trace=1") as ws:
            _send(ws, "traced", "transcript_final", text="Navigate to Chennai Airport")
            frames = _collect(ws, "action_result")
        rows = [f["row"] for f in frames if f.get("kind") == "trace"]
        outputs = [f for f in frames if "kind" not in f]
        assert rows and outputs
        assert {"event_received", "utterance", "ack_sent", "task_started", "tool_read"} <= {
            r["event"] for r in rows}
        assert all({"event", "component", "epoch", "t_ms"} <= r.keys() for r in rows)
        assert [f["type"] for f in outputs][0] == "ack"  # outputs are unchanged and in order


def test_a_trace_listener_that_raises_does_not_break_tracing(tmp_path):
    t = TraceLogger("x", tmp_path)
    seen: list[str] = []

    def bad(_row):
        raise RuntimeError("boom")

    t.subscribe(bad)
    unsub = t.subscribe(lambda r: seen.append(r["event"]))
    t.emit("event_received", component="perception", epoch=1)
    unsub()
    t.emit("event_received", component="perception", epoch=1)
    t.close()
    assert seen == ["event_received"]
    assert len(t.path.read_text(encoding="utf-8").splitlines()) == 2


async def test_every_utterance_leaves_a_classification_row(open_session):
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow")
    await asyncio.sleep(0.03)
    await d.say("okay")  # a backchannel while the agent is busy
    await d.say("Actually, next week instead")  # a correction
    await d.idle()
    rows = d.events("utterance")
    by_text = {r["text"]: r for r in rows}
    new = by_text["Book a flight to Delhi tomorrow"]
    assert new["label"] == "new_request" and new["acted"] is True
    noise = by_text["okay"]
    assert noise["label"] == "backchannel" and noise["acted"] is False
    fix = by_text["Actually, next week instead"]
    assert fix["label"] == "correction" and fix["acted"] is True and fix["tier"] == "rules"
    assert all(isinstance(r["classify_ms"], float) and r["classify_ms"] >= 0 for r in rows)


async def test_an_explicit_hesitation_in_a_partial_is_traced_once_and_never_acted_on(open_session):
    s = await open_session()
    d = Driver(s)
    await d.partial("Navigate to")  # ordinary mid-sentence piece: not worth showing as a hesitation
    await d.partial("I want to boo…")
    await d.partial("I want to boo…")  # repeated partial: traced once
    await d.partial("I want to boo… um…")
    await asyncio.sleep(0.02)
    rows = [r for r in d.events("utterance") if r["partial"]]
    assert [r["text"] for r in rows] == ["I want to boo…", "I want to boo… um…"]
    assert all(r["label"] == "hesitation" and r["acted"] is False for r in rows)
    assert {r["note"] for r in rows} == {"trailing_ellipsis"}
    assert s.epochs.current() == 1 and not d.events("epoch_bumped")
    await d.say("actually, book a table for 4")
    await d.idle()
    finals = [r for r in d.events("utterance") if not r["partial"]]
    assert finals and finals[-1]["acted"] is True
