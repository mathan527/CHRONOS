import json

from chronos.trace.logger import TraceLogger


def test_trace_file_valid_jsonl(tmp_path):
    t = TraceLogger("sess1", tmp_path)
    t.emit("event_received", component="perception", epoch=0, type="text")
    t.bind("coordination").emit("epoch_bumped", epoch=1, reason="correction")
    t.emit("tool_read", component="tools", epoch=1, tool="get_route", args={"d": "x"})
    t.close()
    assert t.path == tmp_path / "sess1.jsonl"
    lines = t.path.read_text().splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    for r in rows:
        assert {"event", "component", "epoch", "t_ms", "mono_ms", "session_id"} <= set(r)
        assert r["session_id"] == "sess1"
    assert [r["event"] for r in rows] == ["event_received", "epoch_bumped", "tool_read"]
    assert rows[1]["component"] == "coordination" and rows[1]["epoch"] == 1
    assert rows[0]["t_ms"] <= rows[1]["t_ms"] <= rows[2]["t_ms"]
    assert rows[2]["args"] == {"d": "x"}


def test_sessions_write_separate_files(tmp_path):
    a, b = TraceLogger("a", tmp_path), TraceLogger("b", tmp_path)
    a.emit("event_received", component="x", epoch=0)
    b.emit("event_received", component="x", epoch=0)
    a.close(); b.close()
    assert len(a.path.read_text().splitlines()) == 1
    assert len(b.path.read_text().splitlines()) == 1
