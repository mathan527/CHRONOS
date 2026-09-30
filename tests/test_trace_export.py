import json
import re
import subprocess
import sys
from pathlib import Path

from chronos.trace import export
from chronos.trace.logger import TraceLogger

VIS = Path(export.TEMPLATE)


def make_trace(tmp_path, extra_text: str = "hello") -> Path:
    t = TraceLogger("demo1", tmp_path)
    t.emit("event_received", component="perception", epoch=1, type="text", text=extra_text)
    t.emit("ack_sent", component="fast", epoch=1, text="On it", latency_ms=1.5)
    t.emit("epoch_bumped", component="coordination", epoch=2, old_epoch=1, reason="goal_change")
    t.emit("write_committed", component="coordination", epoch=2, tool="set_navigation", key="abc")
    t.close()
    return t.path


def embedded_rows(page: str) -> list[dict]:
    """Recover the embedded JSON the way the browser's JS engine would read it."""
    m = re.search(r"window\.__CHRONOS_TRACE__ = (\[.*?\]);\n", page, re.DOTALL)
    assert m, "no embedded trace array"
    return json.loads(m.group(1))  # the escapes used are plain JSON escapes


# ------------------------------------------------------------------ the template ------------
def test_template_has_the_marker_once_and_is_fully_self_contained():
    page = VIS.read_text(encoding="utf-8")
    assert page.count(export.MARKER) == 1
    assert not re.search(r"""(src|href)\s*=\s*["']\s*(https?:)?//""", page)  # no CDN, no fonts
    assert not re.search(r"@import|url\(\s*['\"]?https?:", page)
    assert page.count("<script") == 2 and page.count("</script>") == 2  # model + ui, nothing else
    assert 'id="model"' in page and 'id="ui"' in page


def test_template_names_the_five_lanes_and_the_required_features():
    page = VIS.read_text(encoding="utf-8")
    for label in ("Perception", "Fast path", "Slow path", "Coordination", "Tools"):
        assert label in page
    for feature in ("hatch", "line-through", "epoch_bumped", "write_blocked_duplicate",
                    "dragover", "/traces/", "JSON.stringify"):
        assert feature in page, feature


def test_model_script_touches_no_dom():
    page = VIS.read_text(encoding="utf-8")
    model = re.search(r'<script id="model">(.*?)</script>', page, re.DOTALL).group(1)
    for banned in ("document.", "window.", "querySelector", "addEventListener", "innerHTML"):
        assert banned not in model, banned


# ------------------------------------------------------------------------- export -----------
def test_export_embeds_the_trace_and_writes_next_to_it(tmp_path, capsys):
    path = make_trace(tmp_path)
    assert export.main([str(path)]) == 0
    out = path.with_suffix(".html")
    page = out.read_text(encoding="utf-8")
    assert export.MARKER not in page and "<title>CHRONOS trace demo1</title>" in page
    rows = embedded_rows(page)
    assert [r["event"] for r in rows] == ["event_received", "ack_sent", "epoch_bumped",
                                          "write_committed"]
    assert rows[2]["reason"] == "goal_change" and rows[3]["tool"] == "set_navigation"
    assert "wrote" in capsys.readouterr().out
    assert not re.search(r"""(src|href)\s*=\s*["']\s*(https?:)?//""", page)  # still offline


def test_export_explicit_output_path(tmp_path):
    path = make_trace(tmp_path)
    out = tmp_path / "nested_name.html"
    assert export.main([str(path), "-o", str(out)]) == 0 and out.exists()


def test_embedded_data_cannot_break_out_of_the_script_element(tmp_path):
    evil = "</script><script>alert(1)</script><!-- x   y"
    path = make_trace(tmp_path, extra_text=evil)
    export.main([str(path)])
    page = path.with_suffix(".html").read_text(encoding="utf-8")
    assert page.count("</script>") == 2  # still only the template's own two
    assert page.count("<script") == 2 and "<!--" not in page  # nothing from the data survives
    assert "alert(1)" in page  # ...but the text itself is there, just inert (<...)
    assert " " not in page
    assert embedded_rows(page)[0]["text"] == evil  # ...and the data survives intact


def test_export_error_cases(tmp_path, capsys):
    assert export.main([str(tmp_path / "missing.jsonl")]) == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    assert export.main([str(empty)]) == 1
    junk = tmp_path / "junk.jsonl"
    junk.write_text('not json\n{"event":"ack_sent","t_ms":1,"epoch":1}\n[1,2]\n', encoding="utf-8")
    assert export.main([str(junk)]) == 0
    err = capsys.readouterr().err
    assert "skipped 2 line(s)" in err
    assert len(embedded_rows(junk.with_suffix(".html").read_text(encoding="utf-8"))) == 1


def test_command_line_entry_point(tmp_path):
    path = make_trace(tmp_path)
    r = subprocess.run([sys.executable, "-m", "chronos.trace.export", str(path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert path.with_suffix(".html").exists() and "wrote" in r.stdout
    r2 = subprocess.run([sys.executable, "-m", "chronos.trace.export", str(tmp_path / "nope")],
                        capture_output=True, text=True, timeout=60)
    assert r2.returncode == 2 and "no such trace file" in r2.stderr
