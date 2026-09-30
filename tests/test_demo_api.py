"""The demo client's server side: static page, sample frames, and its exact WebSocket payloads."""
import base64
import json

import pytest
from fastapi.testclient import TestClient
from helpers import IMAGES
from test_api import make_app, read_until

from chronos.protocol import OutputMessage, OutputStatus, OutputType
from chronos.trace import export


@pytest.fixture
def client(fast_settings):
    with TestClient(make_app(fast_settings)) as c:
        yield c


def test_demo_page_serves_the_console(client):
    r = client.get("/demo")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "CHRONOS demo" in r.text
    js = client.get("/assets/app.js").text
    assert "/ws/" in js and "transcript_partial" in js


@pytest.mark.parametrize("route", ["/", "/demo", "/how-it-works", "/results", "/timeline"])
def test_site_pages_are_self_contained_and_never_inject_server_text_as_html(client, route):
    import re
    r = client.get(route)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert not re.search(r"""(src|href)\s*=\s*["']\s*(https?:)?//""", r.text)
    assert "innerHTML" not in r.text  # every server-provided string goes through textContent
    for asset in re.findall(r'(?:src|href)="(/assets/[^"]+)"', r.text):
        a = client.get(asset)
        assert a.status_code == 200, asset
        assert "innerHTML" not in a.text, asset


def test_results_api_serves_the_benchmark_output(client):
    r = client.get("/api/results")
    assert r.status_code == 200 and "chronos" in r.json()["modes"]["mock"]["agents"]


def test_traces_api_lists_sessions_newest_first(client):
    client.post("/sessions/tl-a/events?wait=true", json={"type": "text", "payload": {"text": "hello"}})
    rows = client.get("/api/traces").json()
    assert any(r["session_id"] == "tl-a" and r["events"] > 0 for r in rows)
    assert [r["modified"] for r in rows] == sorted((r["modified"] for r in rows), reverse=True)


def test_sample_images_are_served_as_png(client):
    for name in ("panel_disconnected_cable.png", "machine_overheating_fan.png",
                 "breaker_tripped.png"):
        r = client.get(f"/samples/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content == (IMAGES / name).read_bytes()


def test_sample_route_rejects_anything_that_could_escape_the_folder(client):
    assert client.get("/samples/nope.png").status_code == 404
    assert client.get("/samples/secret.txt").status_code == 400
    assert client.get("/samples/..%2Fpyproject.toml").status_code in (400, 404)
    assert client.get("/samples/..%5Cpyproject.png").status_code in (400, 404)


def test_camera_frame_exactly_as_the_web_client_sends_it(client):
    """web/index.html sends {b64, name, text} as a camera_frame event."""
    b64 = base64.b64encode((IMAGES / "panel_disconnected_cable.png").read_bytes()).decode()
    with client.websocket_connect("/ws/cam1") as ws:
        ws.send_json({"type": "camera_frame", "payload": {
            "b64": b64, "name": "panel_disconnected_cable.png",
            "text": "What's wrong with this machine?"}})
        msgs = read_until(ws, OutputType.RESPONSE)
    assert msgs[0].type is OutputType.ACK and msgs[-1].status is OutputStatus.DONE
    diag = msgs[-1].data["diagnosis"]
    assert diag["grounded"] is True and diag["kb_ids"][0] == "loose_power_cable"


def test_partial_stream_then_final_over_the_wire_like_simulated_speech(client):
    words = ["Book", "a", "table", "for", "4"]
    with client.websocket_connect("/ws/sp1") as ws:
        for i in range(len(words)):
            ws.send_json({"type": "transcript_partial",
                          "payload": {"text": " ".join(words[: i + 1])}})
        ws.send_json({"type": "transcript_final", "payload": {"text": " ".join(words)}})
        msgs = read_until(ws, OutputType.ACTION_RESULT)
    assert [m.type for m in msgs][0] is OutputType.ACK
    assert sum(m.type is OutputType.ACK for m in msgs) == 1  # one ack for the whole utterance
    state = client.get("/sessions/sp1/state").json()
    assert len(state["world"]["reservations"]) == 1


def test_explicit_interrupt_event_from_the_client(client):
    with client.websocket_connect("/ws/int1") as ws:
        ws.send_json({"type": "text", "payload": {"text": "Book a flight to Delhi tomorrow at 6pm"}})
        first = OutputMessage.model_validate(ws.receive_json())
        assert first.type is OutputType.ACK
        ws.send_json({"type": "interrupt", "payload": {"action": "cancel"}})
        msgs = read_until(ws, OutputType.RESPONSE)
    assert msgs[-1].status is OutputStatus.CANCELLED
    state = client.get("/sessions/int1/state").json()
    assert state["world"]["bookings"] == [] and state["epoch"] == 2


def test_live_trace_carries_task_ids_and_exports_to_a_standalone_page(client, tmp_path):
    client.post("/sessions/tr1/events?wait=true",
                json={"type": "text", "payload": {"text": "Book a flight to Delhi tomorrow at 6pm"}})
    text = client.get("/traces/tr1").text
    rows = [json.loads(x) for x in text.splitlines()]
    started = {r["task_id"]: r for r in rows if r["event"] == "task_started"}
    finished = {r["task_id"] for r in rows if r["event"] == "task_finished"}
    assert started and finished <= set(started)
    assert {r["task_id"] for r in rows if r["event"] == "task_started"} == finished  # all closed
    lanes = {r["component"] for r in rows}
    assert {"perception", "fast", "slow", "coordination", "tools"} <= lanes
    trace_file = tmp_path / "tr1.jsonl"
    trace_file.write_text(text, encoding="utf-8")
    assert export.main([str(trace_file)]) == 0
    assert 'id="model"' in (tmp_path / "tr1.html").read_text(encoding="utf-8")
    view = client.get("/traces/tr1/view")
    assert view.status_code == 200 and 'id="model"' in view.text and "livewrap" in view.text
