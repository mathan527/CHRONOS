import dataclasses
import json

import httpx
import pytest
from conftest import TODAY
from fastapi.testclient import TestClient

from chronos.agent.session import AgentSession
from chronos.api.server import create_app
from chronos.protocol import OutputMessage, OutputStatus, OutputType

TAGS_OK = {"models": [{"name": "llama3.2:3b"}, {"name": "moondream:latest"}]}


def ollama_transport(payload=TAGS_OK, status=200):
    return httpx.MockTransport(lambda _r: httpx.Response(status, json=payload))


def make_app(settings, **kw):
    async def factory(sid: str) -> AgentSession:
        return await AgentSession.create(sid, settings, today=TODAY)

    return create_app(settings, session_factory=factory,
                      ollama_transport=kw.pop("ollama_transport", ollama_transport()), **kw)


@pytest.fixture
def client(fast_settings):
    with TestClient(make_app(fast_settings)) as c:
        yield c


def read_until(ws, stop_type: OutputType, limit: int = 12) -> list[OutputMessage]:
    msgs: list[OutputMessage] = []
    for _ in range(limit):
        m = OutputMessage.model_validate(ws.receive_json())  # every frame must be protocol-valid
        msgs.append(m)
        if m.type is stop_type:
            return msgs
    raise AssertionError(f"never saw {stop_type}: {[m.type for m in msgs]}")


# --------------------------------------------------------------------------- websocket ------
def test_websocket_streams_protocol_compliant_messages(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_json({"type": "text", "payload": {"text": "Book a flight to Delhi tomorrow at 6pm"}})
        msgs = read_until(ws, OutputType.ACTION_RESULT)
    assert [m.type for m in msgs] == [OutputType.ACK, OutputType.PROGRESS, OutputType.ACTION_RESULT]
    assert [m.status for m in msgs] == [
        OutputStatus.PENDING, OutputStatus.COMMITTED, OutputStatus.DONE]
    assert {m.session_id for m in msgs} == {"s1"} and {m.epoch for m in msgs} == {1}
    assert msgs[0].text.startswith("On it") and "Booked" in msgs[2].text
    state = client.get("/sessions/s1/state").json()
    assert len(state["world"]["bookings"]) == 1 and state["metrics"]["stale_emitted"] == 0


def test_websocket_barge_in_over_the_wire(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_json({"type": "text", "payload": {"text": "Book a flight to Delhi tomorrow"}})
        read_until(ws, OutputType.ACTION_RESULT)
        ws.send_json({"type": "transcript_final", "payload": {"text": "Actually, next week instead"}})
        msgs = read_until(ws, OutputType.ACTION_RESULT, limit=12)
        msgs += read_until(ws, OutputType.ACTION_RESULT, limit=12)
    assert msgs[0].type is OutputType.ACK and msgs[0].epoch == 2
    assert msgs[0].text == "Got it, switching to next week…"
    assert any(m.status is OutputStatus.CANCELLED for m in msgs)  # the earlier booking undone
    state = client.get("/sessions/s1/state").json()
    assert len(state["world"]["bookings"]) == 1 and state["epoch"] == 2


def test_bad_input_gets_a_protocol_error_and_the_connection_survives(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_text("this is not json")
        err = OutputMessage.model_validate(ws.receive_json())
        assert err.type is OutputType.ERROR and err.status is OutputStatus.FAILED
        ws.send_json({"type": "no_such_type", "payload": {}})
        assert OutputMessage.model_validate(ws.receive_json()).type is OutputType.ERROR
        ws.send_json({"type": "text", "payload": {}, "session_id": "someone-else"})
        assert "does not match" in OutputMessage.model_validate(ws.receive_json()).text
        ws.send_json({"type": "text", "payload": {"text": "hello there"}})  # still alive
        msgs = read_until(ws, OutputType.RESPONSE)
    assert msgs[0].type is OutputType.ACK


def test_a_client_cannot_forge_the_epoch(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_json({"type": "text", "epoch": 99, "ts_monotonic": 0.0,
                      "payload": {"text": "hello there"}})
        msgs = read_until(ws, OutputType.RESPONSE)
    assert {m.epoch for m in msgs} == {1}
    assert client.get("/sessions/s1/state").json()["metrics"]["ack_ms"]["max"] < 300


def test_bad_session_id_is_rejected_on_websocket(client):
    with pytest.raises(Exception):  # noqa: B017 - Starlette raises on close code 1008
        with client.websocket_connect("/ws/bad id!") as ws:
            ws.receive_json()


# ------------------------------------------------------------------------ HTTP fallback -----
def test_http_fallback_with_wait_returns_the_outputs(client):
    r = client.post("/sessions/h1/events?wait=true",
                    json={"type": "text", "payload": {"text": "Book a table for 4"}})
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] and body["idle"] is True and body["epoch"] == 1
    outs = [OutputMessage.model_validate(m) for m in body["outputs"]]
    assert [m.type for m in outs] == [OutputType.ACK, OutputType.PROGRESS, OutputType.ACTION_RESULT]
    state = client.get("/sessions/h1/state").json()
    assert len(state["world"]["reservations"]) == 1
    assert state["ledger"][0]["tool"] == "reserve_table" and state["ledger"][0]["status"] == "COMMITTED"


def test_http_fallback_without_wait_is_202_immediately(client):
    r = client.post("/sessions/h2/events", json={"type": "text", "payload": {"text": "hello there"}})
    assert r.status_code == 202 and "outputs" not in r.json()


def test_http_invalid_event_is_422_with_an_error_message(client):
    r = client.post("/sessions/h3/events", json={"type": "bogus"})
    assert r.status_code == 422
    err = OutputMessage.model_validate(r.json())
    assert err.type is OutputType.ERROR and err.data["detail"]


def test_state_of_unknown_session_is_404(client):
    assert client.get("/sessions/nope/state").status_code == 404


def test_session_ids_are_validated(client):
    assert client.get("/sessions/a%20b/state").status_code == 400
    assert client.post("/sessions/a%20b/events", json={"type": "text"}).status_code == 400
    assert client.get("/traces/a%20b").status_code == 400
    assert client.get("/traces/a%20b/view").status_code == 400
    assert client.get("/traces/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)


# --------------------------------------------------------------------------- traces --------
def test_trace_endpoints(client):
    client.post("/sessions/t1/events?wait=true",
                json={"type": "text", "payload": {"text": "Book a flight to Delhi tomorrow at 6pm"}})
    raw = client.get("/traces/t1")
    assert raw.status_code == 200 and raw.headers["content-type"].startswith(
        "application/x-ndjson")
    rows = [json.loads(line) for line in raw.text.splitlines()]
    assert rows and all({"event", "component", "epoch", "t_ms"} <= r.keys() for r in rows)
    events = {r["event"] for r in rows}
    assert {"event_received", "ack_sent", "tool_read", "write_committed",
            "response_sent"} <= events
    view = client.get("/traces/t1/view")
    assert view.status_code == 200 and view.headers["content-type"].startswith("text/html")
    assert "CHRONOS" in view.text
    assert client.get("/traces/never-happened").status_code == 404
    assert client.get("/traces/never-happened/view").status_code == 404


# --------------------------------------------------------------------------- health --------
def test_health_reports_ollama_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm_mode"] == "mock"
    assert h["ollama"]["reachable"] is True
    assert h["ollama"]["models"] == {"llama3.2:3b": True, "moondream": True}  # ":latest" matched


def test_health_flags_a_missing_model_when_running_against_ollama(fast_settings):
    s = dataclasses.replace(fast_settings, llm_mode="ollama")
    only_llama = ollama_transport({"models": [{"name": "llama3.2:3b"}]})
    with TestClient(make_app(s, ollama_transport=only_llama)) as c:
        h = c.get("/health").json()
    assert h["ollama"]["reachable"] and h["ollama"]["models"] == {
        "llama3.2:3b": True, "moondream": False}
    assert h["status"] == "degraded"


def test_health_when_ollama_is_down_is_ok_in_mock_mode_degraded_otherwise(fast_settings):
    def refuse(_r):
        raise httpx.ConnectError("refused")

    down = httpx.MockTransport(refuse)
    with TestClient(make_app(fast_settings, ollama_transport=down)) as c:
        h = c.get("/health").json()
    assert h["ollama"]["reachable"] is False and h["status"] == "ok"
    real = dataclasses.replace(fast_settings, llm_mode="ollama")
    with TestClient(make_app(real, ollama_transport=down)) as c:
        assert c.get("/health").json()["status"] == "degraded"


def test_sessions_are_isolated(client):
    client.post("/sessions/a/events?wait=true",
                json={"type": "text", "payload": {"text": "Book a flight to Delhi tomorrow at 6pm"}})
    client.post("/sessions/b/events?wait=true",
                json={"type": "text", "payload": {"text": "Book a table for 2"}})
    a = client.get("/sessions/a/state").json()
    b = client.get("/sessions/b/state").json()
    assert len(a["world"]["bookings"]) == 1 and a["world"]["reservations"] == []
    assert len(b["world"]["reservations"]) == 1 and b["world"]["bookings"] == []
    assert client.get("/health").json()["sessions"] == 2
