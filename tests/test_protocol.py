import pytest
from pydantic import ValidationError

from chronos.protocol import (
    BargeInType,
    Event,
    EventType,
    OutputMessage,
    OutputStatus,
    OutputType,
    ToolCall,
    ToolResult,
)


def test_event_defaults_and_roundtrip():
    e = Event(session_id="s", type=EventType.TEXT, payload={"text": "hi"}, epoch=2)
    assert e.event_id and e.ts_monotonic > 0
    assert Event.model_validate_json(e.model_dump_json()) == e


def test_event_ids_unique():
    ids = {Event(session_id="s", type=EventType.TEXT).event_id for _ in range(200)}
    assert len(ids) == 200


def test_event_is_frozen_and_validates():
    e = Event(session_id="s", type=EventType.TEXT)
    with pytest.raises(ValidationError):
        e.epoch = 5
    with pytest.raises(ValidationError):
        Event(session_id="s", type="bogus")
    with pytest.raises(ValidationError):
        Event(session_id="s", type=EventType.TEXT, epoch=-1)


def test_output_roundtrip_and_enum_enforced():
    m = OutputMessage(type=OutputType.ACK, session_id="s", epoch=1,
                      status=OutputStatus.PENDING, text="Got it")
    assert OutputMessage.model_validate_json(m.model_dump_json()) == m
    assert set(m.model_dump(mode="json")) == {
        "type", "session_id", "epoch", "status", "text", "data", "trace_id", "ts"}
    with pytest.raises(ValidationError):
        OutputMessage(type="nope", session_id="s", epoch=0, status="pending")


def test_tool_models_roundtrip():
    c = ToolCall(tool="book_flight", args={"a": 1}, epoch=3, is_write=True)
    r = ToolResult(call_id=c.call_id, tool=c.tool, data={"x": 1}, epoch=3)
    assert ToolCall.model_validate_json(c.model_dump_json()) == c
    assert ToolResult.model_validate_json(r.model_dump_json()) == r


def test_epoch_bumping_types():
    bumps = {t for t in BargeInType if t.bumps_epoch}
    assert bumps == {BargeInType.CORRECTION, BargeInType.GOAL_CHANGE, BargeInType.CANCEL}
