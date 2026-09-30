"""A camera frame with no question used to produce no response at all (found in the demo UI:
the sample-image buttons send a frame alone when the text box is empty)."""
import base64
from pathlib import Path

from helpers import Driver

from chronos.protocol import EventType
from chronos.protocol import OutputType as T

IMAGES = Path(__file__).resolve().parent.parent / "scenarios" / "images"


async def test_a_frame_with_no_question_defaults_to_whats_wrong_with_this(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "panel_disconnected_cable.png")  # no text at all
    await d.idle()
    assert [a.text for a in d.acks()] == ["Let me take a look — checking what the camera sees…"]
    (resp,) = d.of(T.RESPONSE)
    diag = resp.data["diagnosis"]
    assert diag["grounded"] is True and diag["kb_ids"][0] == "loose_power_cable"
    assert s.vision.calls == ["panel_disconnected_cable.png"]  # described exactly once
    assert s.epochs.current() == 1 and not d.events("epoch_bumped")
    # ack -> vision description -> diagnosis, in that order
    names = [r["event"] for r in d.rows()]
    assert names.index("ack_sent") < names.index("frame_described") < names.index("diagnosis_ready")


async def test_the_uploaded_frame_path_used_by_the_web_client_also_answers(open_session):
    """The browser sends base64 with a file name, not a server-side path."""
    s = await open_session()
    d = Driver(s)
    b64 = base64.b64encode((IMAGES / "panel_disconnected_cable.png").read_bytes()).decode()
    await d.send(EventType.CAMERA_FRAME, {"b64": b64, "name": "panel_disconnected_cable.png"})
    await d.idle()
    assert len(d.of(T.RESPONSE)) == 1 and not d.of(T.ERROR)


async def test_a_question_typed_after_the_frame_is_the_only_question_asked(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "machine_overheating_fan.png")
    await d.text("what is wrong with this machine")  # arrives inside the grace period
    await d.idle()
    assert len(d.of(T.RESPONSE)) == 1 and len(d.acks()) == 1  # no second, default, question
    assert s.vision.calls == ["machine_overheating_fan.png"]
    assert d.of(T.RESPONSE)[0].data["diagnosis"]["kb_ids"][0] == "overheating_fan"


async def test_a_question_sent_with_the_frame_is_not_replaced_by_the_default(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "breaker_tripped.png", text="Why did the power go out?")
    await d.idle()
    assert len(d.of(T.RESPONSE)) == 1 and len(d.acks()) == 1
    assert not [r for r in d.rows() if r["event"] == "frame_default_question"]


async def test_a_frame_while_a_booking_runs_does_not_hijack_it(open_session):
    """A picture with no words must not become a question in the middle of another goal."""
    s = await open_session()
    d = Driver(s)
    await d.say("Book a flight to Delhi tomorrow")
    await d.frame(IMAGES / "breaker_tripped.png")
    await d.idle()
    assert not [r for r in d.rows() if r["event"] == "frame_default_question"]
    assert s.epochs.current() == 1
    assert len(d.of(T.ACTION_RESULT)) == 1


async def test_any_question_about_a_fresh_frame_is_a_question_about_the_picture(open_session):
    """Found in the browser: 'why is it not powering on' + a frame answered 'Hello! How can I
    help?' and ignored the picture, because no rule recognises the phrase."""
    for question in ("why is it not powering on", "is this safe?", "it keeps shutting off"):
        s = await open_session(f"q-{abs(hash(question))}")
        d = Driver(s)
        await d.frame(IMAGES / "breaker_tripped.png", text=question)
        await d.idle()
        assert d.acks()[0].text == "Let me take a look — checking what the camera sees…", question
        (resp,) = d.of(T.RESPONSE)
        assert resp.data["diagnosis"]["grounded"] is True and s.vision.calls == ["breaker_tripped.png"]
        assert d.events("frame_question")


async def test_a_question_typed_a_moment_after_the_frame_is_also_about_the_picture(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "machine_overheating_fan.png")
    await d.text("why does it keep shutting down")
    await d.idle()
    (resp,) = d.of(T.RESPONSE)
    assert resp.data["diagnosis"]["kb_ids"][0] == "overheating_fan"


async def test_greetings_and_unrelated_words_are_not_turned_into_diagnoses(open_session):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "breaker_tripped.png")
    await d.say("hello")  # rules know this one: small talk, even with a fresh picture
    await d.idle()
    assert d.of(T.RESPONSE)[-1].text == "Hello! How can I help?"
    s2 = await open_session("no-frame")
    d2 = Driver(s2)
    await d2.say("why is it not powering on")  # no picture at all: nothing to ask about
    await d2.idle()
    assert d2.of(T.RESPONSE)[-1].text == "Hello! How can I help?" and not s2.vision.calls


async def test_an_old_frame_is_not_used_for_a_later_unrelated_question(open_session, monkeypatch):
    s = await open_session()
    d = Driver(s)
    await d.frame(IMAGES / "breaker_tripped.png")
    await d.idle()
    monkeypatch.setattr(s.planner.frames, "age_s", lambda: 3600.0)
    await d.say("why is it not powering on")
    await d.idle()
    assert d.of(T.RESPONSE)[-1].text == "Hello! How can I help?"
