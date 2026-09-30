import time
from datetime import date

import pytest

from chronos.fastpath.ack import (
    ALL_TEMPLATES,
    COMPLETION_WORDS,
    AckBuilder,
    contains_completion_claim,
    describe_slot,
    fmt_date,
    fmt_time,
)
from chronos.perception.intent import Intent
from chronos.protocol import BargeInType as B
from chronos.protocol import OutputMessage, OutputStatus, OutputType

TODAY = date(2026, 10, 1)
ack = AckBuilder(lambda: TODAY)

SAMPLE_SLOTS = {"dest": "DEL", "destination": "Chennai Airport", "via": "Gas Station",
                "stop": "gas station", "party_size": 4, "date": "2026-10-08", "time": "18:00"}
SAMPLE_CHANGED = [{"time": "18:00"}, {"date": "2026-10-08"}, {"party_size": 5},
                  {"seat_pref": "window"}, {"dest": "BOM"}, {"destination": "Home"}, {}]


def test_completion_detector_itself_works():
    for bad in ("Booked!", "All set.", "Your flight is booked", "Done", "Table reserved",
                "That is confirmed", "Navigation set"):
        assert contains_completion_claim(bad), bad
    for ok in ("Got it, switching to 6 pm…", "Okay, stopping that.", "settings menu",
               "looking up flights", "Setting"):
        assert not contains_completion_claim(ok), ok


def test_spec_words_are_all_banned():
    assert {"booked", "done", "confirmed", "reserved", "set"} <= set(COMPLETION_WORDS)


def test_no_template_contains_completion_words():
    assert len(ALL_TEMPLATES) >= 20
    for t in ALL_TEMPLATES:
        assert not contains_completion_claim(t), t


def test_no_rendered_ack_contains_completion_words_or_placeholders():
    checked = 0
    for intent in [*Intent, None]:
        for barge in [None, *B]:
            for changed in SAMPLE_CHANGED:
                for slots in (SAMPLE_SLOTS, {}):
                    for undo in (False, True):
                        text = ack.build(intent, barge, slots, changed=changed, has_writes=undo)
                        if text is None:
                            continue
                        checked += 1
                        assert text and "{" not in text and "}" not in text, text
                        assert not contains_completion_claim(text), text
    assert checked > 500


def test_spec_examples():
    assert ack.build(Intent.BOOK_FLIGHT, None, {"dest": "DEL"}) == "On it — looking up flights to Delhi…"
    assert ack.build(Intent.BOOK_FLIGHT, B.CORRECTION,
                     changed={"date": "2026-10-08"}) == "Got it, switching to next week…"
    assert ack.build(Intent.BOOK_FLIGHT, B.CORRECTION,
                     changed={"time": "18:00"}) == "Got it, switching to 6 pm…"
    assert ack.build(None, B.CANCEL) == "Okay, stopping that."
    assert ack.build(Intent.NAVIGATE, B.GOAL_CHANGE, {"stop": "gas station"}) \
        == "Okay, changing plans — one moment…"
    assert ack.build(Intent.ADD_STOP, B.GOAL_CHANGE, {"stop": "gas station"}) \
        == "Okay, changing plans — gas station first…"


def test_backchannel_and_hesitation_get_no_ack():
    for barge in (B.BACKCHANNEL, B.HESITATION):
        for intent in [*Intent, None]:
            assert ack.build(intent, barge, SAMPLE_SLOTS) is None


def test_missing_slot_uses_neutral_wording_not_invented_values():
    assert ack.build(Intent.BOOK_FLIGHT, None, {}) == "On it — looking up flights…"
    assert ack.build(Intent.NAVIGATE, None, {}) == "On it — working out the route…"
    assert "that" not in ack.build(Intent.RESERVE_TABLE, None, {}).split()


def test_cancel_mentions_undo_only_when_writes_exist():
    assert ack.build(None, B.CANCEL, has_writes=True).startswith("Okay, stopping that")
    assert "undo" in ack.build(None, B.CANCEL, has_writes=True)
    assert "undo" not in ack.build(None, B.CANCEL, has_writes=False)


def test_formatters():
    assert [fmt_time(t) for t in ("18:00", "06:30", "00:00", "12:15", "12:00")] == [
        "6 pm", "6:30 am", "12 am", "12:15 pm", "12 pm"]
    assert fmt_date("2026-10-01", TODAY) == "today"
    assert fmt_date("2026-10-02", TODAY) == "tomorrow"
    assert fmt_date("2026-10-08", TODAY) == "next week"
    assert fmt_date("2026-10-12", TODAY) == "Mon 12 Oct"
    assert describe_slot("seat_pref", "window", TODAY) == "a window seat"
    assert describe_slot("nonsense", 1, TODAY) is None


def test_ack_message_is_protocol_compliant_and_pending():
    m = ack.message("s1", 2, "Okay, stopping that.")
    assert m.type is OutputType.ACK and m.status is OutputStatus.PENDING and m.epoch == 2
    assert OutputMessage.model_validate_json(m.model_dump_json()) == m


@pytest.mark.parametrize("n", [1000])
def test_1000_acks_p99_under_5ms(n):
    combos = [(i, b) for i in [*Intent, None] for b in [None, *B]]
    lats = []
    for k in range(n):
        intent, barge = combos[k % len(combos)]
        t0 = time.perf_counter()
        ack.build(intent, barge, SAMPLE_SLOTS, changed=SAMPLE_CHANGED[k % len(SAMPLE_CHANGED)])
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    p99 = lats[int(n * 0.99)]
    assert p99 < 5.0, f"p99={p99:.3f}ms"
