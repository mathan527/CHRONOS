from collections import Counter

from bench.scenarios import _PHASE_RANGE, IMAGES, KINDS, PHASES, TODAY, VARIANTS, generate

TOMORROW, NEXT_WEEK = "2026-10-02", "2026-10-08"


def test_generation_is_reproducible_and_seed_sensitive():
    a, b = generate(60, seed=7), generate(60, seed=7)
    assert a == b
    assert generate(60, seed=8) != a
    assert [s.seed for s in a] == [s.seed for s in b]
    assert generate(0) == [] and len(generate(200)) == 200


def test_use_cases_are_dealt_evenly_and_every_variant_appears():
    scns = generate(400, seed=1)
    kinds = Counter(s.kind for s in scns)
    assert set(kinds) == set(KINDS) and max(kinds.values()) - min(kinds.values()) <= 1
    seen = {(s.kind, s.variant) for s in scns}
    assert seen == {(k, v) for k, opts in VARIANTS.items() for v, _ in opts}
    assert {s.phase for s in scns} == set(PHASES) | {None}


def test_structure_invariants():
    for s in generate(300, seed=2):
        times = [st.at_ms for st in s.steps]
        assert times == sorted(times) and times[0] >= 0
        assert any(st.respond for st in s.steps)  # every scenario has something to answer
        lo, hi = s.tool_latency_ms
        assert 3 <= lo < hi and s.tags["mean_latency_ms"] > 0
        if s.variant == "control":
            assert s.interrupt_ms is None and s.phase is None
        elif s.interrupt_ms is not None:
            assert s.phase in PHASES
        assert all(st.image in IMAGES for st in s.steps if st.kind == "frame")
        assert all(st.kind in ("partial", "final", "frame") for st in s.steps)


def test_interrupt_timing_follows_the_phase_and_scales_with_tool_latency():
    checked = 0
    for s in generate(500, seed=3):
        if s.interrupt_ms is None:
            continue
        m = s.tags["mean_latency_ms"]
        offset = s.interrupt_ms - s.first_final_ms
        lo, hi = _PHASE_RANGE[s.phase]
        extra = 30 if s.phase == "after" else 0
        if s.kind == "field":  # the first utterance is at t=0 for frame scenarios
            assert lo * m <= s.interrupt_ms - extra <= hi * m + 1e-6
        else:
            assert lo * m + extra - 1e-6 <= offset <= hi * m + extra + 1e-6, (s.phase, m, offset)
        checked += 1
    assert checked > 200


def test_latency_range_is_respected_and_random():
    scns = generate(200, seed=4, latency_range_ms=(50, 60))
    means = [s.tags["mean_latency_ms"] for s in scns]
    # means are rounded to 0.1 ms inside a 10 ms window, so at most 100 distinct values exist
    assert all(50 <= m <= 60 for m in means) and len(set(means)) > 60


def test_oracles_encode_the_users_final_intent():
    by = {}
    for s in generate(600, seed=5):
        by.setdefault((s.kind, s.variant), s)
    e = by[("support", "date_fix")].expect
    assert e.bookings == ({"dest": "DEL", "date": NEXT_WEEK},) and not e.reservations
    e = by[("support", "time_fix")].expect
    assert e.bookings == ({"dest": "DEL", "date": TOMORROW, "time": "18:00"},)
    assert by[("support", "cancel")].expect.bookings == ()
    assert by[("support", "control")].expect.bookings == ({"dest": "DEL", "date": TOMORROW},)
    assert by[("incar", "goal_change")].expect.navigation == (
        {"destination": "chennai airport", "via": "gas station"},)
    assert by[("incar", "cancel")].expect.navigation == ()
    assert by[("incar", "control")].expect.navigation == (
        {"destination": "chennai airport", "via": None},)
    assert by[("access", "hesitate_then_request")].expect.reservations == ({"party_size": 4},)
    assert by[("access", "request_then_fix")].expect.reservations == ({"party_size": 6},)
    assert by[("access", "cancel")].expect.reservations == ()
    ctrl = by[("field", "control")]
    assert ctrl.expect.diagnosis_kb == IMAGES[ctrl.steps[0].image]
    swap = by[("field", "swap")]
    assert swap.expect.diagnosis_kb == IMAGES[swap.steps[-1].image]  # the LAST frame wins
    assert swap.steps[0].image != swap.steps[-1].image


def test_hesitation_scenarios_never_carry_a_finished_request_before_the_final():
    for s in generate(300, seed=6):
        if s.variant == "hesitation_then_request" or s.variant == "hesitate_then_request":
            partials = [st.text for st in s.steps if st.kind == "partial"]
            assert partials == ["I want to boo…", "I want to boo… um…"]
            assert [st.text for st in s.steps if st.kind == "final" and st.respond] == [
                "actually, book a table for 4"]


def test_speech_is_streamed_word_by_word_when_streamed():
    streamed = [s for s in generate(100, seed=9) if s.tags["streamed"] and s.kind == "support"]
    assert streamed
    s = streamed[0]
    partials = [st for st in s.steps if st.kind == "partial"]
    final = next(st for st in s.steps if st.kind == "final" and st.respond)
    assert partials and all(final.text.startswith(p.text) for p in partials)
    assert [len(p.text.split()) for p in partials] == list(range(1, len(partials) + 1))
    assert TODAY.isoformat() == "2026-10-01"
