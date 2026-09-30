"""The metric definitions, on hand-built inputs where the right answer is obvious."""
import math

import pytest

from bench.instrument import RecordingWorld, WriteTap
from bench.metrics import (
    check_state,
    duplicate_and_double_charge,
    group_by,
    is_stale,
    live_ids,
    pct,
    percentile,
    stale_stats,
    summarize,
    ttfr_ms,
    wilson,
)
from bench.scenarios import Expect

NEXT_WEEK = {"dest": "DEL", "date": "2026-10-08"}


# ---------------------------------------------------------------------------- statistics -----
def test_percentile_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([7], 99) == 7
    xs = [10, 20, 30, 40]
    assert percentile(xs, 0) == 10 and percentile(xs, 100) == 40
    assert percentile(xs, 50) == 25.0
    assert percentile(xs, 25) == 17.5
    assert percentile(list(range(101)), 99) == pytest.approx(99.0)
    assert percentile([40, 10, 30, 20], 50) == 25.0  # order of input does not matter


def test_wilson_interval_matches_textbook_values():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(27.8, abs=0.1)
    lo, hi = wilson(10, 10)
    assert lo == pytest.approx(72.2, abs=0.1) and hi == 100.0
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(40.4, abs=0.2) and hi == pytest.approx(59.6, abs=0.2)
    lo, hi = wilson(200, 200)
    assert lo > 98 and hi == 100.0  # a perfect score over 200 runs still has an honest interval


def test_pct():
    assert pct(1, 3) == 33.3 and pct(0, 0) == 0.0 and pct(5, 5) == 100.0


# -------------------------------------------------------------------------- stale writes -----
def test_is_stale_compares_arguments_with_the_final_intent():
    e = Expect(bookings=(NEXT_WEEK,))
    assert is_stale("book_flight", {"dest": "DEL", "date": "2026-10-02", "time": "06:00"}, e)
    assert not is_stale("book_flight", {"dest": "DEL", "date": "2026-10-08", "time": "06:00"}, e)
    assert is_stale("book_flight", {"dest": "BOM", "date": "2026-10-08"}, e)
    # a booking the user cancelled: every booking is stale
    assert is_stale("book_flight", {"dest": "DEL", "date": "2026-10-08"}, Expect())
    # tools that are not what the user asked for are never "stale writes"
    assert not is_stale("cancel_booking", {"booking_id": "BK-0001"}, Expect())
    assert not is_stale("search_flights", {"dest": "DEL"}, Expect())


def test_is_stale_navigation_and_reservations():
    nav = Expect(navigation=({"destination": "chennai airport", "via": "gas station"},))
    assert not is_stale("set_navigation", {"destination": "Chennai Airport", "via": "gas station"}, nav)
    assert is_stale("set_navigation", {"destination": "Chennai Airport", "via": None}, nav)
    res = Expect(reservations=({"party_size": 6},))
    assert is_stale("reserve_table", {"party_size": 4}, res)
    assert not is_stale("reserve_table", {"party_size": 6, "restaurant": None}, res)
    plain = Expect(navigation=({"destination": "chennai airport", "via": None},))
    assert not is_stale("set_navigation", {"destination": "Chennai Airport", "via": None}, plain)


def state(bookings=(), charges=(), reservations=(), navigation=(), violations=()):
    return {"bookings": list(bookings), "charges": list(charges),
            "reservations": list(reservations), "navigation": list(navigation),
            "violations": list(violations)}


def booking(i, status="BOOKED", **kw):
    return {"id": i, "status": status, "origin": "MAA", "dest": "DEL", "date": "2026-10-08",
            "time": "06:00", **kw}


def test_check_state_accepts_exactly_the_expected_live_state():
    e = Expect(bookings=(NEXT_WEEK,))
    assert check_state(e, state(bookings=[booking(2), booking(1, "CANCELLED", date="2026-10-02")]),
                       None) == (True, [])
    ok, why = check_state(e, state(bookings=[booking(1), booking(2)]), None)
    assert not ok and "2 live bookings, expected 1" in why[0]
    ok, why = check_state(e, state(bookings=[booking(1, date="2026-10-02")]), None)
    assert not ok and "no live booking matching" in why[0]
    ok, why = check_state(e, state(), None)
    assert not ok and "0 live bookings, expected 1" in why[0]


def test_check_state_covers_every_kind_and_the_invariants_and_the_diagnosis():
    ok, why = check_state(Expect(), state(reservations=[{"id": 1, "status": "ACTIVE",
                                                         "party_size": 4, "restaurant": "x"}]), None)
    assert not ok and "1 live reservations, expected 0" in why[0]  # a cancelled task left a table
    assert check_state(Expect(), state(violations=["seat accounting broken"]), None)[0] is False
    d = Expect(diagnosis_kb="loose_power_cable")
    assert check_state(d, state(), "loose_power_cable") == (True, [])
    ok, why = check_state(d, state(), "overheating_fan")
    assert not ok and "overheating_fan" in why[0]
    assert check_state(d, state(), None)[0] is False
    nav = Expect(navigation=({"destination": "chennai airport", "via": "gas station"},))
    live = {"id": 1, "status": "ACTIVE", "destination": "Chennai Airport", "via": "gas station"}
    assert check_state(nav, state(navigation=[live]), None)[0]
    assert not check_state(nav, state(navigation=[{**live, "via": None}]), None)[0]
    superseded = {**live, "id": 2, "status": "SUPERSEDED", "via": None}
    assert check_state(nav, state(navigation=[superseded, live]), None)[0]  # only ACTIVE counts


def test_duplicate_and_double_charge_are_read_from_the_final_state():
    one = state(bookings=[booking(1)], charges=[{"booking_id": 1, "status": "CHARGED"}])
    assert duplicate_and_double_charge(one) == (False, False)
    two = state(bookings=[booking(1), booking(2)],
                charges=[{"booking_id": 1, "status": "CHARGED"},
                         {"booking_id": 2, "status": "CHARGED"}])
    assert duplicate_and_double_charge(two) == (True, True)
    fixed = state(bookings=[booking(1, "CANCELLED"), booking(2)],
                  charges=[{"booking_id": 1, "status": "REFUNDED"},
                           {"booking_id": 2, "status": "CHARGED"}])
    assert duplicate_and_double_charge(fixed) == (False, False)  # refunded is not a second charge
    assert live_ids(fixed) == {"BK-0002"}


def test_ttfr_takes_the_first_message_at_or_after_each_utterance():
    samples, missing = ttfr_ms([1.0, 2.0], [1.01, 1.5, 2.3])
    assert missing == 0 and samples == pytest.approx([10.0, 300.0])
    # a message that came BEFORE the utterance does not count as its response
    samples, missing = ttfr_ms([5.0], [1.0, 2.0])
    assert samples == [] and missing == 1
    # two utterances answered by the same later message (the baseline's restart case)
    samples, _ = ttfr_ms([1.0, 1.2], [2.0])
    assert samples == pytest.approx([1000.0, 800.0])
    assert ttfr_ms([], [1.0]) == ([], 0)


# ------------------------------------------------ stale_stats against a real recording world --
async def test_stale_stats_uses_the_worlds_call_log_and_ids(tmp_path):
    w = await RecordingWorld.create(":memory:", latency_ms=(1, 2), today=__import__("datetime").date(2026, 10, 1))
    tap = WriteTap()
    expect = Expect(bookings=(NEXT_WEEK,))
    # attempt 1: stale, reached the world, later undone by a compensating call
    a1 = tap.begin("book_flight", {"dest": "DEL", "date": "2026-10-02"})
    r1 = await w.book_flight("MAA", "DEL", "2026-10-02", "06:00", "G", None)
    tap.end(a1, "committed")
    await w.cancel_booking(r1["booking_id"])
    # attempt 2: stale, never reached the world
    a2 = tap.begin("book_flight", {"dest": "DEL", "date": "2026-10-03"})
    tap.end(a2, "prevented")
    # attempt 3: stale, reached the world and is still live
    a3 = tap.begin("book_flight", {"dest": "DEL", "date": "2026-10-04"})
    await w.book_flight("MAA", "DEL", "2026-10-04", "06:00", "G", None)
    tap.end(a3, "committed")
    # attempt 4: the right one (not stale)
    a4 = tap.begin("book_flight", {"dest": "DEL", "date": "2026-10-08"})
    await w.book_flight("MAA", "DEL", "2026-10-08", "06:00", "G", None)
    tap.end(a4, "committed")

    from bench.metrics import snapshot_world
    st = stale_stats(tap, w, expect, await snapshot_world(w))
    assert st == {"started": 3, "prevented": 1, "compensated": 1, "standing": 1, "superseded": 0}
    await w.close()


# ------------------------------------------------------------------------- aggregation ------
def run(**kw):
    base = {"ttfr_ms": [10.0], "no_response": 0, "wall_s": 1.0, "consistent": True,
            "duplicate_live": False, "double_charge": False, "error": None, "kind": "support",
            "phase": "before", "world_reads": 2, "world_writes": 1, "stale_emitted": 0,
            "stale": {"started": 0, "prevented": 0, "compensated": 0, "standing": 0,
                      "superseded": 0}}
    return {**base, **kw}


def test_summarize_computes_everything_from_the_runs():
    runs = [run(ttfr_ms=[10.0, 20.0]), run(ttfr_ms=[30.0], wall_s=3.0, consistent=False,
                                            duplicate_live=True, double_charge=True,
                                            stale={"started": 2, "prevented": 1, "compensated": 0,
                                                   "standing": 1, "superseded": 0}),
            run(error="boom", consistent=False)]
    s = summarize(runs)
    assert s["scenarios"] == 3 and s["errors"] == 1
    assert s["ttfr_ms"]["n"] == 4 and s["ttfr_ms"]["p50"] == 15.0 and s["ttfr_ms"]["max"] == 30.0
    assert s["wall_s"]["mean"] == pytest.approx(5 / 3)
    assert s["consistent"]["k"] == 1 and s["consistent"]["pct"] == 33.3
    assert s["duplicate_live_writes"]["k"] == 1 and s["double_charge"]["k"] == 1
    sw = s["stale_writes"]
    assert sw["started"] == 2 and sw["prevented_pct"] == 50.0 and sw["standing_pct"] == 50.0
    assert sw["runs_with_stale_writes"] == 1
    assert s["world_calls_per_scenario"] == {"reads": 2.0, "writes_committed": 1.0}
    assert all(math.isfinite(x) for x in sw["standing_ci95"])


def test_summarize_changes_when_the_measurements_change():
    """Guards against hardcoded results: same code, different inputs, different outputs."""
    a = summarize([run(ttfr_ms=[5.0])] * 4)
    b = summarize([run(ttfr_ms=[500.0], consistent=False)] * 4)
    assert a["ttfr_ms"]["p50"] == 5.0 and b["ttfr_ms"]["p50"] == 500.0
    assert a["consistent"]["pct"] == 100.0 and b["consistent"]["pct"] == 0.0


def test_summarize_handles_no_samples_without_dividing_by_zero():
    s = summarize([run(ttfr_ms=[], wall_s=None, no_response=1)])
    assert s["ttfr_ms"]["p50"] is None and s["ttfr_ms"]["no_response"] == 1
    assert s["wall_s"]["mean"] is None and s["stale_writes"]["prevented_pct"] == 0.0


def test_group_by_splits_runs():
    runs = [run(kind="incar"), run(kind="incar", consistent=False), run(kind="field")]
    g = group_by(runs, "kind")
    assert set(g) == {"incar", "field"}
    assert g["incar"]["scenarios"] == 2 and g["incar"]["consistent"]["k"] == 1
