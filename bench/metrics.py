"""Metric definitions. Pure functions: everything reported is computed here from measurements.

Definitions (also printed in results.md):

time-to-first-response (TTFR)
    For each actionable utterance the user sends (a final transcript / frame+question that
    expects an answer, not a stray "okay"): milliseconds from the moment it was sent to the first
    message of any kind the agent emits at or after that moment. CHRONOS emits an
    acknowledgment; the half-duplex baseline speaks only after it has acted, so this metric
    measures responsiveness, NOT how fast the task completes (see wall-clock for that).

stale write
    A write attempt whose arguments do not match the user's FINAL intent (`Expect`), e.g. the
    "tomorrow" booking when the user finally wanted "next week". For each stale write the agent
    STARTED (entered its write pipeline) the world's own call log says what happened:
      prevented    it never reached the world (fenced, blocked, or cancelled before dispatch)
      compensated  it committed, then an explicit compensating call undid it
      superseded   it committed and was replaced by a later write (navigation only)
      standing     it committed and is still live at the end (the harmful outcome)

duplicate writes / double charge
    Measured on the FINAL world state: more than one live booking/reservation/navigation where
    the user wanted at most one; more than one un-refunded charge.

final-state consistency
    The world's live state equals the oracle (`Expect`) exactly, its invariants hold, and, for
    troubleshooting, the last diagnosis identifies the fault shown in the last frame.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from bench.instrument import RESULT_ID_KEY, RecordingWorld, WriteTap
from bench.scenarios import Expect


def percentile(values: Sequence[float], q: float) -> float | None:
    """q in [0, 100], linear interpolation between order statistics."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, as percentages."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1))


def pct(k: int, n: int) -> float:
    return round(100 * k / n, 1) if n else 0.0


# ----------------------------------------------------------------------- oracle matching ------
def _eq(actual: Any, want: Any) -> bool:
    if want is None:
        return not actual
    if isinstance(want, str) and isinstance(actual, str):
        return actual.lower() == want.lower()
    return actual == want


def matches(entity: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(_eq(entity.get(k), v) for k, v in want.items())


def is_stale(tool: str, args: dict[str, Any], expect: Expect) -> bool:
    wants = {"book_flight": expect.bookings, "reserve_table": expect.reservations,
             "set_navigation": expect.navigation}.get(tool)
    if wants is None:
        return False  # not a primary write (compensations, reads)
    return not any(matches(args, w) for w in wants)


async def snapshot_world(world: RecordingWorld) -> dict[str, Any]:
    q = world._all
    bookings = [dict(r) for r in await q(
        "SELECT b.id, b.status, f.origin, f.dest, f.date, f.time FROM bookings b "
        "JOIN flights f ON f.flight_id=b.flight_id")]
    return {
        "bookings": bookings,
        "charges": [dict(r) for r in await q("SELECT booking_id, status FROM charges")],
        "reservations": [dict(r) for r in await q(
            "SELECT id, status, party_size, restaurant FROM reservations")],
        "navigation": [dict(r) for r in await q(
            "SELECT id, status, destination, via FROM navigation")],
        "violations": await world.check_invariants(),
    }


def live(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {"bookings": [b for b in state["bookings"] if b["status"] == "BOOKED"],
            "reservations": [r for r in state["reservations"] if r["status"] == "ACTIVE"],
            "navigation": [n for n in state["navigation"] if n["status"] == "ACTIVE"]}


def live_ids(state: dict[str, Any]) -> set[str]:
    lv = live(state)
    return ({f"BK-{b['id']:04d}" for b in lv["bookings"]}
            | {f"RS-{r['id']:04d}" for r in lv["reservations"]}
            | {f"NAV-{n['id']:04d}" for n in lv["navigation"]})


def check_state(expect: Expect, state: dict[str, Any], last_diagnosis_kb: str | None
                ) -> tuple[bool, list[str]]:
    problems: list[str] = []
    lv = live(state)
    for name, singular, wants in (("bookings", "booking", expect.bookings),
                                  ("reservations", "reservation", expect.reservations),
                                  ("navigation", "navigation", expect.navigation)):
        have = list(lv[name])
        if len(have) != len(wants):
            problems.append(f"{len(have)} live {name}, expected {len(wants)}")
            continue
        for want in wants:
            hit = next((e for e in have if matches(e, want)), None)
            if hit is None:
                problems.append(f"no live {singular} matching {want}")
            else:
                have.remove(hit)
    problems += [f"world invariant: {v}" for v in state["violations"]]
    if expect.diagnosis_kb is not None and last_diagnosis_kb != expect.diagnosis_kb:
        problems.append(f"last diagnosis {last_diagnosis_kb!r}, expected {expect.diagnosis_kb!r}")
    return (not problems, problems)


def duplicate_and_double_charge(state: dict[str, Any]) -> tuple[bool, bool]:
    lv = live(state)
    duplicate = any(len(v) > 1 for v in lv.values())
    charged = sum(1 for c in state["charges"] if c["status"] == "CHARGED")
    return duplicate, charged > 1


def stale_stats(tap: WriteTap, world: RecordingWorld, expect: Expect,
                state: dict[str, Any]) -> dict[str, int]:
    started = [a for a in tap.attempts if is_stale(a.tool, a.args, expect)]
    committed = [c for c in world.primary_writes() if is_stale(c.tool, c.args, expect)]
    ids = [str((c.result or {}).get(RESULT_ID_KEY[c.tool])) for c in committed]
    comp, alive = world.compensated_ids(), live_ids(state)
    compensated = sum(1 for i in ids if i in comp)
    standing = sum(1 for i in ids if i in alive)
    return {"started": len(started), "prevented": max(0, len(started) - len(committed)),
            "compensated": compensated, "standing": standing,
            "superseded": len(ids) - compensated - standing}


def ttfr_ms(sends: Sequence[float], outputs: Sequence[float]) -> tuple[list[float], int]:
    """sends = times of actionable utterances, outputs = times of agent messages (monotonic)."""
    outs = sorted(outputs)
    samples, missing = [], 0
    for t in sends:
        nxt = next((o for o in outs if o >= t), None)
        if nxt is None:
            missing += 1
        else:
            samples.append((nxt - t) * 1000)
    return samples, missing


# ------------------------------------------------------------------------ aggregation --------
def _rate(runs: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(runs)
    k = sum(1 for r in runs if r.get(key))
    return {"k": k, "n": n, "pct": pct(k, n), "ci95": list(wilson(k, n))}


def summarize(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    ttfr = [x for r in runs for x in r["ttfr_ms"]]
    wall = [r["wall_s"] for r in runs if r["wall_s"] is not None]
    st: Counter[str] = Counter()
    for r in runs:
        st.update(r["stale"])
    started = st["started"]
    stale = {k: st[k] for k in ("started", "prevented", "compensated", "superseded", "standing")}
    stale["runs_with_stale_writes"] = sum(1 for r in runs if r["stale"]["started"])
    for name in ("prevented", "compensated", "superseded", "standing"):
        stale[f"{name}_pct"] = pct(st[name], started)
        stale[f"{name}_ci95"] = list(wilson(st[name], started))
    return {
        "scenarios": n,
        "errors": sum(1 for r in runs if r.get("error")),
        "ttfr_ms": {"n": len(ttfr), "no_response": sum(r["no_response"] for r in runs),
                    "p50": percentile(ttfr, 50), "p95": percentile(ttfr, 95),
                    "p99": percentile(ttfr, 99), "mean": (sum(ttfr) / len(ttfr)) if ttfr else None,
                    "max": max(ttfr) if ttfr else None},
        "wall_s": {"mean": (sum(wall) / len(wall)) if wall else None,
                   "p50": percentile(wall, 50), "p95": percentile(wall, 95)},
        "stale_writes": stale,
        "consistent": _rate(runs, "consistent"),
        "duplicate_live_writes": _rate(runs, "duplicate_live"),
        "double_charge": _rate(runs, "double_charge"),
        "world_calls_per_scenario": {
            "reads": round(sum(r["world_reads"] for r in runs) / n, 2) if n else 0,
            "writes_committed": round(sum(r["world_writes"] for r in runs) / n, 2) if n else 0},
        "stale_outputs_emitted": sum(r.get("stale_emitted") or 0 for r in runs),
    }


def group_by(runs: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        buckets.setdefault(str(r.get(key)), []).append(r)
    out = {}
    for name, rs in sorted(buckets.items()):
        s = summarize(rs)
        out[name] = {"scenarios": s["scenarios"], "consistent": s["consistent"],
                     "duplicate_live_writes": s["duplicate_live_writes"],
                     "stale_writes": s["stale_writes"], "ttfr_ms": s["ttfr_ms"],
                     "wall_s": s["wall_s"]}
    return out
