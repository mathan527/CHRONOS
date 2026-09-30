"""Randomised, seeded interruption scenarios drawn from the four use cases, each with an oracle.

A scenario is a timed script of client events plus `Expect`, the world state that matches what
the user finally wanted. Interrupt timing is expressed relative to the scenario's tool latency `m`
(one tool call ~ m ms), so "before / during / after" mean the same thing at any latency:

    before  the interrupt lands while the first read is still running     (0.10-0.80 m)
    during  it lands while the first write is dispatching                 (1.15-1.85 m)
    after   it lands after the first goal has been carried out            (2.30-3.50 m)

Latency varies per call (0.5-1.5 m), so a scenario's *nominal* phase can differ from what
actually happened; the runner reports actual outcomes, and the phase is only a stratification key.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

TODAY = date(2026, 10, 1)
IMAGES = {"panel_disconnected_cable.png": "loose_power_cable",
          "machine_overheating_fan.png": "overheating_fan",
          "breaker_tripped.png": "tripped_breaker"}
KINDS = ("incar", "support", "field", "access")
VARIANTS: dict[str, list[tuple[str, float]]] = {
    "incar": [("goal_change", 0.70), ("control", 0.15), ("cancel", 0.15)],
    "support": [("date_fix", 0.35), ("time_fix", 0.25), ("cancel", 0.20), ("control", 0.20)],
    "field": [("control", 0.50), ("swap", 0.50)],
    "access": [("hesitate_then_request", 0.40), ("request_then_fix", 0.40), ("cancel", 0.20)],
}
PHASES = ("before", "during", "after")
_PHASE_RANGE = {"before": (0.10, 0.80), "during": (1.15, 1.85), "after": (2.30, 3.50)}
NOISE = ("okay", "uh-huh", "mm-hmm")


@dataclass(frozen=True)
class Step:
    at_ms: float
    kind: str  # partial | final | frame
    text: str = ""
    image: str | None = None
    respond: bool = False  # an actionable utterance: counts toward time-to-first-response


@dataclass(frozen=True)
class Expect:
    """What the world (and the last diagnosis) must look like at the end. Empty = none live."""
    bookings: tuple[dict[str, Any], ...] = ()
    reservations: tuple[dict[str, Any], ...] = ()
    navigation: tuple[dict[str, Any], ...] = ()
    diagnosis_kb: str | None = None


@dataclass(frozen=True)
class Scenario:
    id: int
    kind: str
    variant: str
    phase: str | None  # None when there is no interrupt
    seed: int
    tool_latency_ms: tuple[int, int]
    steps: tuple[Step, ...]
    expect: Expect
    first_final_ms: float = 0.0
    interrupt_ms: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def has_interrupt(self) -> bool:
        return self.interrupt_ms is not None


def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str:
    return rng.choices([o for o, _ in options], weights=[w for _, w in options], k=1)[0]


def _speech(rng: random.Random, text: str, start: float, streamed: bool
            ) -> tuple[list[Step], float]:
    """A spoken utterance: word-by-word partials then a final (or just a final)."""
    if not streamed:
        return [Step(start, "final", text, respond=True)], start
    word_ms = rng.uniform(35, 90)
    words = text.split()
    steps = [Step(start + i * word_ms, "partial", " ".join(words[: i + 1]))
             for i in range(len(words) - 1)]
    t_final = start + len(words) * word_ms
    steps.append(Step(t_final, "final", text, respond=True))
    return steps, t_final


def _offset(rng: random.Random, phase: str | None, m: float) -> float:
    """Milliseconds after the first utterance completes at which the interrupt lands."""
    lo, hi = _PHASE_RANGE[phase or "before"]
    return rng.uniform(lo, hi) * m + (30 if phase == "after" else 0)


def generate(n: int = 200, seed: int = 0, latency_range_ms: tuple[int, int] = (20, 120),
             today: date = TODAY) -> list[Scenario]:
    """`n` scenarios, reproducible from `seed`. Use cases are dealt round-robin so each gets an
    equal share; everything else (variant, phase, timing, phrasing, noise, latency) is random."""
    rng = random.Random(seed)
    tomorrow, next_week = (today + timedelta(days=1)).isoformat(), (
        today + timedelta(days=7)).isoformat()
    out: list[Scenario] = []
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        variant = _weighted(rng, VARIANTS[kind])
        interrupt = variant != "control"
        phase = rng.choice(PHASES) if interrupt else None
        m = rng.uniform(*latency_range_ms)
        latency = (max(3, int(0.5 * m)), max(5, int(1.5 * m)))
        streamed = rng.random() < 0.6
        steps: list[Step] = []
        expect = Expect()
        tags: dict[str, Any] = {"mean_latency_ms": round(m, 1), "streamed": streamed}
        t_first = 0.0
        t_int: float | None = None

        if kind == "incar":
            first = rng.choice(["Navigate to Chennai Airport", "Take me to Chennai Airport"])
            sp, t_first = _speech(rng, first, 0.0, streamed)
            steps += sp
            if variant == "goal_change":
                t_int = t_first + _offset(rng, phase, m)
                steps.append(Step(t_int, "final", rng.choice(
                    ["Actually, gas station first", "Wait, gas station first",
                     "Actually, petrol bunker first"]), respond=True))
                expect = Expect(navigation=({"destination": "chennai airport",
                                             "via": "gas station"},))
            elif variant == "cancel":
                t_int = t_first + _offset(rng, phase, m)
                steps.append(Step(t_int, "final", rng.choice(["stop", "never mind", "cancel that"]),
                                  respond=True))
            else:
                expect = Expect(navigation=({"destination": "chennai airport", "via": None},))
        elif kind == "support":
            time_fix = variant == "time_fix"
            first = rng.choice(["Book a flight to Delhi tomorrow", "I need a flight to Delhi tomorrow",
                                "Book me a flight to Delhi tomorrow"])
            if time_fix:
                first += " at 8pm"
            sp, t_first = _speech(rng, first, 0.0, streamed)
            steps += sp
            base = {"dest": "DEL", "date": tomorrow}
            if variant == "control":
                expect = Expect(bookings=(base,))
            else:
                t_int = t_first + _offset(rng, phase, m)
                if variant == "date_fix":
                    text = rng.choice(["Actually, next week instead", "No wait, next week",
                                       "Actually make it next week"])
                    expect = Expect(bookings=({"dest": "DEL", "date": next_week},))
                elif variant == "time_fix":
                    text = rng.choice(["make it 6pm", "Actually, make it 6pm", "No, 6pm"])
                    expect = Expect(bookings=({**base, "time": "18:00"},))
                else:
                    text = rng.choice(["cancel that", "never mind", "forget it"])
                steps.append(Step(t_int, "final", text, respond=True))
        elif kind == "field":
            img_a, img_b = rng.sample(sorted(IMAGES), 2)
            steps.append(Step(0.0, "frame", "What's wrong with this machine?", image=img_a,
                              respond=True))
            t_first = 0.0
            if variant == "swap":
                t_int = _offset(rng, phase, m)
                steps.append(Step(t_int, "frame", "Actually, what's wrong with this machine instead?",
                                  image=img_b, respond=True))
                expect = Expect(diagnosis_kb=IMAGES[img_b])
            else:
                expect = Expect(diagnosis_kb=IMAGES[img_a])
            tags["images"] = [img_a, img_b] if variant == "swap" else [img_a]
        else:  # access
            party = 4
            if variant == "hesitate_then_request":
                gap1, gap2 = rng.uniform(40, 500), rng.uniform(40, 500)
                steps += [Step(0.0, "partial", "I want to boo…"),
                          Step(gap1, "partial", "I want to boo… um…")]
                t_first = gap1 + gap2
                steps.append(Step(t_first, "final", "actually, book a table for 4", respond=True))
                expect = Expect(reservations=({"party_size": party},))
                tags["hesitation_gaps_ms"] = [round(gap1), round(gap2)]
            else:
                sp, t_first = _speech(rng, "Book a table for 4", 0.0, streamed)
                steps += sp
                t_int = t_first + _offset(rng, phase, m)
                if variant == "request_then_fix":
                    steps.append(Step(t_int, "final", rng.choice(
                        ["make it six", "Actually, make it six", "change it to 6"]), respond=True))
                    expect = Expect(reservations=({"party_size": 6},))
                else:
                    steps.append(Step(t_int, "final", rng.choice(["never mind", "stop", "cancel that"]),
                                      respond=True))

        # stray backchannels ("okay", "uh-huh") the agent should not react to
        if rng.random() < 0.35:
            t_noise = rng.uniform(0.3 * m, (t_int if t_int is not None else t_first) + 2 * m)
            steps.append(Step(t_noise, "final", rng.choice(NOISE)))
            tags["noise"] = True
        steps.sort(key=lambda s: s.at_ms)
        out.append(Scenario(id=i, kind=kind, variant=variant,
                            phase=phase if t_int is not None else None,
                            seed=rng.getrandbits(32), tool_latency_ms=latency,
                            steps=tuple(steps), expect=expect, first_final_ms=t_first,
                            interrupt_ms=t_int, tags=tags))
    return out
