import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pytest
import yaml

from chronos.perception.bargein import BargeInClassifier, BargeInContext, tier1
from chronos.perception.intent import Intent
from chronos.protocol import BargeInType as B
from chronos.slowpath.llm import LLMError, MockLLM

TODAY = date(2026, 10, 1)
CASES = yaml.safe_load((Path(__file__).parent / "data" / "bargein_cases.yaml").read_text("utf-8"))


def ctx_of(case) -> BargeInContext | None:
    return BargeInContext(Intent(case["context"])) if case.get("context") else None


@pytest.fixture
def clf():
    return BargeInClassifier(MockLLM(), today=TODAY)


def test_dataset_is_big_and_covers_every_class():
    assert len(CASES) >= 80
    by = Counter(c["label"] for c in CASES)
    assert set(by) == {t.value for t in B}
    assert all(n >= 8 for n in by.values()), by


async def test_accuracy_per_class_report(clf, capsys):
    """Tier 1 + mock LLM on the labelled set. Prints a per-class report."""
    hits, tot = defaultdict(int), defaultdict(int)
    tier_count, misses, lat1 = Counter(), [], []
    for c in CASES:
        r = await clf.classify(c["text"], final=c.get("final", True), context=ctx_of(c))
        tot[c["label"]] += 1
        tier_count[r.tier] += 1
        if r.tier == 1:
            lat1.append(r.latency_ms)
        if r.label.value == c["label"]:
            hits[c["label"]] += 1
        else:
            misses.append((c["text"], c["label"], r.label.value, r.tier, r.reason))
    n, ok = len(CASES), sum(hits.values())
    with capsys.disabled():
        print(f"\n=== barge-in accuracy: {ok}/{n} = {ok / n:.1%} "
              f"(tier1 {tier_count[1]}, tier2 {tier_count[2]}, fallback {tier_count[0]}) ===")
        for label in sorted(tot):
            print(f"  {label:24s} {hits[label]:3d}/{tot[label]:<3d} {hits[label] / tot[label]:.0%}")
        for m in misses:
            print("  MISS", m)
    assert ok / n >= 0.90, f"overall accuracy {ok / n:.1%}"
    for label in tot:
        assert hits[label] / tot[label] >= 0.80, f"{label}: {hits[label]}/{tot[label]}"
    assert tier_count[1] / n >= 0.85  # the LLM is the exception, not the rule


def test_tier1_latency_under_5ms():
    lats = []
    for c in CASES * 3:
        t0 = time.perf_counter()
        tier1(c["text"], final=c.get("final", True), context=ctx_of(c), today=TODAY)
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    assert lats[int(len(lats) * 0.99)] < 5.0, f"p99={lats[int(len(lats) * 0.99)]:.2f}ms"


def test_epoch_bump_semantics_follow_labels():
    assert [tier1(t, today=TODAY).label.bumps_epoch for t in
            ("make it 6pm", "stop", "actually, gas station first", "okay", "um",
             "and add a window seat", "what time is it?")] == [
        True, True, True, False, False, False, False]


def test_spec_examples():
    d = lambda t, **k: tier1(t, today=TODAY, **k).label
    assert d("make it 6pm") is B.CORRECTION
    assert d("actually, gas station first") is B.GOAL_CHANGE
    assert d("and add a window seat") is B.ADDITION
    assert d("stop") is B.CANCEL and d("never mind") is B.CANCEL
    assert d("uh-huh") is B.BACKCHANNEL and d("okay") is B.BACKCHANNEL
    assert d("I want to boo...", final=False) is B.HESITATION and d("um") is B.HESITATION


def test_stop_at_is_not_cancel_and_negated_cancel_is_not_cancel():
    assert tier1("stop at the gas station", today=TODAY).label is B.ADDITION
    d = tier1("don't cancel it", today=TODAY)
    assert d is None or d.label is not B.CANCEL


def test_context_separates_correction_from_goal_change():
    nav = BargeInContext(Intent.NAVIGATE)
    assert tier1("take me to Marina Beach instead", context=nav, today=TODAY).label is B.CORRECTION
    flight = BargeInContext(Intent.BOOK_FLIGHT)
    assert tier1("take me to Marina Beach instead", context=flight,
                 today=TODAY).label is B.GOAL_CHANGE


def test_complete_utterance_with_trailing_dots_not_hesitation_when_final():
    assert tier1("make it 6pm...", final=True, today=TODAY).label is B.CORRECTION
    assert tier1("make it 6pm...", final=False, today=TODAY).label is B.HESITATION


def test_tier1_abstains_on_unclear_input():
    assert tier1("the flight was nice last time", today=TODAY) is None


# ----------------------------------------------------------------------- tier 2 ------------
class CountingLLM(MockLLM):
    pass


async def test_tier1_decision_never_calls_llm():
    llm = MockLLM()
    r = await BargeInClassifier(llm, today=TODAY).classify("make it 6pm")
    assert r.tier == 1 and llm.calls == []


async def test_unsure_goes_to_tier2_and_reports_tier_and_latency():
    llm = MockLLM()
    r = await BargeInClassifier(llm, today=TODAY).classify("the flight was nice last time")
    assert r.tier == 2 and len(llm.calls) == 1 and r.latency_ms >= 0 and r.reason == "llm"


async def test_llm_timeout_falls_back_to_hesitation_within_budget():
    slow = MockLLM(latency_s=2.0)
    t0 = time.perf_counter()
    r = await BargeInClassifier(slow, timeout_s=0.25, today=TODAY).classify(
        "the flight was nice last time")
    elapsed = time.perf_counter() - t0
    assert r.label is B.HESITATION and r.tier == 0 and r.reason == "fallback:llm_timeout"
    assert elapsed < 0.6  # never freezes
    assert not r.bumps_epoch


@pytest.mark.parametrize("bad,reason", [
    ({"label": "explode"}, "llm_invalid"), ({"nolabel": 1}, "llm_invalid"),
    ({"label": None}, "llm_invalid")])
async def test_llm_invalid_output_falls_back(bad, reason):
    llm = MockLLM(handlers={"bargein": lambda _p: bad})
    r = await BargeInClassifier(llm, today=TODAY).classify("the flight was nice last time")
    assert r.tier == 0 and r.reason == f"fallback:{reason}" and r.label is B.HESITATION


async def test_llm_error_falls_back():
    def boom(_p):
        raise LLMError("down")
    r = await BargeInClassifier(MockLLM(handlers={"bargein": boom}), today=TODAY).classify(
        "the flight was nice last time")
    assert r.tier == 0 and r.reason == "fallback:llm_error"


async def test_no_llm_configured_falls_back():
    r = await BargeInClassifier(None, today=TODAY).classify("the flight was nice last time")
    assert r.tier == 0 and r.label is B.HESITATION


HOLDOUT = yaml.safe_load(
    (Path(__file__).parent / "data" / "bargein_holdout.yaml").read_text("utf-8"))


async def test_holdout_generalisation_report(clf, capsys):
    hits, tot, misses = defaultdict(int), defaultdict(int), []
    for c in HOLDOUT:
        r = await clf.classify(c["text"], final=c.get("final", True), context=ctx_of(c))
        tot[c["label"]] += 1
        if r.label.value == c["label"]:
            hits[c["label"]] += 1
        else:
            misses.append((c["text"], c["label"], r.label.value, r.tier, r.reason))
    n, ok = len(HOLDOUT), sum(hits.values())
    with capsys.disabled():
        print(f"\n=== HOLDOUT accuracy: {ok}/{n} = {ok / n:.1%} ===")
        for label in sorted(tot):
            print(f"  {label:24s} {hits[label]:3d}/{tot[label]:<3d}")
        for m in misses:
            print("  MISS", m)
    assert ok / n >= 0.80, f"holdout accuracy {ok / n:.1%}"
