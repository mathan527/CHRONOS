import asyncio

from chronos.config import Settings
from chronos.perception.eou import EndOfUtteranceDetector

SILENCE_MS = 40
WAIT = SILENCE_MS / 1000 * 4


def make(silence_ms=SILENCE_MS):
    commits, holds = [], []

    async def on_commit(t):
        commits.append(t)

    async def on_hold(t):
        holds.append(t)

    return EndOfUtteranceDetector(on_commit, silence_ms=silence_ms, on_hold=on_hold), commits, holds


def test_default_silence_is_700ms():
    assert Settings().eou_silence_ms == 700
    assert Settings.from_env().eou_silence_ms == 700


async def test_commits_only_after_silence():
    d, commits, _ = make()
    await d.feed_partial("book a table for four")
    await asyncio.sleep(SILENCE_MS / 1000 / 2)
    assert commits == []  # still inside the silence window
    await asyncio.sleep(WAIT)
    assert commits == ["book a table for four"] and d.pending == ""


async def test_rapid_partials_reset_the_timer_and_commit_once():
    d, commits, _ = make()
    for t in ("book", "book a", "book a table", "book a table for four"):
        await d.feed_partial(t)
        await asyncio.sleep(SILENCE_MS / 1000 / 3)  # faster than the silence window
    assert commits == []
    await asyncio.sleep(WAIT)
    assert commits == ["book a table for four"]


async def test_hesitation_is_never_committed():
    d, commits, holds = make()
    await d.feed_partial("I want to boo...")
    await asyncio.sleep(WAIT * 2)
    assert commits == [] and holds == ["I want to boo..."]
    assert d.pending == "I want to boo..."  # held, not dropped


async def test_hesitation_then_self_correction_commits_one_utterance():
    """'I want to boo… um… actually, book a table for 4' -> exactly one commit."""
    d, commits, holds = make()
    await d.feed_partial("I want to boo...")
    await asyncio.sleep(WAIT)
    await d.feed_partial("I want to boo... um...")
    await asyncio.sleep(WAIT)
    assert commits == [] and len(holds) == 2
    await d.feed_partial("actually, book a table for 4")
    await asyncio.sleep(WAIT)
    assert commits == ["actually, book a table for 4"]


async def test_final_commits_immediately_and_cancels_pending_timer():
    d, commits, _ = make()
    await d.feed_partial("book a table for four")
    await d.feed_final("book a table for four please")
    assert commits == ["book a table for four please"]
    await asyncio.sleep(WAIT)
    assert commits == ["book a table for four please"]  # timer did not double-commit


async def test_final_hesitation_is_dropped_not_committed():
    d, commits, holds = make()
    await d.feed_final("um")
    assert commits == [] and holds == ["um"]


async def test_backchannel_final_still_commits_for_the_classifier_to_handle():
    d, commits, _ = make()
    await d.feed_final("okay")
    assert commits == ["okay"]  # not hesitation: classifier will label it BACKCHANNEL


async def test_aclose_cancels_pending_commit():
    d, commits, _ = make()
    await d.feed_partial("book a table for four")
    await d.aclose()
    await asyncio.sleep(WAIT)
    assert commits == [] and d.pending == ""


async def test_partial_arriving_during_commit_is_not_lost():
    commits = []
    gate = asyncio.Event()

    async def slow_commit(t):
        commits.append(t)
        await gate.wait()

    d = EndOfUtteranceDetector(slow_commit, silence_ms=SILENCE_MS)
    await d.feed_partial("book a table for four")
    await asyncio.sleep(WAIT)  # commit is now in progress (blocked on gate)
    await d.feed_partial("make it six")
    gate.set()
    await asyncio.sleep(WAIT)
    assert commits == ["book a table for four", "make it six"]
