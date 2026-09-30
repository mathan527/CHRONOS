"""Shared fixtures for session / end-to-end / API tests."""
from __future__ import annotations

from datetime import date

import pytest

from chronos.agent.session import AgentSession
from chronos.config import Settings

TODAY = date(2026, 10, 1)


@pytest.fixture
def fast_settings(tmp_path) -> Settings:
    """Mock LLM, in-memory ledger, fast-but-real tool latency, short end-of-utterance silence."""
    return Settings(llm_mode="mock", trace_dir=str(tmp_path / "traces"), db_path=":memory:",
                    tool_latency_ms=(30, 80), eou_silence_ms=120)


@pytest.fixture
async def open_session(fast_settings):
    """`await open_session("id", llm=..., policy=..., start=False)` -> AgentSession (closed after)."""
    made: list[AgentSession] = []

    async def _open(session_id: str = "s1", *, start: bool = True, **kw) -> AgentSession:
        s = await AgentSession.create(session_id, kw.pop("settings", fast_settings), today=TODAY,
                                      **kw)
        if start:
            s.start()
        made.append(s)
        return s

    yield _open
    for s in made:
        await s.aclose()
