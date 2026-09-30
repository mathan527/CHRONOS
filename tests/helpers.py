"""Test driver + protocol-level assertions shared by the session and end-to-end tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronos.agent.session import AgentSession
from chronos.protocol import Event, EventType, OutputMessage, OutputType

IMAGES = Path(__file__).parent.parent / "scenarios" / "images"
ACK_BUDGET_MS = 300.0


class Driver:
    """Plays a user against an AgentSession."""

    def __init__(self, session: AgentSession) -> None:
        self.s = session

    async def send(self, type_: EventType, payload: dict[str, Any]) -> Event:
        return await self.s.submit(Event(session_id=self.s.session_id, type=type_,
                                         payload=payload))

    async def say(self, text: str) -> Event:  # a complete (final) utterance
        return await self.send(EventType.TRANSCRIPT_FINAL, {"text": text})

    async def partial(self, text: str) -> Event:
        return await self.send(EventType.TRANSCRIPT_PARTIAL, {"text": text})

    async def text(self, text: str) -> Event:
        return await self.send(EventType.TEXT, {"text": text})

    async def frame(self, path: Path | str, text: str | None = None) -> Event:
        payload: dict[str, Any] = {"path": str(path)}
        if text:
            payload["text"] = text
        return await self.send(EventType.CAMERA_FRAME, payload)

    async def interrupt(self, **payload: Any) -> Event:
        return await self.send(EventType.INTERRUPT, payload)

    async def idle(self, timeout: float = 10.0) -> None:
        await self.s.wait_idle(timeout)

    # ------------------------------------------------------------------- observations -------
    @property
    def outs(self) -> list[OutputMessage]:
        return self.s.outputs

    def of(self, type_: OutputType) -> list[OutputMessage]:
        return [m for m in self.outs if m.type is type_]

    def acks(self) -> list[OutputMessage]:
        return self.of(OutputType.ACK)

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(x) for x in self.s.tracer.path.read_text(encoding="utf-8").splitlines()]

    def events(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.rows() if r["event"] == name]

    async def ledger(self, tool: str, status: str | None = None) -> list[dict[str, Any]]:
        rows = [r for r in await self.s.ledger.all_rows(self.s.session_id) if r["tool"] == tool]
        return [r for r in rows if status is None or r["status"] == status]

    async def state(self) -> dict[str, Any]:
        return await self.s.describe()


def assert_acks_fast(d: Driver, expected: int) -> None:
    lat = d.s.metrics.ack_latencies_ms
    assert len(lat) == expected, f"expected {expected} acks, got {len(lat)}: {lat}"
    assert max(lat, default=0) < ACK_BUDGET_MS, f"ack latencies (ms): {lat}"


def assert_no_stale_output(d: Driver) -> None:
    """Two independent checks that nothing from a superseded epoch ever reached the client:
    the session's own counter, and a replay of the trace file (every ack/response must carry the
    epoch that was current when it was sent)."""
    assert d.s.metrics.stale_emitted == 0
    epoch = 1
    for r in d.rows():
        if r["event"] == "epoch_bumped":
            epoch = r["epoch"]
        elif r["event"] in ("ack_sent", "response_sent"):
            assert r["epoch"] == epoch, f"stale output emitted: {r} (current epoch {epoch})"


def assert_ack_precedes_slow_work(d: Driver) -> None:
    """Within every epoch, the fast-path ack is emitted before any slow-path output."""
    first_ack: dict[int, int] = {}
    first_slow: dict[int, int] = {}
    for i, m in enumerate(d.outs):
        (first_ack if m.type is OutputType.ACK else first_slow).setdefault(m.epoch, i)
    for epoch, i in first_slow.items():
        if epoch in first_ack:
            assert first_ack[epoch] < i, f"slow output before the ack in epoch {epoch}"


async def assert_world_consistent(d: Driver) -> dict[str, Any]:
    state = await d.state()
    assert state["world"]["invariant_violations"] == []
    json.dumps(state)  # the state endpoint payload must be plain JSON
    return state
