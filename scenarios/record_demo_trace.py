"""Runs all four demo scenarios in ONE session and writes a standalone timeline page.

    python scenarios/record_demo_trace.py            # -> traces/demo_all.jsonl + traces/demo_all.html

No server, no Ollama (mock LLM), and it works offline: open the HTML file in any browser.
Latency is realistic-but-short so the interruptions land mid-task.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from chronos.agent.session import AgentSession
from chronos.config import Settings
from chronos.protocol import Event, EventType
from chronos.trace import export

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "scenarios" / "images"


async def main() -> None:
    out = ROOT / "traces"
    out.mkdir(exist_ok=True)
    (out / "demo_all.jsonl").unlink(missing_ok=True)
    settings = Settings(llm_mode="mock", trace_dir=str(out), db_path=":memory:",
                        tool_latency_ms=(120, 320), eou_silence_ms=300)
    s = await AgentSession.create("demo_all", settings)
    s.start()

    async def send(kind: EventType, **payload: object) -> None:
        await s.submit(Event(session_id="demo_all", type=kind, payload=payload))

    async def say(text: str) -> None:
        await send(EventType.TRANSCRIPT_FINAL, text=text)

    # 1. in-car: change of goal while the route is still being calculated
    await say("Navigate to Chennai Airport")
    await asyncio.sleep(0.08)
    await say("Actually, gas station first")
    await s.wait_idle()
    # 2. customer support: booking committed, then changed (compensating action)
    await say("Book a flight to Delhi tomorrow")
    await s.wait_idle()
    await say("Actually, next week instead")
    await asyncio.sleep(0.05)
    await say("okay")  # a backchannel: traced and ignored
    await s.wait_idle()
    # 3. field troubleshooting: camera frame + question
    await send(EventType.CAMERA_FRAME, path=str(IMAGES / "machine_overheating_fan.png"),
               text="What's wrong with this machine?")
    await s.wait_idle()
    # 4. accessibility: hesitation, then a self-correction
    await send(EventType.TRANSCRIPT_PARTIAL, text="I want to boo…")
    await asyncio.sleep(0.05)
    await send(EventType.TRANSCRIPT_PARTIAL, text="I want to boo… um…")
    await asyncio.sleep(0.05)
    await say("actually, book a table for 4")
    await s.wait_idle()

    state = await s.describe()
    await s.aclose()
    trace = out / "demo_all.jsonl"
    export.main([str(trace), "-o", str(out / "demo_all.html")])
    print(f"epochs: 1 -> {state['epoch']}, stale outputs emitted: {state['metrics']['stale_emitted']}, "
          f"world violations: {state['world']['invariant_violations']}")


if __name__ == "__main__":
    asyncio.run(main())
