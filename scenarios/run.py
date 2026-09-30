"""Plays scenario YAML files against an in-process session and checks their `expect` block.

    python scenarios/run.py                      # all scenarios, mock LLM (no Ollama needed)
    python scenarios/run.py incar field          # just these
    python scenarios/run.py --llm ollama         # local llama3.2:3b + moondream
    python scenarios/run.py --html               # also write traces/scn_<name>.html timelines

Step kinds: say, partial, frame (+text), wait (ms), wait_for_idle.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path
from typing import Any

import yaml

from chronos.agent.session import AgentSession
from chronos.config import Settings
from chronos.protocol import Event, EventType
from chronos.trace import export

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "scenarios"
ORDER = ["incar", "support_mid", "support_after", "field", "access"]


def load(name: str) -> dict[str, Any]:
    return yaml.safe_load((HERE / f"{name}.yaml").read_text(encoding="utf-8"))


def all_names() -> list[str]:
    return ORDER + sorted(p.stem for p in HERE.glob("*.yaml") if p.stem not in ORDER)


def check(spec: dict[str, Any], state: dict[str, Any]) -> list[str]:
    exp, world, problems = spec.get("expect", {}), state["world"], []
    if "epoch" in exp and state["epoch"] != exp["epoch"]:
        problems.append(f"epoch {state['epoch']}, expected {exp['epoch']}")
    for kind, want in exp.get("live", {}).items():
        got = len(world[kind])
        if got != want:
            problems.append(f"{got} live {kind}, expected {want}")
    if "charged" in exp:
        got = sum(1 for c in world["charges"] if c.get("status") == "CHARGED")
        if got != exp["charged"]:
            problems.append(f"{got} charges, expected {exp['charged']}")
    if "ledger_status" in exp:
        got = sorted(r["status"] for r in state["ledger"]
                     if r["tool"] in ("book_flight", "set_navigation", "reserve_table"))
        if got != sorted(exp["ledger_status"]):
            problems.append(f"ledger {got}, expected {sorted(exp['ledger_status'])}")
    if world["invariant_violations"]:
        problems.append(f"invariants: {world['invariant_violations']}")
    if state["metrics"]["stale_emitted"]:
        problems.append("a stale-epoch output was emitted")
    return problems


async def play(name: str, llm: str, html: bool, verbose: bool) -> list[str]:
    spec = load(name)
    lo, hi = spec.get("tool_latency_ms", [120, 320])
    out = ROOT / "traces"
    out.mkdir(exist_ok=True)
    sid = f"scn_{name}"
    (out / f"{sid}.jsonl").unlink(missing_ok=True)
    settings = dataclasses.replace(Settings.from_env(), llm_mode=llm, db_path=":memory:",
                                   tool_latency_ms=(lo, hi), eou_silence_ms=300,
                                   trace_dir=str(out))
    s = await AgentSession.create(sid, settings)
    s.start()

    async def send(kind: EventType, **payload: object) -> None:
        await s.submit(Event(session_id=sid, type=kind, payload=payload))

    print(f"\n== {spec['title']}")
    for step in spec["steps"]:
        if "say" in step:
            print(f"   you: {step['say']}")
            await send(EventType.TRANSCRIPT_FINAL, text=step["say"])
        elif "partial" in step:
            print(f"   you (partial): {step['partial']}")
            await send(EventType.TRANSCRIPT_PARTIAL, text=step["partial"])
        elif "frame" in step:
            print(f"   camera: {step['frame']}  {step.get('text', '')}")
            await send(EventType.CAMERA_FRAME, path=str(HERE / step["frame"]),
                       text=step.get("text", ""))
        elif "wait" in step:
            await asyncio.sleep(step["wait"] / 1000)
        elif step.get("wait_for_idle"):
            await s.wait_idle(60)
    await s.wait_idle(60)
    state = await s.describe()
    await s.aclose()
    for m in state["outputs"]:
        if verbose or m["type"] in ("ack", "response", "action_result"):
            print(f"   [{m['type']:13s} e{m['epoch']} {m['status']:9s}] {m['text']}")
    problems = check(spec, state)
    print("   " + ("PASS" if not problems else "FAIL: " + "; ".join(problems)))
    if html:
        export.main([str(out / f"{sid}.jsonl"), "-o", str(out / f"{sid}.html")])
    return problems


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--html", action="store_true", help="write traces/scn_<name>.html timelines")
    ap.add_argument("-v", "--verbose", action="store_true", help="show progress messages too")
    a = ap.parse_args(argv)
    failed = 0
    for name in a.names or all_names():
        failed += bool(await play(name, a.llm, a.html, a.verbose))
    print(f"\n{'all scenarios passed' if not failed else f'{failed} scenario(s) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
