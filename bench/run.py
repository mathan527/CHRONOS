"""Benchmark runner.

    python -m bench.run                      # 200 scenarios, mock LLM, Ollama too if it is up
    python -m bench.run --n 50 --llm mock    # quick
    python -m bench.run --llm ollama --n-ollama 40

Every number in bench/results.json and bench/results.md is computed from this run's
measurements; nothing is hardcoded. The Ollama pass is skipped (and the report says so) when
Ollama or either model is not available.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.baseline_agent import BaselineAgent
from bench.instrument import RecordingWorld, WriteTap, tap_chronos_session
from bench.metrics import (
    check_state,
    duplicate_and_double_charge,
    group_by,
    live,
    snapshot_world,
    stale_stats,
    summarize,
    ttfr_ms,
)
from bench.scenarios import TODAY, Scenario, Step, generate
from chronos.agent.session import AgentSession
from chronos.api.server import probe_ollama
from chronos.config import Settings
from chronos.protocol import Event, EventType
from chronos.slowpath.llm import make_llm
from chronos.slowpath.vision import make_vision

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "scenarios" / "images"
AGENTS = {"chronos": "CHRONOS", "baseline": "Baseline", "baseline_strict": "Baseline (strict)"}


@dataclasses.dataclass(frozen=True)
class RunEnv:
    mode: str  # "mock" | "ollama"
    settings: Settings
    silence_ms: int
    wait_timeout_s: float
    today: Any = TODAY


def make_event(step: Step, session_id: str) -> Event:
    kind = {"partial": EventType.TRANSCRIPT_PARTIAL, "final": EventType.TRANSCRIPT_FINAL,
            "frame": EventType.CAMERA_FRAME}[step.kind]
    payload: dict[str, Any] = {"text": step.text}
    if step.kind == "frame":
        payload["path"] = str(IMAGES / str(step.image))
    return Event(session_id=session_id, type=kind, payload=payload)  # stamped with "now"


def last_diagnosis_kb(outputs: list[Any]) -> str | None:
    for m in reversed(outputs):
        diag = (m.data or {}).get("diagnosis")
        if diag:
            ids = diag.get("kb_ids") or []
            return ids[0] if ids else None
    return None


async def run_one(scn: Scenario, agent_name: str, env: RunEnv) -> dict[str, Any]:
    """Play one scenario against one agent in a fresh world and measure what happened."""
    base = {"scenario": scn.id, "kind": scn.kind, "variant": scn.variant, "phase": scn.phase,
            "agent": agent_name}
    with tempfile.TemporaryDirectory() as trace_dir:
        settings = dataclasses.replace(env.settings, trace_dir=trace_dir)
        world = await RecordingWorld.create(":memory:", seed=scn.seed,
                                            latency_ms=scn.tool_latency_ms, today=env.today)
        tap = WriteTap()
        sid = f"bench-{agent_name}-{scn.id}"
        llm, vision = make_llm(settings), make_vision(settings)
        chronos = agent_name == "chronos"
        if chronos:
            agent: Any = await AgentSession.create(sid, settings, llm=llm, vision=vision,
                                                   world=world, today=env.today)
            tap_chronos_session(agent, tap)
        else:
            agent = BaselineAgent(sid, world=world, llm=llm, vision=vision, today=env.today,
                                  silence_ms=env.silence_ms, tap=tap,
                                  filter_noise=agent_name == "baseline",
                                  llm_timeout_s=settings.llm_timeout_s)
        out_times: list[float] = []
        agent.subscribe(lambda _m: out_times.append(time.perf_counter()))
        agent.start()
        sends: list[float] = []
        error: str | None = None
        t0 = time.perf_counter()
        try:
            for step in scn.steps:
                await asyncio.sleep(max(0.0, t0 + step.at_ms / 1000 - time.perf_counter()))
                ev = make_event(step, sid)
                if step.respond:
                    sends.append(ev.ts_monotonic)
                await agent.submit(ev)
            try:
                await agent.wait_idle(env.wait_timeout_s)
            except TimeoutError:
                error = "timed out waiting for the agent to go idle"
        except Exception as e:  # noqa: BLE001 - a crashing agent is a result, not a harness bug
            error = f"{type(e).__name__}: {e}"

        state = await snapshot_world(world)
        ok, problems = check_state(scn.expect, state, last_diagnosis_kb(agent.outputs))
        duplicate, double = duplicate_and_double_charge(state)
        samples, missing = ttfr_ms(sends, out_times)
        stamps = [t for t in (world.last_activity(), max(out_times, default=None))
                  if t is not None]
        result = {
            **base, "consistent": ok and error is None, "problems": problems, "error": error,
            "live": live(state),
            "duplicate_live": duplicate, "double_charge": double,
            "stale": stale_stats(tap, world, scn.expect, state),
            "ttfr_ms": samples, "no_response": missing,
            "wall_s": (max(stamps) - t0) if stamps else None,
            "world_reads": sum(1 for c in world.calls if c.kind == "read"),
            "world_writes": len(world.committed("write")),
            "stale_emitted": agent.metrics.stale_emitted if chronos else None,
            "outputs": len(agent.outputs),
        }
        await agent.aclose()
        if not chronos:  # the session closes the world it was given; the baseline does not
            with contextlib.suppress(Exception):
                await world.close()
        return result


async def run_mode(env: RunEnv, scenarios: list[Scenario], agents: list[str],
                   concurrency: int) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    runs: dict[str, list[dict[str, Any]]] = {a: [] for a in agents}
    done = 0

    async def job(scn: Scenario) -> None:
        nonlocal done
        async with sem:
            for name in agents:  # alternate agents scenario by scenario
                runs[name].append(await run_one(scn, name, env))
        done += 1
        if done % 20 == 0 or done == len(scenarios):
            print(f"  [{env.mode}] {done}/{len(scenarios)} scenarios", file=sys.stderr, flush=True)

    await asyncio.gather(*(job(s) for s in scenarios))
    for rs in runs.values():
        rs.sort(key=lambda r: r["scenario"])
    return {
        "status": "measured", "llm_mode": env.mode, "scenarios": len(scenarios),
        "mix": {"kind": dict(Counter(s.kind for s in scenarios)),
                "variant": dict(Counter(f"{s.kind}/{s.variant}" for s in scenarios)),
                "phase": dict(Counter(str(s.phase) for s in scenarios))},
        "agents": {a: summarize(runs[a]) for a in agents},
        "by_kind": {a: group_by(runs[a], "kind") for a in agents},
        "by_phase": {a: group_by(runs[a], "phase") for a in agents},
        "by_variant": {a: group_by([{**r, "vk": f"{r['kind']}/{r['variant']}"} for r in runs[a]],
                                   "vk") for a in agents},
        "runs": runs,
    }


# ------------------------------------------------------------------------------ environment ----
def _cpu_name() -> str:
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    return platform.processor() or platform.machine()


def _total_ram_gb() -> float | None:
    with contextlib.suppress(Exception):
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return round(st.ullTotalPhys / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    return None


def hardware_info() -> dict[str, Any]:
    import fastapi
    import pydantic
    return {"os": platform.platform(), "cpu": _cpu_name(), "logical_cores": os.cpu_count(),
            "ram_gb": _total_ram_gb(), "gpu": "not probed",
            "python": platform.python_version(), "pydantic": pydantic.__version__,
            "fastapi": fastapi.__version__}


def parse_latency(text: str) -> tuple[int, int]:
    lo, hi = (int(x) for x in text.split(","))
    return lo, hi


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = set(agents) - set(AGENTS)
    if unknown:
        raise SystemExit(f"unknown agent(s): {sorted(unknown)}; choose from {sorted(AGENTS)}")
    latency = parse_latency(args.latency_range)
    results: dict[str, Any] = {"meta": {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python -m bench.run " + " ".join(sys.argv[1:]),
        "seed": args.seed, "agents": agents, "concurrency": args.concurrency,
        "tool_latency_mean_range_ms": list(latency), "eou_silence_ms": args.silence_ms,
        "today": TODAY.isoformat(), "hardware": hardware_info()}, "modes": {}}

    if args.llm in ("mock", "both"):
        print(f"mock LLM: {args.n} scenarios x {len(agents)} agents", file=sys.stderr)
        env = RunEnv("mock", Settings(llm_mode="mock", db_path=":memory:",
                                      eou_silence_ms=args.silence_ms), args.silence_ms, 30.0)
        scenarios = generate(args.n, args.seed, latency)
        t = time.perf_counter()
        results["modes"]["mock"] = await run_mode(env, scenarios, agents, args.concurrency)
        results["modes"]["mock"]["elapsed_s"] = round(time.perf_counter() - t, 1)

    if args.llm in ("ollama", "both"):
        settings = Settings(llm_mode="ollama", db_path=":memory:", eou_silence_ms=args.silence_ms,
                            llm_timeout_s=60.0)
        probe = await probe_ollama(settings)
        results["meta"]["ollama_probe"] = probe
        if not probe["reachable"]:
            results["modes"]["ollama"] = {"status": "skipped",
                                          "reason": f"Ollama is not reachable at {probe['url']}"}
        elif not all(probe["models"].values()):
            missing = [m for m, ok in probe["models"].items() if not ok]
            results["modes"]["ollama"] = {"status": "skipped",
                                          "reason": f"Ollama is up but models are missing: {missing}"}
        else:
            print(f"Ollama: {args.n_ollama} scenarios x {len(agents)} agents", file=sys.stderr)
            env = RunEnv("ollama", settings, args.silence_ms, 300.0)
            t = time.perf_counter()
            results["modes"]["ollama"] = await run_mode(
                env, generate(args.n_ollama, args.seed, latency), agents, args.concurrency)
            results["modes"]["ollama"]["elapsed_s"] = round(time.perf_counter() - t, 1)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="scenarios for the mock-LLM pass")
    ap.add_argument("--n-ollama", type=int, default=40, help="scenarios for the Ollama pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", choices=["mock", "ollama", "both"], default="both")
    ap.add_argument("--agents", default="chronos,baseline,baseline_strict")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="scenarios in flight at once (1 = cleanest timing)")
    ap.add_argument("--silence-ms", type=int, default=300, help="end-of-utterance silence")
    ap.add_argument("--latency-range", default="20,120",
                    help="per-scenario MEAN tool latency is drawn uniformly from this range (ms)")
    ap.add_argument("--out", type=Path, default=REPO / "bench")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    results = asyncio.run(amain(args))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                           encoding="utf-8")
    from bench.report import write_reports
    write_reports(results, args.out, charts=not args.no_charts)
    print(f"wrote {args.out / 'results.json'}, results.md" + ("" if args.no_charts else ", charts/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
