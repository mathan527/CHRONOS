"""Cross-platform task runner behind the Makefile (works on Windows without `make`).

    python scripts/tasks.py setup    # create .venv, install CHRONOS + dev tools
    python scripts/tasks.py test     # pytest -q, then ruff check
    python scripts/tasks.py run      # start the API + web client on http://localhost:8000
    python scripts/tasks.py bench    # full benchmark -> bench/results.{json,md} + charts
    python scripts/tasks.py demo     # play the 4 scenarios, write timelines, then serve the UI
    python scripts/tasks.py docker   # docker compose up --build (Ollama + models + CHRONOS)

Extra arguments after the task name are passed through (e.g. `bench --n 50 --llm mock`).
The LLM defaults to the deterministic mock; set CHRONOS_LLM=ollama to use llama3.2:3b.
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")
PY = BIN / ("python.exe" if os.name == "nt" else "python")
PORT = os.getenv("PORT", "8000")


def sh(*cmd: str | Path, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=ROOT, env={**os.environ, **(env or {})})


def need_venv() -> None:
    if not PY.exists():
        sys.exit("No .venv yet. Run: python scripts/tasks.py setup")


def setup(_extra: list[str]) -> int:
    if not PY.exists():
        print(f"creating {VENV}")
        venv.create(VENV, with_pip=True)
    rc = sh(PY, "-m", "pip", "install", "-q", "--upgrade", "pip")
    rc = rc or sh(PY, "-m", "pip", "install", "-q", "-e", ".[dev]")
    if rc == 0:
        print("\nsetup done. For the local models (optional; mock mode needs none):\n"
              "  ollama pull llama3.2:3b && ollama pull moondream\n"
              "then run with CHRONOS_LLM=ollama")
    return rc


def test(extra: list[str]) -> int:
    need_venv()
    rc = sh(PY, "-m", "pytest", "-q", *extra)
    return rc or sh(PY, "-m", "ruff", "check", ".")


def serve() -> int:
    need_venv()
    print(f"\nCHRONOS UI:  http://localhost:{PORT}/\nhealth:      http://localhost:{PORT}/health\n"
          "traces:      http://localhost:%s/traces/<session_id>/view\n" % PORT, flush=True)
    return sh(PY, "-m", "uvicorn", "chronos.api.server:create_app", "--factory",
              "--host", "127.0.0.1", "--port", PORT)


def run(_extra: list[str]) -> int:
    return serve()


def bench(extra: list[str]) -> int:
    need_venv()
    return sh(PY, "-m", "bench.run", *(extra or ["--n", "200", "--seed", "0", "--llm", "both",
                                                  "--concurrency", "1", "--out", "bench"]))


def demo(_extra: list[str]) -> int:
    need_venv()
    env = {"PYTHONIOENCODING": "utf-8"}
    rc = sh(PY, "scenarios/run.py", "--html", env=env)
    rc = rc or sh(PY, "scenarios/record_demo_trace.py", env=env)
    if rc:
        return rc
    print("\nOffline timelines: traces/demo_all.html (all four scenarios in one session)\n"
          "                   traces/scn_<name>.html (one per scenario)")
    # Demo pacing: every tool call takes 350-700 ms, so the scripted interruptions always land
    # mid-task (the default 100-800 ms is random enough that some runs finish first).
    os.environ.setdefault("CHRONOS_TOOL_LATENCY_MS", "350,700")
    print(f"\nPresenter mode: http://localhost:{PORT}/?present=1   (keys 1-5 play the scenarios)")
    return serve()


def docker(extra: list[str]) -> int:
    return sh("docker", "compose", "up", "--build", *extra)


TASKS = {"setup": setup, "test": test, "run": run, "bench": bench, "demo": demo, "docker": docker}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TASKS:
        print(__doc__)
        return 2
    return TASKS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
