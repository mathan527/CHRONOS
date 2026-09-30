"""The runner, the report, and the honesty of the output."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bench import run as bench_run
from bench.report import write_reports
from bench.run import RunEnv, hardware_info, run_mode, run_one
from bench.scenarios import generate
from chronos.config import Settings

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def mock_env() -> RunEnv:
    return RunEnv("mock", Settings(llm_mode="mock", db_path=":memory:", eou_silence_ms=300),
                  300, 30.0)


async def test_run_one_reports_every_field_for_every_agent():
    scn = generate(4, seed=11, latency_range_ms=(15, 30))[1]  # a support scenario
    for agent in ("chronos", "baseline", "baseline_strict"):
        r = await run_one(scn, agent, mock_env())
        assert r["agent"] == agent and r["scenario"] == scn.id and r["error"] is None
        assert set(r["stale"]) == {"started", "prevented", "compensated", "standing", "superseded"}
        assert r["ttfr_ms"] and all(x >= 0 for x in r["ttfr_ms"]) and r["no_response"] == 0
        assert r["wall_s"] > 0 and r["world_reads"] >= 0 and isinstance(r["consistent"], bool)
    assert (await run_one(scn, "chronos", mock_env()))["stale_emitted"] == 0


async def test_chronos_acks_fast_while_the_baseline_answers_after_its_tool_calls():
    scns = [s for s in generate(40, seed=12, latency_range_ms=(80, 100)) if s.kind == "support"][:3]
    chronos = [await run_one(s, "chronos", mock_env()) for s in scns]
    base = [await run_one(s, "baseline", mock_env()) for s in scns]
    c_first = [r["ttfr_ms"][0] for r in chronos]
    b_first = [r["ttfr_ms"][0] for r in base]
    assert max(c_first) < 50  # deterministic fast path
    assert min(b_first) > 60  # at least one search + one write (each ~40-150 ms) came first
    assert all(c < b for c, b in zip(c_first, b_first, strict=True))


async def test_a_fixed_seed_mini_benchmark_is_reproducible_and_chronos_stays_consistent():
    scns = generate(16, seed=21, latency_range_ms=(15, 40))
    data = await run_mode(mock_env(), scns, ["chronos", "baseline"], concurrency=4)
    assert data["status"] == "measured" and data["scenarios"] == 16
    c, b = data["agents"]["chronos"], data["agents"]["baseline"]
    assert c["errors"] == 0 and b["errors"] == 0
    assert c["consistent"]["k"] == 16, [r["problems"] for r in data["runs"]["chronos"]
                                        if not r["consistent"]]
    assert c["duplicate_live_writes"]["k"] == 0 and c["double_charge"]["k"] == 0
    assert c["stale_writes"]["standing"] == 0 and c["stale_outputs_emitted"] == 0
    assert b["consistent"]["k"] <= c["consistent"]["k"]
    assert c["ttfr_ms"]["p99"] < b["ttfr_ms"]["p99"]
    # the very same scenarios produce the very same world outcomes for the baseline
    again = await run_mode(mock_env(), scns, ["baseline"], concurrency=4)
    assert [r["consistent"] for r in again["runs"]["baseline"]] == [
        r["consistent"] for r in data["runs"]["baseline"]]
    assert [r["problems"] for r in again["runs"]["baseline"]] == [
        r["problems"] for r in data["runs"]["baseline"]]


# ------------------------------------------------------------------------------ the report ---
async def test_reports_are_generated_from_the_data_charts_included(tmp_path):
    data = await run_mode(mock_env(), generate(6, seed=31, latency_range_ms=(15, 30)),
                          ["chronos", "baseline"], concurrency=3)
    meta = {"generated_utc": "t", "command": "python -m bench.run --n 6", "seed": 31,
            "agents": ["chronos", "baseline"], "concurrency": 3,
            "tool_latency_mean_range_ms": [15, 30], "eou_silence_ms": 100,
            "hardware": hardware_info()}
    results = {"meta": meta, "modes": {"mock": data, "ollama": {
        "status": "skipped", "reason": "Ollama is not reachable at http://localhost:11434"}}}
    write_reports(results, tmp_path, charts=True)
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    p50 = data["agents"]["chronos"]["ttfr_ms"]["p50"]
    assert "# CHRONOS benchmark results" in md and "python -m bench.run --n 6" in md
    assert "Time to first response" in md and "95% Wilson" in md
    assert f"{p50:.1f} ms" in md or f"{p50:,.0f} ms" in md  # the figure comes from the data
    assert "Not measured" in md and "none were estimated" in md  # skipped mode is not faked
    assert "What was measured, and what these numbers do and do not mean" in md
    assert "Scenarios CHRONOS got wrong" in md and hardware_info()["python"] in md
    for name in ("mock_ttfr.png", "mock_correctness.png", "mock_stale_writes.png",
                 "mock_wall_clock.png"):
        png = tmp_path / "charts" / name
        assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 5000
        assert f"charts/{name}" in md
    assert not (tmp_path / "charts" / "ollama_ttfr.png").exists()


def test_the_report_changes_when_the_data_changes(tmp_path):
    def results(p50):
        agent = {"scenarios": 1, "errors": 0,
                 "ttfr_ms": {"n": 1, "no_response": 0, "p50": p50, "p95": p50, "p99": p50,
                             "mean": p50, "max": p50},
                 "wall_s": {"mean": 1.0, "p50": 1.0, "p95": 1.0},
                 "stale_writes": {"started": 0, "prevented": 0, "compensated": 0, "superseded": 0,
                                  "standing": 0, "runs_with_stale_writes": 0,
                                  **{f"{k}_pct": 0.0 for k in ("prevented", "compensated",
                                                               "superseded", "standing")},
                                  **{f"{k}_ci95": [0, 0] for k in ("prevented", "compensated",
                                                                    "superseded", "standing")}},
                 "consistent": {"k": 1, "n": 1, "pct": 100.0, "ci95": [20.7, 100.0]},
                 "duplicate_live_writes": {"k": 0, "n": 1, "pct": 0.0, "ci95": [0, 79.3]},
                 "double_charge": {"k": 0, "n": 1, "pct": 0.0, "ci95": [0, 79.3]},
                 "world_calls_per_scenario": {"reads": 1, "writes_committed": 1},
                 "stale_outputs_emitted": 0}
        grp = {"x": {"scenarios": 1, "consistent": agent["consistent"], "wall_s": agent["wall_s"],
                     "duplicate_live_writes": agent["duplicate_live_writes"],
                     "stale_writes": agent["stale_writes"], "ttfr_ms": agent["ttfr_ms"]}}
        mode = {"status": "measured", "scenarios": 1, "elapsed_s": 1, "agents": {"chronos": agent},
                "mix": {"kind": {"x": 1}, "variant": {"x/y": 1}, "phase": {"None": 1}},
                "by_kind": {"chronos": grp}, "by_phase": {"chronos": grp},
                "by_variant": {"chronos": grp}, "runs": {"chronos": []}}
        meta = {"generated_utc": "t", "command": "c", "seed": 0, "agents": ["chronos"],
                "concurrency": 1, "tool_latency_mean_range_ms": [1, 2], "eou_silence_ms": 1,
                "hardware": hardware_info()}
        return {"meta": meta, "modes": {"mock": mode}}

    write_reports(results(123.0), tmp_path / "a", charts=False)
    write_reports(results(456.0), tmp_path / "b", charts=False)
    a = (tmp_path / "a" / "results.md").read_text(encoding="utf-8")
    b = (tmp_path / "b" / "results.md").read_text(encoding="utf-8")
    assert "123 ms" in a and "456 ms" not in a and "456 ms" in b and "123 ms" not in b


# ------------------------------------------------------------------- CLI and Ollama handling --
async def test_ollama_pass_is_skipped_and_reported_when_it_is_not_available(monkeypatch):
    async def down(_settings, transport=None):
        return {"url": "http://localhost:11434", "reachable": False,
                "models": {"llama3.2:3b": False, "moondream": False}}

    monkeypatch.setattr(bench_run, "probe_ollama", down)
    args = bench_run.argparse.Namespace(agents="chronos", latency_range="20,40", seed=0,
                                        concurrency=1, silence_ms=100, llm="ollama", n=1,
                                        n_ollama=1)
    res = await bench_run.amain(args)
    assert res["modes"]["ollama"]["status"] == "skipped"
    assert "not reachable" in res["modes"]["ollama"]["reason"] and "mock" not in res["modes"]


async def test_ollama_pass_is_skipped_when_a_model_is_missing(monkeypatch):
    async def half(_settings, transport=None):
        return {"url": "u", "reachable": True, "models": {"llama3.2:3b": True, "moondream": False}}

    monkeypatch.setattr(bench_run, "probe_ollama", half)
    args = bench_run.argparse.Namespace(agents="chronos", latency_range="20,40", seed=0,
                                        concurrency=1, silence_ms=100, llm="ollama", n=1,
                                        n_ollama=1)
    res = await bench_run.amain(args)
    assert res["modes"]["ollama"]["status"] == "skipped"
    assert "moondream" in res["modes"]["ollama"]["reason"]


def test_unknown_agent_name_is_rejected():
    with pytest.raises(SystemExit):
        bench_run.main(["--agents", "chronos,nonsense", "--llm", "mock", "--n", "1"])


def test_command_line_end_to_end_writes_json_markdown_and_charts(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "bench.run", "--n", "6", "--seed", "3", "--llm",
                        "mock", "--latency-range", "15,30", "--concurrency", "3", "--out", str(out)],
                       capture_output=True, text=True, encoding="utf-8", timeout=300,
                       cwd=Path(__file__).resolve().parent.parent)
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["meta"]["seed"] == 3 and res["modes"]["mock"]["scenarios"] == 6
    assert set(res["modes"]["mock"]["agents"]) == {"chronos", "baseline", "baseline_strict"}
    assert (out / "results.md").exists() and len(list((out / "charts").glob("mock_*.png"))) == 4
