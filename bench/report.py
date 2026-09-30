"""results.md + PNG charts, generated from results.json (every figure is a measurement)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

LABEL = {"chronos": "CHRONOS", "baseline": "Baseline", "baseline_strict": "Baseline (strict)"}
COLOR = {"chronos": "#0072B2", "baseline": "#E69F00", "baseline_strict": "#D55E00"}  # Okabe-Ito


# --------------------------------------------------------------------------- formatting -------
def ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.0f} ms" if v >= 100 else f"{v:.1f} ms"


def secs(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f} s"


def rate(r: dict[str, Any]) -> str:
    lo, hi = r["ci95"]
    return f"{r['pct']:.1f}% ({r['k']}/{r['n']}) [{lo:.0f}-{hi:.0f}]"


def stale_pct(s: dict[str, Any], name: str) -> str:
    if not s["started"]:
        return "n/a"
    lo, hi = s[f"{name}_ci95"]
    return f"{s[f'{name}_pct']:.1f}% ({s[name]}/{s['started']}) [{lo:.0f}-{hi:.0f}]"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------- charts -------
def _chart_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _bar_labels(ax: Any, bars: Any, fmt: str) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                    fontsize=8, xytext=(0, 2), textcoords="offset points")


def make_charts(mode: str, data: dict[str, Any], out_dir: Path) -> list[str]:
    plt = _chart_setup()
    agents = list(data["agents"])
    n = data["scenarios"]
    sub = f"{mode} LLM, {n} seeded scenarios"
    files: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        files.append(name)

    # 1. time to first response
    fig, ax = plt.subplots(figsize=(7, 4))
    qs = ["p50", "p95", "p99"]
    w = 0.8 / len(agents)
    for i, a in enumerate(agents):
        vals = [data["agents"][a]["ttfr_ms"][q] or 0 for q in qs]
        bars = ax.bar([x + i * w for x in range(len(qs))], vals, w, label=LABEL[a], color=COLOR[a])
        _bar_labels(ax, bars, "{:.0f}")
    ax.set_xticks([x + w * (len(agents) - 1) / 2 for x in range(len(qs))], qs)
    ax.set_yscale("log")
    ax.set_ylabel("ms (log scale)")
    ax.set_title(f"Time to first response, lower is better\n{sub}")
    ax.legend()
    save(fig, f"{mode}_ttfr.png")

    # 2. safety metrics (percent of scenarios)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["final state\nconsistent", "duplicate live\nwrites", "double\ncharge"]
    for i, a in enumerate(agents):
        s = data["agents"][a]
        vals = [s["consistent"]["pct"], s["duplicate_live_writes"]["pct"], s["double_charge"]["pct"]]
        bars = ax.bar([x + i * w for x in range(3)], vals, w, label=LABEL[a], color=COLOR[a])
        _bar_labels(ax, bars, "{:.1f}%")
    ax.set_xticks([x + w * (len(agents) - 1) / 2 for x in range(3)], labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of scenarios")
    ax.set_title(f"Correctness of the final world state\n{sub}")
    ax.legend(loc="center right")
    save(fig, f"{mode}_correctness.png")

    # 3. what happened to stale writes
    fig, ax = plt.subplots(figsize=(7, 4))
    parts = [("prevented", "#009E73", "stopped before dispatch"), ("compensated", "#56B4E9",
             "committed, then undone"), ("superseded", "#CCCCCC", "committed, replaced by later write"),
             ("standing", "#D55E00", "committed and still live")]
    bottoms = [0.0] * len(agents)
    for key, colour, text in parts:
        vals = [data["agents"][a]["stale_writes"][f"{key}_pct"] for a in agents]
        ax.bar([LABEL[a] for a in agents], vals, bottom=bottoms, label=text, color=colour)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    for i, a in enumerate(agents):
        ax.annotate(f"n={data['agents'][a]['stale_writes']['started']}", (i, 101), ha="center",
                    fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of stale writes the agent started")
    ax.set_title(f"Fate of stale writes\n{sub}")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    save(fig, f"{mode}_stale_writes.png")

    # 4. wall clock per scenario, by use case
    kinds = sorted(next(iter(data["by_kind"].values())))
    fig, ax = plt.subplots(figsize=(8, 4))
    w2 = 0.8 / len(agents)
    for i, a in enumerate(agents):
        vals = [data["by_kind"][a][k]["wall_s"]["mean"] or 0 for k in kinds]
        bars = ax.bar([x + i * w2 for x in range(len(kinds))], vals, w2, label=LABEL[a],
                      color=COLOR[a])
        _bar_labels(ax, bars, "{:.2f}")
    ax.set_xticks([x + w2 * (len(agents) - 1) / 2 for x in range(len(kinds))], kinds)
    ax.set_ylabel("seconds")
    ax.set_title(f"Mean wall-clock per scenario (includes the scripted speech and pauses)\n{sub}")
    ax.legend()
    save(fig, f"{mode}_wall_clock.png")
    return files


# ------------------------------------------------------------------------------ markdown ------
def _headline(data: dict[str, Any]) -> str:
    agents = list(data["agents"])
    A = data["agents"]
    rows = [
        ["**Time to first response** p50"] + [ms(A[a]["ttfr_ms"]["p50"]) for a in agents],
        ["  p95"] + [ms(A[a]["ttfr_ms"]["p95"]) for a in agents],
        ["  p99"] + [ms(A[a]["ttfr_ms"]["p99"]) for a in agents],
        ["  samples / unanswered utterances"] + [
            f"{A[a]['ttfr_ms']['n']} / {A[a]['ttfr_ms']['no_response']}" for a in agents],
        ["**Wall-clock per scenario** mean"] + [secs(A[a]["wall_s"]["mean"]) for a in agents],
        ["  p95"] + [secs(A[a]["wall_s"]["p95"]) for a in agents],
        ["**Stale writes started** (all scenarios)"] + [
            f"{A[a]['stale_writes']['started']} in {A[a]['stale_writes']['runs_with_stale_writes']} "
            "scenarios" for a in agents],
        ["  stopped before dispatch"] + [stale_pct(A[a]["stale_writes"], "prevented") for a in agents],
        ["  committed, then undone"] + [stale_pct(A[a]["stale_writes"], "compensated") for a in agents],
        ["  committed, replaced by a later write"] + [
            stale_pct(A[a]["stale_writes"], "superseded") for a in agents],
        ["  **committed and left standing**"] + [stale_pct(A[a]["stale_writes"], "standing")
                                                 for a in agents],
        ["**Duplicate live writes** (scenarios)"] + [rate(A[a]["duplicate_live_writes"]) for a in agents],
        ["**Double charge** (scenarios)"] + [rate(A[a]["double_charge"]) for a in agents],
        ["**Final state consistent with user's final intent**"] + [rate(A[a]["consistent"])
                                                                   for a in agents],
        ["World reads / committed writes per scenario"] + [
            f"{A[a]['world_calls_per_scenario']['reads']} / "
            f"{A[a]['world_calls_per_scenario']['writes_committed']}" for a in agents],
        ["Errors or timeouts"] + [str(A[a]["errors"]) for a in agents],
        ["Stale-epoch outputs emitted (CHRONOS invariant)"] + [
            str(A[a]["stale_outputs_emitted"]) if a == "chronos" else "n/a" for a in agents],
    ]
    return table(["Metric"] + [LABEL[a] for a in agents], rows)


def _breakdown(data: dict[str, Any], key: str, title: str) -> str:
    agents = list(data["agents"])
    groups = sorted(next(iter(data[key].values())))
    rows = []
    for g in groups:
        n = next(iter(data[key].values()))[g]["scenarios"]
        cells = [f"{data[key][a][g]['consistent']['pct']:.0f}% "
                 f"({data[key][a][g]['consistent']['k']}/{data[key][a][g]['consistent']['n']})"
                 for a in agents]
        rows.append([g, str(n)] + cells)
    return f"**{title}**: final state consistent\n\n" + table(
        ["", "scenarios"] + [LABEL[a] for a in agents], rows)


def _failures(data: dict[str, Any], agent: str, limit: int = 12) -> str:
    bad = [r for r in data["runs"].get(agent, []) if not r["consistent"]]
    if not bad:
        return f"{LABEL[agent]} had no inconsistent scenarios in this run.\n"
    lines = [f"{LABEL[agent]}: {len(bad)} inconsistent scenario(s); first {min(limit, len(bad))}:\n"]
    for r in bad[:limit]:
        why = "; ".join(r["problems"]) or r.get("error") or "unknown"
        lines.append(f"- scenario {r['scenario']} ({r['kind']}/{r['variant']}, phase "
                     f"{r['phase']}): {why}")
    return "\n".join(lines) + "\n"


def _mode_section(mode: str, data: dict[str, Any], charts: list[str]) -> str:
    title = {"mock": "Mock LLM", "ollama": "Local LLM (Ollama: llama3.2:3b + moondream)"}[mode]
    if data.get("status") == "skipped":
        return (f"## {title}\n\n**Not measured.** {data['reason']}. No numbers are reported for "
                "this configuration; none were estimated or carried over from the mock run.\n")
    mix = data["mix"]
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(mix["kind"].items()))
    phases = ", ".join(f"{k} {v}" for k, v in sorted(mix["phase"].items()))
    intro = (f"{data['scenarios']} scenarios per agent (use cases: {kinds}; interrupt phase: "
             f"{phases}). Elapsed {data.get('elapsed_s', '?')} s. Brackets are 95% Wilson "
             "intervals.\n")
    parts = [f"## {title}\n", intro, _headline(data)]
    for c in charts:
        parts.append(f"![{c}](charts/{c})\n")
    parts.append(_breakdown(data, "by_kind", "By use case"))
    parts.append(_breakdown(data, "by_phase", "By interrupt phase (nominal, see notes)"))
    parts.append(_breakdown(data, "by_variant", "By scenario variant"))
    parts.append("**Scenarios CHRONOS got wrong**\n\n" + _failures(data, "chronos"))
    for a in data["agents"]:
        if a != "chronos":
            parts.append(_failures(data, a, limit=4))
    return "\n".join(parts)


def _hardware(meta: dict[str, Any]) -> str:
    h = meta["hardware"]
    rows = [["OS", h["os"]], ["CPU", f"{h['cpu']} ({h['logical_cores']} logical cores)"],
            ["RAM", f"{h['ram_gb']} GB" if h["ram_gb"] else "unknown"],
            ["GPU", h["gpu"] + " (only relevant to the Ollama pass)"],
            ["Python / pydantic / fastapi", f"{h['python']} / {h['pydantic']} / {h['fastapi']}"],
            ["Scenario concurrency", str(meta["concurrency"])],
            ["Mean tool latency per scenario",
             (f"drawn from {meta['tool_latency_mean_range_ms'][0]}-"
              f"{meta['tool_latency_mean_range_ms'][1]} ms (each call 0.5-1.5x that)")],
            ["End-of-utterance silence", f"{meta['eou_silence_ms']} ms"],
            ["Seed", str(meta["seed"])], ["Generated (UTC)", meta["generated_utc"]]]
    probe = meta.get("ollama_probe")
    if probe:
        rows.append(["Ollama", "reachable" if probe["reachable"] else f"not reachable at {probe['url']}"])
    return table(["", ""], rows)


NOTES = """\
## What was measured, and what these numbers do and do not mean

**Definitions.** *Time to first response* is measured per actionable utterance: milliseconds from
the moment it is sent until the agent emits any message. CHRONOS emits an acknowledgment from its
deterministic fast path; the half-duplex baseline speaks only after it has finished acting, so this
metric shows the effect of the acknowledgment, not faster task completion. Task completion is the
*wall-clock* row (which includes the scripted speech and pauses, identical for every agent).
A *stale write* is a write attempt whose arguments do not match the user's final intent; what
happened to each one is read from the world's own call log (a recording wrapper around the mock
world), not from the agent's bookkeeping. *Duplicate writes*, *double charge* and *consistency* are
read from the final world state against a per-scenario oracle, and consistency also requires the
world's invariants to hold and, for troubleshooting, the last diagnosis to match the last frame.

**Agents.** All agents share the same world, tools and argument schemas, intent and slot rules,
LLM and vision clients, diagnosis code, and (rule-based) utterance labelling. The baseline is
sequential and half-duplex: it acts only on complete utterances, speaks only after acting, and on an
interrupt cancels what is in flight and restarts the goal. It has no speculation, epochs, fencing
token, idempotency ledger, read cache, or compensation of stale writes. To avoid a strawman it
gets two concessions: it ignores "okay"/"um" using CHRONOS's own rules, and on "cancel that" it
undoes the writes it remembers from the current attempt. *Baseline (strict)* drops the first
concession: every input is treated as an interrupt. The world is transactional, so cancelling a
baseline mid-write rolls back rather than corrupting; the baseline's failures come from timing,
not from a broken database.

**Limits you should state alongside these numbers.**
- Everything is simulated: tool latency is random sleep (not real services), speech is scripted
  text (no ASR), and scenarios come from a generator written by the same authors as the agent, so
  the oracle and the phrasing reflect what the rules were built to understand. Real user speech
  would be harder for both agents.
- The *mock* LLM is deterministic and rule-based. With it, results show the effect of the
  coordination layer, not language-model quality; in these scenarios the intent rules handle
  every utterance, so the LLM is exercised only for the troubleshooting diagnosis.
- Interrupt phases (before/during/after) are *nominal*: they are scheduled from the mean tool
  latency, and each call's latency varies, so a "during" scenario can land slightly earlier or
  later. Use the phase breakdown to see trends, not exact boundaries.
- Percentages come from a few hundred scenarios; the brackets are 95% Wilson intervals. Small
  tail counts (p99, rare failures) are noisy.
- Timing was taken on one machine with scenarios run one at a time (concurrency shown above), so
  absolute milliseconds will differ elsewhere; the comparison between agents on the same run is
  the meaningful part. Idle detection polls at ~20 ms, so wall-clock is taken from the last
  observed activity instead.
"""


def write_reports(results: dict[str, Any], out_dir: Path, charts: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_names: dict[str, list[str]] = {}
    if charts:
        for mode, data in results["modes"].items():
            if data.get("status") == "measured":
                chart_names[mode] = make_charts(mode, data, out_dir / "charts")
    meta = results["meta"]
    md = ["# CHRONOS benchmark results\n",
          ("Generated by `python -m bench.run`; every number below is computed from that run "
           f"(`{meta['command']}`). Raw per-scenario data is in `results.json`.\n"),
          "## Environment\n", _hardware(meta)]
    for mode in ("mock", "ollama"):
        if mode in results["modes"]:
            md.append(_mode_section(mode, results["modes"][mode], chart_names.get(mode, [])))
    md.append(NOTES)
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
