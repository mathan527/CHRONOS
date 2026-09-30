/* Fills the Home stat tiles and the Results page from /api/results (bench/results.json). */
(async () => {
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  let data;
  try {
    const r = await fetch("/api/results", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    data = await r.json();
  } catch {
    const m = $("#meta"); if (m) m.textContent = "No benchmark results found. Run: python -m bench.run";
    return;
  }
  const mock = data.modes && data.modes.mock;
  const C = mock && mock.agents && mock.agents.chronos, B = mock && mock.agents && mock.agents.baseline;
  if (!C || !B) return;

  const ms = v => (v < 10 ? v.toFixed(1) : Math.round(v)) + " ms";
  const pct = o => o.pct.toFixed(o.pct % 1 ? 1 : 0) + "%";
  const frac = o => `${o.k}/${o.n}`;
  const stat = {
    ttfr_chronos: ms(C.ttfr_ms.p50), ttfr_baseline: ms(B.ttfr_ms.p50),
    cons_chronos: pct(C.consistent), cons_baseline: pct(B.consistent),
    dup_chronos: String(C.duplicate_live_writes.k), dup_baseline: String(B.duplicate_live_writes.k),
    dc_chronos: String(C.double_charge.k), dc_baseline: String(B.double_charge.k),
  };
  document.querySelectorAll("[data-stat]").forEach(n => { const v = stat[n.dataset.stat]; if (v != null) n.textContent = v; });

  const meta = $("#meta");
  if (meta) {
    const m = data.meta || {}, hw = m.hardware || {};
    meta.textContent = `${mock.scenarios} scenarios per agent · seed ${m.seed} · ${hw.cpu || "unknown CPU"} · LLM: ${mock.llm_mode} · generated ${(m.generated_utc || "").slice(0, 10)}`;
  }
  const oll = $("#ollama");
  if (oll) {
    const o = data.modes.ollama || {};
    oll.textContent = o.status === "measured" ? "Measured with local Ollama models; see bench/results.md."
      : `Skipped in this run: ${o.reason || "Ollama not available"}. Real-model timings are not claimed here.`;
  }

  const bars = $("#bars");
  if (bars) {
    const rows = [
      ["Final state matches intent", C.consistent.pct, B.consistent.pct, true],
      ["Runs with duplicate live writes", C.duplicate_live_writes.pct, B.duplicate_live_writes.pct, false],
      ["Runs with a double charge", C.double_charge.pct, B.double_charge.pct, false],
      ["Stale writes left standing", C.stale_writes.standing_pct, B.stale_writes.standing_pct, false],
    ];
    for (const [label, c, b] of rows) {
      const max = Math.max(100, c, b);
      const wrap = el("div"); wrap.append(el("div", null, label));
      wrap.lastChild.style.cssText = "font-size:.85rem;color:var(--muted);margin:.4rem 0 .2rem";
      for (const [who, v, cls] of [["CHRONOS", c, "c"], ["Baseline", b, "b"]]) {
        const row = el("div", "brow"), track = el("div", "track"), fill = el("div", "fill " + cls);
        fill.style.width = (v / max * 100) + "%"; track.appendChild(fill);
        row.append(el("span", null, who), track, el("span", "mono", v.toFixed(1) + "%"));
        wrap.appendChild(row);
      }
      bars.appendChild(wrap);
    }
  }

  const tb = $("#cmp tbody");
  if (tb) {
    const ci = o => `[${Math.round(o.ci95[0])}–${Math.round(o.ci95[1])}]`;
    const sw = (a, k) => `${a.stale_writes[k + "_pct"].toFixed(1)}% (${a.stale_writes[k]}/${a.stale_writes.started})`;
    const rows = [
      ["Time to first response, median", ms(C.ttfr_ms.p50), ms(B.ttfr_ms.p50), C.ttfr_ms.p50 < B.ttfr_ms.p50],
      ["Time to first response, p95", ms(C.ttfr_ms.p95), ms(B.ttfr_ms.p95), C.ttfr_ms.p95 < B.ttfr_ms.p95],
      ["Wall-clock per scenario, mean", C.wall_s.mean.toFixed(2) + " s", B.wall_s.mean.toFixed(2) + " s", null],
      ["Final state matches intent", `${pct(C.consistent)} (${frac(C.consistent)}) ${ci(C.consistent)}`, `${pct(B.consistent)} (${frac(B.consistent)}) ${ci(B.consistent)}`, C.consistent.pct >= B.consistent.pct],
      ["Duplicate live writes (scenarios)", `${frac(C.duplicate_live_writes)} ${ci(C.duplicate_live_writes)}`, `${frac(B.duplicate_live_writes)} ${ci(B.duplicate_live_writes)}`, C.duplicate_live_writes.k <= B.duplicate_live_writes.k],
      ["Double charge (scenarios)", `${frac(C.double_charge)} ${ci(C.double_charge)}`, `${frac(B.double_charge)} ${ci(B.double_charge)}`, C.double_charge.k <= B.double_charge.k],
      ["Stale writes: stopped before dispatch", sw(C, "prevented"), sw(B, "prevented"), null],
      ["Stale writes: committed, then undone", sw(C, "compensated"), sw(B, "compensated"), null],
      ["Stale writes: left standing", sw(C, "standing"), sw(B, "standing"), C.stale_writes.standing <= B.stale_writes.standing],
    ];
    for (const [name, c, b, win] of rows) {
      const tr = el("tr"); tr.append(el("td", null, name), el("td", win ? "good" : null, c), el("td", win === false ? "badv" : null, b));
      tb.appendChild(tr);
    }
  }
})();
