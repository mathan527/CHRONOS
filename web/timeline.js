/* Session picker for the Timeline page: lists /api/traces and embeds /traces/<id>/view. */
(async () => {
  const $ = s => document.querySelector(s);
  const pick = $("#pick"), frame = $("#frame"), empty = $("#empty"), open = $("#open"), raw = $("#raw");
  let rows = [];
  try {
    const r = await fetch("/api/traces", { cache: "no-store" });
    rows = r.ok ? await r.json() : [];
  } catch { rows = []; }
  pick.replaceChildren();
  if (!rows.length) { pick.disabled = true; pick.appendChild(new Option("no sessions", "")); empty.hidden = false; return; }
  for (const t of rows) {
    const when = new Date(t.modified * 1000).toLocaleString();
    pick.appendChild(new Option(`${t.session_id} · ${t.events} events · ${when}`, t.session_id));
  }
  pick.disabled = false;
  const show = id => {
    const url = `/traces/${encodeURIComponent(id)}/view`;
    frame.src = url; frame.hidden = false; empty.hidden = true;
    open.href = url; open.hidden = false;
    raw.href = `/traces/${encodeURIComponent(id)}`; raw.hidden = false;
    history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  };
  pick.onchange = () => show(pick.value);
  const want = decodeURIComponent(location.hash.slice(1));
  if (want && rows.some(t => t.session_id === want)) pick.value = want;
  show(pick.value);
})();
