/* CHRONOS demo client. Vanilla JS, no build step, no network beyond this server.
 *
 * Data sources (nothing in the protocol changed):
 *   - WebSocket /ws/{session}?trace=1  -> OutputMessage frames (as before) plus opt-in
 *                                         {"kind":"trace","row":{...}} rows straight from the trace logger
 *   - GET /sessions/{id}/state         -> plan, ledger, slots (refreshed after activity)
 *   - GET /health                      -> LLM mode and Ollama/model availability
 */
(() => {
"use strict";

// ================================================================================ helpers ===
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const NS = "http://www.w3.org/2000/svg";
const epIdx = e => ((((e | 0) - 1) % 4) + 4) % 4;      // epoch 1 blue, 2 violet, 3 amber, 4 teal, then cycle
const epVar = e => `var(--e${epIdx(e) + 1})`;
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function icon(id, cls = "ic") {
  const s = document.createElementNS(NS, "svg");
  s.setAttribute("class", cls); s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS(NS, "use");
  u.setAttribute("href", "#i-" + id);
  s.appendChild(u);
  return s;
}
const fmtMs = x => (x < 10 ? x.toFixed(1) : String(Math.round(x))) + " ms";
const median = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const label = s => String(s || "").replace(/_/g, " ").toUpperCase();
const norm = s => String(s || "").toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();
const fmtDate = iso => {
  const d = new Date(iso + "T00:00:00Z");
  return isNaN(d) ? String(iso) : d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
};
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), 2600);
}
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// ==================================================================================== state ===
const S = {
  id: "", ws: null, closing: false, attempt: 0, retry: null,
  t0: performance.now(), clock: null,                 // clock: last trace row time (server ms) + local time
  epoch: 0, epochs: new Map(), groups: new Map(), cards: [], epochDone: new Set(),
  speech: 0, speaking: false, run: 0, chain: 0, partialBubble: null, sig: {}, reuseShown: new Set(), waiters: [], lastActivity: performance.now(),
  m: { acks: [], stale: 0, dup: 0, commit: 0, comp: 0, lastAck: null, staleReached: 0 },
  tools: new Map(), tasks: new Map(), committed: new Set(), guards: [],
  state: null, prevSlots: null, oldPlan: null,
  pendingUser: [], pendingAcks: [], ackRows: [], lastActed: null, lastInterrupt: null,
  lastFrame: null, health: null, stateTimer: 0, stateBusy: false, events: 0,
};

// ================================================================================ connection ===
function setPill(id, cls, text, title) {
  const p = $("#" + id);
  p.className = "pill " + cls; $(".t", p).textContent = text;
  if (title !== undefined) p.title = title;
}
function connect() {
  clearTimeout(S.retry);
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${encodeURIComponent(S.id)}?trace=1`);
  S.ws = ws; S.closing = false;
  ws.onopen = () => {
    S.attempt = 0; setPill("pill-ws", "ok", "connected"); $("#banner").hidden = !S.health || !S.health.bannerShown;
    scheduleState();
  };
  ws.onmessage = ev => {
    let m; try { m = JSON.parse(ev.data); } catch { toast("Bad message from server"); return; }
    if (m.kind === "trace") onTrace(m.row); else onOutput(m);
  };
  ws.onclose = () => {
    if (S.ws !== ws || S.closing) return;
    const delay = Math.min(8000, 500 * 2 ** S.attempt++);
    setPill("pill-ws", "warn", `reconnecting… ${Math.round(delay / 100) / 10}s`, "The socket dropped; retrying with backoff");
    S.retry = setTimeout(connect, delay);
  };
  ws.onerror = () => { /* onclose follows and handles the retry */ };
}
function send(type, payload) {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) { toast("Not connected, reconnecting…"); return false; }
  S.ws.send(JSON.stringify({ type, payload }));
  S.events++; S.lastActivity = performance.now();
  return true;
}

// ======================================================================= session lifecycle ===
function resetUI() {
  $("#feed").replaceChildren($("#empty"));
  $("#empty").hidden = false;
  S.epoch = 0; S.epochs.clear(); S.groups.clear(); S.cards = []; S.epochDone.clear();
  S.m = { acks: [], stale: 0, dup: 0, commit: 0, comp: 0, lastAck: null, staleReached: 0 };
  S.tools.clear(); S.tasks.clear(); S.committed.clear(); S.guards = []; S.reuseShown.clear();
  S.state = null; S.prevSlots = null; S.oldPlan = null; S.pendingUser = []; S.pendingAcks = []; S.ackRows = []; S.partialBubble = null;
  S.lastActed = null; S.lastInterrupt = null; S.lastFrame = null; S.clock = null; S.events = 0;
  S.speech++; setLive(false); S.sig = {};
  renderMetrics(true); renderRail(); renderPlan(); renderInflight(); renderLedger(); renderSlots();
  tlReset(); setCaption(null); jumpCheck();
}
function startSession(id) {
  S.closing = true; if (S.ws) S.ws.close();
  clearTimeout(S.retry); S.attempt = 0;
  S.id = id; S.t0 = performance.now(); S.run++;
  $("#sid").textContent = id;
  const q = new URLSearchParams(location.search); q.set("session", id); q.delete("run");
  history.replaceState(null, "", "?" + q.toString());
  $("#trace-link").href = $("#tl-link").href = `/traces/${encodeURIComponent(id)}/view?live=1`;
  $("#state-link").href = `/sessions/${encodeURIComponent(id)}/state`;
  resetUI(); markScenario(null);
  setPill("pill-ws", "warn", "connecting…");
  connect();
}
const newId = () => "demo-" + Math.random().toString(36).slice(2, 7);

// ==================================================================================== time ===
function nowRel() {                                   // seconds since the session started (server clock)
  if (S.clock) return (S.clock.ms + (performance.now() - S.clock.at)) / 1000;
  return (performance.now() - S.t0) / 1000;
}
function whenEl() {
  const w = el("span", "when", "+" + nowRel().toFixed(2) + " s");
  const d = new Date();
  w.title = d.toLocaleTimeString([], { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
  return w;
}

// ============================================================================ the feed ===
const feed = () => $("#feed");
function group(epoch) {
  let g = S.groups.get(epoch);
  if (g) return g;
  $("#empty").hidden = true;
  const root = el("div", "egroup current");
  root.style.setProperty("--ec", epVar(epoch));
  const head = el("div", "ghead");
  const info = S.epochs.get(epoch);
  head.append(el("span", "", "Epoch " + epoch),
              el("span", "why", epoch === 1 ? "initial request" : ""),
              el("span", "state", "active"));
  root.appendChild(head);
  g = { el: root, head, state: $(".state", head), why: $(".why", head) };
  S.groups.set(epoch, g);
  feed().appendChild(root);
  return g;
}
function pinned() { const f = feed(); return f.scrollHeight - f.scrollTop - f.clientHeight < 90; }
function toBottom() { const f = feed(); f.scrollTop = f.scrollHeight; jumpCheck(); }
function jumpCheck() { $("#jump").hidden = pinned(); }
function append(parent, node) {
  const stick = pinned();
  parent.appendChild(node);
  if (stick) requestAnimationFrame(toBottom); else jumpCheck();
}
function curGroup() { return group(Math.max(1, S.epoch)); }

function jsonNode(v, depth = 0) {
  const span = (c, t) => el("span", c, t);
  if (v === null) return span("j-z", "null");
  if (typeof v === "string") return span("j-s", JSON.stringify(v));
  if (typeof v === "number") return span("j-n", String(v));
  if (typeof v === "boolean") return span("j-b", String(v));
  const frag = document.createDocumentFragment();
  const isArr = Array.isArray(v), keys = isArr ? v.map((_, i) => i) : Object.keys(v);
  if (!keys.length) return span("j-z", isArr ? "[]" : "{}");
  const pad = "  ".repeat(depth + 1);
  frag.append(isArr ? "[\n" : "{\n");
  keys.forEach((k, i) => {
    frag.append(pad);
    if (!isArr) { frag.append(span("j-k", JSON.stringify(k)), ": "); }
    frag.append(jsonNode(v[k], depth + 1));
    frag.append(i < keys.length - 1 ? ",\n" : "\n");
  });
  frag.append("  ".repeat(depth) + (isArr ? "]" : "}"));
  return frag;
}
function dataViewer(data) {
  if (!data || !Object.keys(data).length) return null;
  const d = el("details", "data");
  const s = el("summary"); s.append(icon("chevron"), "data");
  const pre = el("pre", "json"); pre.appendChild(jsonNode(data));
  d.append(s, pre);
  return d;
}

function badge(status) {
  const map = { pending: "circle", committed: "dot", done: "check", cancelled: "x", failed: "alert" };
  const b = el("span", "badge " + status);
  b.append(icon(map[status] || "circle"), status);
  return b;
}
function chip(cls, text, iconId, title) {
  const c = el("span", "chip " + cls);
  if (iconId) c.appendChild(icon(iconId));
  c.append(text);
  if (title) c.title = title;
  return c;
}

function bubble(kind, epoch, whoText) {
  const wrap = el("div", "msg " + kind);
  wrap.style.setProperty("--ec", epVar(epoch));
  wrap.dataset.epoch = epoch;
  const av = el("div", "avatar", kind === "user" ? "YOU" : "C");
  const b = el("div", "bubble");
  const meta = el("div", "meta");
  meta.appendChild(el("span", "who", whoText));
  b.appendChild(meta);
  wrap.append(av, b);
  return { wrap, b, meta };
}

function addUser(text, { thumb = null, note = "" } = {}) {
  const g = curGroup();
  const { wrap, b, meta } = bubble("user", Math.max(1, S.epoch), "You");
  const slot = el("span", "slot-chip");                      // the classification chip lands here
  slot.style.display = "contents";
  meta.append(slot, whenEl());
  if (thumb) { const im = el("img"); im.src = thumb.url; im.alt = thumb.name; im.style.cssText = "max-height:84px;border-radius:8px;display:block;margin:.2rem 0 .3rem"; b.appendChild(im); }
  if (text) b.appendChild(el("div", "body", text));
  if (note) b.appendChild(el("div", "hint", note));
  append(g.el, wrap);
  const rec = { wrap, slot, text, chipped: false };
  S.pendingUser.push(rec);
  if (!thumb && !note.startsWith("partial")) S.partialBubble = null;
  return rec;
}
function setUserChip(rec, u) {
  rec.slot.replaceChildren();
  const lab = label(u.label);
  if (u.acted) rec.slot.appendChild(chip("acted", lab, null, "Classified as " + lab));
  else rec.slot.append(chip("muted", lab), chip("muted", "ignored, not acted on", null, "Backchannels and hesitations never interrupt"));
  const tier = u.tier === "llm" ? "LLM" : u.tier;
  rec.slot.appendChild(chip("tier", `${tier} · ${fmtMs(u.classify_ms)}`, null, `Classifier tier: ${u.tier}; took ${u.classify_ms} ms`));
  rec.chipped = true;
  if (!u.acted) rec.wrap.classList.add("ignored");
}
function attachUtterance(u) {
  if (u.partial) {                                          // "I want to boo… um…": heard, held, never acted on
    let rec = S.partialBubble;
    if (!rec) {
      rec = addUser(u.text, { note: "partial transcript: heard, but not acted on" });
      rec.wrap.querySelector(".meta .who").textContent = "You (speaking)";
      S.partialBubble = rec;
    } else $(".body", rec.wrap).textContent = u.text;
    setUserChip(rec, u); rec.chipped = true;
    return;
  }
  S.partialBubble = null;
  const open = S.pendingUser.filter(r => !r.chipped);
  let rec = open.reverse().find(r => norm(r.text) === norm(u.text)) || open[0];
  if (!rec) rec = addUser(u.text);
  setUserChip(rec, u);
  if (u.acted) S.lastActed = { text: u.text, label: u.label, at: performance.now() };
}

// ------------------------------------------------------------------------ agent messages ---
const TOOL_ICON = { book_flight: "plane", cancel_booking: "undo", set_navigation: "route", reserve_table: "table" };
const RESULT = {
  book_flight: d => ({ ico: "plane", title: `Flight booked: ${d.origin} → ${d.dest}`, ref: d.booking_id, ref2: d.charge_id && "charge " + d.charge_id,
    kv: [["Date", fmtDate(d.date)], ["Time", d.time], ["Fare", "₹" + d.amount], ["Passenger", d.passenger]] }),
  set_navigation: d => { const r = d.route || {}; return { ico: "route", title: `Route set: ${r.destination || ""}`, ref: d.nav_id,
    kv: [["From", r.origin], ["Via", r.via || "direct"], ["ETA", r.eta_min + " min"], ["Distance", r.distance_km + " km"]] }; },
  reserve_table: d => ({ ico: "table", title: `Table reserved: ${d.restaurant}`, ref: d.reservation_id,
    kv: [["Party", d.party_size], ["Date", fmtDate(d.date)], ["Time", d.time], ["Table", "#" + d.table_id]] }),
  cancel_booking: d => ({ ico: "undo", title: `Compensated: booking ${d.booking_id} cancelled`, ref: d.booking_id,
    kv: [["Refunded", d.refunded ? "yes ✓" : "no"], ["Why", "the plan changed after it committed"]] }),
  clear_navigation: d => ({ ico: "undo", title: `Compensated: route ${d.nav_id} cleared`, ref: d.nav_id,
    kv: [["Cleared", d.cleared ? "yes ✓" : "no"], ["Why", "the plan changed after it committed"]] }),
};
function resultBody(m) {
  const d = m.data || {}, f = RESULT[d.tool];
  const box = el("div", "result");
  const info = f ? f(d) : { ico: "wrench", title: m.text, kv: Object.entries(d).filter(([k, v]) => typeof v !== "object" && k !== "tool").slice(0, 4) };
  const ico = el("div", "ico"); ico.appendChild(icon(info.ico));
  const h = el("div"); h.append(el("h4", "", info.title));
  const sum = el("div", "sum", m.text);
  const right = el("div"); right.append(h, sum);
  box.append(ico, right);
  const kv = el("div", "kv");
  info.kv.filter(([, v]) => v !== undefined && v !== null && v !== "").forEach(([k, v]) => { const s = el("span", "", k); s.appendChild(el("b", "", String(v))); kv.appendChild(s); });
  if (info.ref) { const s = el("span"); s.append(el("span", "ref", info.ref)); kv.appendChild(s); }
  if (info.ref2) kv.appendChild(el("span", "", info.ref2));
  box.appendChild(kv);
  return box;
}
function visionBody(m) {
  const d = m.data.diagnosis, box = el("div", "vision");
  if (S.lastFrame) {
    const shot = el("div", "shot"), im = el("img"); im.src = S.lastFrame.url; im.alt = "The camera frame";
    shot.append(im, el("div", "cap", S.lastFrame.name));
    box.appendChild(shot);
  } else box.style.gridTemplateColumns = "1fr";
  const txt = el("div");
  const sec = (ic, lab, node) => { const s = el("div", "vsec"), l = el("div", "lab"); l.append(icon(ic), lab); s.append(l, node); return s; };
  txt.appendChild(sec("eye", "What I see · Moondream", el("p", "", d.observed || "No camera image was available.")));
  const like = el("div"); like.appendChild(el("p", "", d.likely_cause || d.diagnosis));
  const chips = el("div", "meta"); chips.style.marginTop = ".3rem";
  chips.append(d.grounded ? chip("ok", "grounded in the frame", "check") : chip("warn", "not grounded in a frame", "alert"),
               chip("", "source: " + d.source), ...((d.kb_ids || []).slice(0, 2).map(k => chip("info", "kb: " + k))));
  like.appendChild(chips);
  txt.appendChild(sec("wrench", "Likely issue", like));
  if ((d.steps || []).length) { const ol = el("ol"); d.steps.forEach(s => ol.appendChild(el("li", "", s))); txt.appendChild(sec("chevron", "Next step", ol)); }
  box.appendChild(txt);
  return box;
}

function onOutput(m) {
  S.lastActivity = performance.now();
  if (typeof m.epoch === "number") {
    if (m.epoch < S.epoch && m.status !== "cancelled") { S.m.staleReached++; renderMetrics(); }   // must stay 0
    if (m.epoch > S.epoch) bumpEpoch(m.epoch, "", "");                                          // no trace: still show it
  }
  const ep = Math.max(1, m.epoch || 1), g = group(ep);
  const { wrap, b, meta } = bubble("agent", ep, m.type === "action_result" ? "Result" : m.type === "ack" ? "Ack" : m.type === "progress" ? "Progress" : m.type === "error" ? "Error" : "CHRONOS");
  const badgeSlot = el("span", "slot-chip"); badgeSlot.style.display = "contents";
  badgeSlot.appendChild(badge(m.status));
  meta.append(badgeSlot);
  const chipSlot = el("span", "slot-chip"); chipSlot.style.display = "contents";
  meta.append(chipSlot, whenEl());
  if (m.type === "action_result") b.appendChild(resultBody(m));
  else if (m.data && m.data.diagnosis) b.appendChild(visionBody(m));
  else b.appendChild(el("div", "body", m.text));
  if (m.type === "error") meta.prepend(icon("alert"));
  const dv = dataViewer(m.data); if (dv) b.appendChild(dv);
  append(g.el, wrap);

  const card = { wrap, epoch: ep, status: m.status, type: m.type, text: m.text, data: m.data || {}, badgeSlot, chipSlot };
  S.cards.push(card);
  if (m.type === "ack") {                                   // pair with its ack_sent trace row (either arrival order)
    const i = S.ackRows.findIndex(r => r.text === m.text);
    if (i >= 0) { setAckChip(card, S.ackRows.splice(i, 1)[0].latency_ms); } else S.pendingAcks.push(card);
  }
  if ((m.type === "action_result" || m.type === "response") && m.status === "done") S.epochDone.add(ep);
  if (m.type === "action_result" && m.status === "cancelled") undoCard(m.data);
  S.waiters = S.waiters.filter(w => { if (w.type === m.type) { w.resolve(m); return false; } return true; });
  scheduleState();
}
function setAckChip(card, ms) {
  card.chipSlot.replaceChildren(chip(ms < 300 ? "ok" : "bad", `ack ${fmtMs(ms)}`, "clock", "Deterministic fast-path acknowledgment (no LLM)"));
}
function goneCard(card, why) {
  card.wrap.classList.add("gone");
  card.status = "cancelled";
  card.badgeSlot.replaceChildren(badge("cancelled"));
  if (why) card.wrap.title = why;
}
function undoCard(d) {                                        // strike the earlier card this compensation undid
  const id = d.booking_id || d.nav_id, k = d.booking_id ? "booking_id" : "nav_id";
  if (!id) return;
  S.cards.filter(c => c.type === "action_result" && c.status === "done" && c.data[k] === id)
    .forEach(c => goneCard(c, "Undone by a compensating action"));
}
function awaitMessage(type, ms) {
  return new Promise(resolve => {
    const w = { type, resolve };
    S.waiters.push(w);
    setTimeout(() => { S.waiters = S.waiters.filter(x => x !== w); resolve(null); }, ms);
  });
}

function evtLine(tone, iconId, html) {
  const g = curGroup(), d = el("div", "evt " + tone);
  d.append(icon(iconId), ...html);
  append(g.el, d);
}
const bold = t => el("b", "", t);

// ------------------------------------------------------------------------- epoch handling ---
function bumpEpoch(to, reason, text) {
  if (to <= S.epoch) return;
  const from = S.epoch;
  S.epoch = to;
  const why = reason ? `${label(reason)}${text ? ": “" + text + "”" : ""}` : "";
  const info = { n: to, reason: why, cancelled: 0, reused: 0, reusedTools: new Set() };
  S.epochs.set(to, info);
  if (from > 0) {
    const old = S.groups.get(from);
    if (old) { old.el.classList.replace("current", "superseded"); old.state.textContent = S.epochDone.has(from) ? "done, then superseded" : "superseded"; }
    S.cards.filter(c => c.epoch < to && (c.status === "pending" || c.status === "committed") && !S.epochDone.has(c.epoch))
      .forEach(c => goneCard(c, "Superseded: this work belongs to an older epoch"));
    snapshotOldPlan();
    const bar = el("div", "bumpline"); bar.style.setProperty("--ec", epVar(to));
    bar.setAttribute("role", "separator");
    const det = el("span", "det");
    bar.append(el("span", "sym", "⟳"), el("span", "ttl", `Epoch ${from} → ${to}`), det);
    info.bar = bar; info.det = det;
    feed().appendChild(bar);
    updateBump(info);
  }
  group(to);
  renderRail(); renderMetrics();
  requestAnimationFrame(toBottom);
}
function updateBump(info) {
  if (!info.det) return;
  const det = info.det; det.replaceChildren();
  const bits = [];
  if (info.reason) { const [lab, ...rest] = info.reason.split(": "); const q = el("span"); q.append(bold(lab)); if (rest.length) { q.append(": ", el("q", "", rest.join(": ").replace(/^“|”$/g, ""))); } bits.push(q); }
  bits.push(`${info.cancelled} task${info.cancelled === 1 ? "" : "s"} cancelled`);
  const reused = Math.max(info.reused, info.reusedTools.size);
  bits.push(`${reused} read result${reused === 1 ? "" : "s"} reused`);
  bits.forEach((b, i) => { if (i) det.append(" · "); det.append(b); });
}

// ============================================================================ trace rows ===
const TOOLISH = /^(read:|write:|kb\+vision|compensate)/;
function onTrace(r) {
  S.lastActivity = performance.now();
  S.clock = { ms: r.t_ms, at: performance.now() };
  tlAdd(r);
  switch (r.event) {
    case "utterance": attachUtterance(r); break;
    case "hesitation_held":
      if (S.partialBubble) S.partialBubble.slot.appendChild(chip("info", "waiting for you to continue", "clock"));
      break;
    case "frame_default_question": {
      const rec = addUser(r.text, { note: "no question was sent with the frame: CHRONOS asks the obvious one" });
      rec.wrap.querySelector(".meta .who").textContent = "You (default)";
      break;
    }
    case "ack_sent": {
      S.m.acks.push(r.latency_ms); S.m.lastAck = r.latency_ms; renderMetrics();
      const i = S.pendingAcks.findIndex(c => c.text === r.text);
      if (i >= 0) setAckChip(S.pendingAcks.splice(i, 1)[0], r.latency_ms); else S.ackRows.push(r);
      break;
    }
    case "epoch_bumped": {
      const recent = S.lastActed && performance.now() - S.lastActed.at < 4000;
      const text = r.reason === "cancel" && !recent ? "explicit interrupt" : recent ? S.lastActed.text : "";
      bumpEpoch(r.epoch, r.reason, text);
      break;
    }
    case "task_started": taskStart(r); break;
    case "task_finished": taskEnd(r, "done"); break;
    case "task_cancelled": {
      taskEnd(r, "cancelled");
      const info = S.epochs.get(r.superseded_by); if (info) { info.cancelled++; updateBump(info); }
      break;
    }
    case "tool_read": setTool(r.epoch, r.tool, "done"); if (r.cached) reuse(r.epoch, r.tool, ""); break;
    case "read_reused": reuse(r.epoch, r.tool, "snapshot"); break;
    case "read_carried_over": reuse(r.to_epoch || r.epoch, r.tool, r.from_epoch ? `from epoch ${r.from_epoch}` : ""); break;
    case "plan_patched": { const info = S.epochs.get(r.epoch); if (info) { info.reused += r.reused || 0; updateBump(info); } renderRail(); break; }
    case "plan_committed": S.committed.add(r.epoch); break;
    case "write_committed": S.m.commit++; setTool(r.epoch, r.tool, "done"); renderMetrics(); break;
    case "write_compensated":
      S.m.comp++; setTool(r.original_epoch, r.tool, "compensated"); renderMetrics();
      evtLine("warn", "undo", [bold("compensating action"), ` ${r.compensator} undid ${r.tool} from epoch ${r.original_epoch}`]);
      break;
    case "write_blocked_duplicate":
      S.m.dup++; S.guards.unshift({ kind: "dup", tool: r.tool, key: r.key, epoch: r.epoch }); renderMetrics(); renderLedger();
      setTool(r.epoch, r.tool, "blocked");
      evtLine("bad", "shield", [bold("duplicate write blocked"), ` ${r.tool} · key ${String(r.key).slice(0, 8)}`]);
      break;
    case "write_blocked":
      S.guards.unshift({ kind: "blocked", tool: r.tool, reason: r.reason, epoch: r.epoch }); renderLedger();
      setTool(r.epoch, r.tool, "blocked");
      evtLine("warn", "shield", [bold("write blocked"), ` ${r.tool}: ${String(r.reason).replace(/_/g, " ")}`]);
      break;
    case "stale_result_dropped":
      S.m.stale++; renderMetrics();
      evtLine("warn", "x", [bold("stale result dropped"), ` (${r.what}, epoch ${r.epoch} < ${r.current_epoch})`]);
      break;
    case "frame_described": evtLine("info", "eye", [bold("Moondream"), " described the frame"]); break;
    case "diagnosis_ready": evtLine(r.grounded ? "ok" : "warn", r.grounded ? "check" : "alert", [bold("diagnosis ready"), ` · ${r.grounded ? "grounded in the frame" : "not grounded"} · ${r.source}`]); break;
    case "ignored":
      if (r.reason === "nothing_to_cancel" && S.lastInterrupt) {
        S.lastInterrupt.slot.replaceChildren(chip("muted", "CANCEL"), chip("muted", "nothing to cancel, not acted on"));
        S.lastInterrupt = null;
      }
      break;
    default: break;
  }
  scheduleState();
  syncHot();
}

function reuse(epoch, tool, how) {                        // a speculative read result served again instead of re-run
  const info = S.epochs.get(epoch);
  if (info) { info.reusedTools.add(tool); updateBump(info); }
  const k = epoch + ":" + tool;
  if (S.reuseShown.has(k)) return;
  S.reuseShown.add(k);
  evtLine("ok", "refresh", [bold("read result reused"), ` ${tool}` + (how ? ` (${how})` : "")]);
}
// ------------------------------------------------------------------------ tool bookkeeping ---
function setTool(epoch, tool, state) {
  const k = epoch + ":" + tool, cur = S.tools.get(k);
  if (cur && cur.state === "done" && state === "running") return;
  S.tools.set(k, { state, at: performance.now() });
  renderPlan();
}
function taskStart(r) {
  const m = /^(read|write):(.+)$/.exec(r.task);
  if (m) setTool(r.epoch, m[2], "running");
  if (!TOOLISH.test(r.task)) return;
  S.tasks.set(r.task_id, { name: r.task.replace(/^(read|write):/, "$1 · "), epoch: r.epoch, t0: performance.now(), li: null, kind: r.task });
  renderInflight();
}
function taskEnd(r, how) {
  const m = /^(read|write):(.+)$/.exec(r.task || "");
  if (m && how === "cancelled") { const cur = S.tools.get(r.epoch + ":" + m[2]); if (!cur || cur.state !== "done") setTool(r.epoch, m[2], "cancelled"); }
  const t = S.tasks.get(r.task_id);
  if (!t) return;
  t.ended = how; t.li && t.li.classList.add(how === "cancelled" ? "gone" : "done");
  if (t.li) { const sp = $(".spin", t.li); if (sp) sp.replaceWith(icon(how === "cancelled" ? "x" : "check")); }
  setTimeout(() => { S.tasks.delete(r.task_id); renderInflight(); syncHot(); }, how === "cancelled" ? 1800 : 900);
}

// ============================================================================ agent brain ===
const changed = (key, value) => {                       // true (and remembered) only when `value` differs
  const v = JSON.stringify(value);
  if (S.sig[key] === v) return false;
  S.sig[key] = v; return true;
};
function scheduleState() {
  clearTimeout(S.stateTimer);
  S.stateTimer = setTimeout(refreshState, 90);
}
async function refreshState() {
  if (S.stateBusy) { scheduleState(); return; }
  const id = S.id; S.stateBusy = true;
  try {
    const r = await fetch(`/sessions/${encodeURIComponent(id)}/state`);
    if (r.ok && id === S.id) { S.state = await r.json(); renderPlan(); renderLedger(); renderSlots(); renderRail(); renderMetrics(); }
  } catch { /* offline: keep what we have */ } finally { S.stateBusy = false; }
}

function renderRail() {
  const plan0 = S.state && S.state.plan;
  if (!changed("rail", [S.epoch, [...S.epochs.values()].map(e => e.reason), plan0 && [plan0.epoch, plan0.status], [...S.epochDone]])) return;
  const ol = $("#epoch-rail"); ol.replaceChildren();
  if (!S.epoch) { ol.appendChild(el("li", "none", "No epochs yet")); return; }
  const plan = S.state && S.state.plan;
  for (let n = 1; n <= S.epoch; n++) {
    const info = S.epochs.get(n) || { reason: n === 1 ? "initial request" : "" };
    const li = el("li"); li.style.setProperty("--ec", epVar(n));
    const cur = n === S.epoch;
    li.className = cur ? "cur" : "old";
    let st = "superseded";
    if (cur) {
      const ps = plan && plan.epoch === n ? plan.status : null;
      st = { done: "done ✓", cancelled: "cancelled", committed: "committing", draft: "planning", failed: "failed" }[ps] || "active";
    } else if (S.epochDone.has(n)) st = "done, superseded";
    const first = n === 1 && !info.reason ? "initial request" : info.reason;
    li.append(el("span", "pt"), el("span", "nm", "Epoch " + n), el("span", "st", st), el("span", "rs", first || "initial request"));
    ol.appendChild(li);
    if (cur) { const g = S.groups.get(n); if (g) g.state.textContent = st; }
  }
}

const READ_TOOLS = new Set(["search_flights", "get_route", "check_table_availability", "get_booking", "lookup_troubleshooting_kb"]);
function stepLabel(s) {
  const vals = Object.entries(s.args || {}).filter(([, v]) => v !== null && v !== undefined && v !== "" && typeof v !== "object").map(([, v]) => v);
  const l = el("span", "lbl", s.tool);
  if (vals.length) l.appendChild(el("em", "", "  " + vals.slice(0, 3).join(" · ")));
  return l;
}
function stepState(step, plan) {
  const t = S.tools.get(plan.epoch + ":" + step.tool);
  if (t) return t.state;
  if (plan.status === "done") return "done";
  if (plan.status === "cancelled" || plan.status === "failed" || S.epoch > plan.epoch) return "cancelled";
  if (step.write) return (plan.status === "committed" || plan.status === "done" || S.committed.has(plan.epoch)) ? "queued" : "locked";
  return "queued";
}
function planEl(plan, old) {
  const box = el("div", "plan" + (old ? " old" : "")); box.style.setProperty("--ec", epVar(plan.epoch));
  const ph = el("div", "ph"); ph.append(el("span", "", `Epoch ${plan.epoch} plan`));
  const stat = old ? "superseded" : plan.status === "draft" ? "speculative" : plan.status;
  ph.append(chip(old ? "bad" : plan.status === "done" ? "ok" : plan.status === "committed" ? "info" : "", stat));
  box.appendChild(ph);
  const ul = el("ul", "steps");
  (plan.steps || []).forEach(s => {
    const ts = (S.tools.get(plan.epoch + ":" + s.tool) || { state: "cancelled" }).state;
    const st = old ? (ts === "done" || ts === "compensated" ? ts : "cancelled") : stepState(s, plan);
    const li = el("li", st + (s.write ? " wr" : " rd"));
    const si = el("span", "si");
    if (st === "running") si.appendChild(el("span", "spinner"));
    else si.appendChild(icon({ queued: "clock", locked: "lock", done: "check", cancelled: "x", blocked: "x", compensated: "undo" }[st] || "clock"));
    const rw = el("span", "rw " + (s.write ? "write" : "read"), s.write ? "WRITE" : "READ");
    li.append(si, stepLabel(s), rw, el("span", "sl", st === "locked" ? "locked until commit" : st));
    ul.appendChild(li);
  });
  box.appendChild(ul);
  return box;
}
function snapshotOldPlan() {
  const cur = $("#plan .plan:not(.old)");
  const plan = S.state && S.state.plan;
  if (!cur || !plan) { S.oldPlan = null; return; }
  S.oldPlan = JSON.parse(JSON.stringify(plan));
  S.oldPlan.stateAtBump = {};
}
function renderPlan() {
  const host = $("#plan"), plan = S.state && S.state.plan;
  if (!changed("plan", [plan, S.oldPlan, S.epoch, [...S.tools], [...S.committed]].map(x => x && JSON.parse(JSON.stringify(x, (k, v) => v instanceof Map ? [...v] : v))))) return;
  host.replaceChildren();
  if (S.oldPlan && (!plan || plan.epoch > S.oldPlan.epoch)) host.appendChild(planEl(S.oldPlan, true));
  if (!plan) { if (!S.oldPlan) host.appendChild(el("p", "none", "No plan yet")); return; }
  $("#plan-meta").textContent = plan.status === "draft" ? "reads speculative · writes locked" : "";
  host.appendChild(planEl(plan, false));
  if (plan.missing && plan.missing.length) host.appendChild(el("p", "hint", "Waiting for: " + plan.missing.join(", ")));
}
function renderInflight() {
  const ul = $("#inflight"); ul.replaceChildren();
  if (!S.tasks.size) { ul.appendChild(el("li", "none", "Nothing running")); return; }
  S.tasks.forEach(t => {
    const li = el("li", t.ended === "cancelled" ? "gone" : t.ended === "done" ? "done" : ""); li.style.setProperty("--ec", epVar(t.epoch));
    li.append(t.ended ? icon(t.ended === "cancelled" ? "x" : "check") : el("span", "spinner spin"),
              el("span", "grow", t.name), el("span", "ep", "E" + t.epoch), el("span", "el", ""));
    t.li = li; ul.appendChild(li);
  });
  tickElapsed();
}
function tickElapsed() {
  S.tasks.forEach(t => { if (t.li && !t.ended) $(".el", t.li).textContent = ((performance.now() - t.t0) / 1000).toFixed(2) + " s"; });
}
function renderLedger() {
  if (!changed("ledger", [S.guards, S.state && S.state.ledger])) return;
  const ul = $("#ledger"); ul.replaceChildren();
  const rows = [];
  S.guards.slice(0, 3).forEach(g => rows.push({ guard: g }));
  ((S.state && S.state.ledger) || []).slice(-6).reverse().forEach(l => rows.push({ l }));
  if (!rows.length) { ul.appendChild(el("li", "none", "No writes yet")); return; }
  const stIcon = { COMMITTED: "check", IN_FLIGHT: "clock", COMPENSATED: "undo", BLOCKED: "x", FAILED: "x" };
  rows.forEach(({ guard, l }) => {
    const li = el("li"); const ep = guard ? guard.epoch : l.epoch; li.style.setProperty("--ec", epVar(ep));
    const status = guard ? "BLOCKED" : l.status;
    if (guard && guard.kind === "dup") li.classList.add("dupe"); else if (guard) li.classList.add("blockrow");
    const s = el("span", "lstat " + status); s.append(icon(stIcon[status] || "clock", "ic"), guard && guard.kind === "dup" ? "DUPLICATE BLOCKED" : status.replace("_", " "));
    li.append(s, el("span", "grow", (guard ? guard.tool : l.tool)), el("span", "key", guard ? (guard.key ? String(guard.key).slice(0, 8) : guard.reason.replace(/_/g, " ")) : String(l.key).slice(0, 8)), el("span", "ep", "E" + ep));
    ul.appendChild(li);
  });
}
function renderSlots() {
  const host = $("#slots"), slots = (S.state && S.state.slots) || {};
  const keys = Object.keys(slots);
  if (!changed("slots", [slots, S.epoch, !!S.state])) return;
  host.replaceChildren();
  $("#snap-epoch").textContent = S.state ? "epoch " + S.epoch : "";
  if (!keys.length) { host.appendChild(el("span", "none", "No slots yet")); S.prevSlots = S.state ? {} : null; return; }
  keys.forEach(k => {
    const v = slots[k], had = S.prevSlots && Object.keys(S.prevSlots).length > 0;
    const hit = had && JSON.stringify(S.prevSlots[k]) !== JSON.stringify(v);   // a correction changed or added it
    const c = el("span", "slot" + (hit ? " changed" : "")); c.style.setProperty("--ec", epVar(S.epoch));
    c.append(k, el("b", "", typeof v === "object" ? JSON.stringify(v) : String(v)));
    host.appendChild(c);
  });
  S.prevSlots = JSON.parse(JSON.stringify(slots));
}

// =========================================================================== metrics bar ===
const shown = {};
function setMetric(id, text, cls, num) {
  const m = $("#" + id), v = $("[data-v]", m);
  if (shown[id] === text) return;
  const from = shown[id]; shown[id] = text;
  m.classList.remove("good", "bad"); if (cls) m.classList.add(cls);
  if (typeof num === "number" && from !== undefined && !isNaN(+from) && +from < num) {           // count up
    const a = +from, t0 = performance.now();
    const step = now => { const p = Math.min(1, (now - t0) / 380); v.textContent = String(Math.round(a + (num - a) * p)); if (p < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  } else v.textContent = text;
  m.classList.remove("tick"); void m.offsetWidth; m.classList.add("tick");
}
function renderMetrics(reset) {
  if (reset) Object.keys(shown).forEach(k => delete shown[k]);
  const m = S.m, stale = Math.max(m.stale, (S.state && S.state.metrics && S.state.metrics.stale_dropped) || 0);
  setMetric("m-epoch", S.epoch ? String(S.epoch) : "–", "", S.epoch || undefined);
  $("#m-epoch").style.setProperty("--ec", S.epoch ? epVar(S.epoch) : "var(--fg)");
  setMetric("m-ack", m.lastAck === null ? "–" : fmtMs(m.lastAck), m.lastAck === null ? "" : m.lastAck < 300 ? "good" : "bad");
  $("[data-sub]", $("#m-ack")).textContent = m.lastAck === null ? "target < 300 ms" : m.lastAck < 300 ? "under 300 ms ✓" : "over 300 ms ✗";
  const p50 = median(m.acks);
  setMetric("m-p50", p50 === null ? "–" : fmtMs(p50), "");
  $("[data-sub]", $("#m-p50")).textContent = m.acks.length ? `of ${m.acks.length} ack${m.acks.length === 1 ? "" : "s"}` : "no acks yet";
  setMetric("m-stale", String(stale), "", stale);
  $("#m-stale .sub").textContent = `${m.staleReached} reached the client`;
  setMetric("m-dup", String(m.dup), m.dup ? "bad" : "", m.dup);
  setMetric("m-commit", String(m.commit), "", m.commit);
  setMetric("m-comp", String(m.comp), "", m.comp);
}

// ================================================================================ talking ===
function setLive(on, text) {
  const live = $("#live");
  live.classList.toggle("speaking", !!on);
  $("#live-state").textContent = on ? "speaking…" : "idle";
  const t = $("#live-text"); t.replaceChildren();
  if (on) { t.append(text || ""); t.appendChild(el("span", "caret")); }
  else t.append(text || "Type below or press a scenario.");
  S.speaking = !!on; syncHot();
}
function syncHot() {
  const busy = [...S.tasks.values()].some(t => !t.ended && !/^slow:/.test(t.kind));
  document.body.classList.toggle("hot", S.speaking || busy);
}
async function speak(text, msPerWord, withFinal) {
  const token = ++S.speech, words = text.trim().split(/\s+/);
  for (let i = 0; i < words.length; i++) {
    if (token !== S.speech) { setLive(false); return false; }
    const part = words.slice(0, i + 1).join(" ");
    setLive(true, part); send("transcript_partial", { text: part });
    await sleep(msPerWord);
  }
  if (token !== S.speech) { setLive(false); return false; }
  setLive(false, "“" + text + "”");
  if (withFinal) { addUser(text); send("transcript_final", { text }); }
  else addUser(text, { note: "partials only: the server commits after a short silence" });
  return true;
}
function sayNow(text) { S.speech++; setLive(false, "“" + text + "”"); addUser(text); return send("transcript_final", { text }); }
async function sendUtterance() {
  const box = $("#utt"), text = box.value.trim();
  if (!text) return;
  box.value = "";
  if ($("#sim").checked) await speak(text, +$("#speed").value, $("#withfinal").checked); else sayNow(text);
}
function interrupt() {
  S.speech++; S.run++; S.chain++; setLive(false, "Interrupted.");
  markScenario(null); setCaption(null);
  const rec = addUser("Interrupt", { note: "explicit barge-in signal" });
  rec.slot.replaceChildren(chip("acted", "CANCEL", null, "Explicit interrupt event"), chip("tier", "explicit"));
  rec.chipped = true; S.lastInterrupt = rec;
  send("interrupt", { action: "cancel" });
}

function blobToB64(blob) {
  return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1] || ""); r.onerror = () => rej(r.error); r.readAsDataURL(blob); });
}
async function sendFrame(blob, name, question) {
  if (blob.size > 8 * 1024 * 1024) { toast("Image is over 8 MB, choose a smaller one"); return; }
  const url = URL.createObjectURL(blob);
  S.lastFrame = { url, name };
  const rec = addUser(question || "", { thumb: { url, name }, note: question ? "" : "camera frame, no question typed" });
  rec.wrap.querySelector(".meta .who").textContent = "You · camera";
  rec.chipped = true; rec.slot.appendChild(chip("info", "frame", "camera"));
  S.speech++; setLive(false);
  const payload = { b64: await blobToB64(blob), name };
  if (question) payload.text = question;
  send("camera_frame", payload);
}
async function sendSample(name, question) {
  try {
    const r = await fetch(`/samples/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(r.status);
    await sendFrame(await r.blob(), name, question);
  } catch { toast("Sample not available: " + name); }
}
function pickFile(file) {
  if (!file || !file.type.startsWith("image/")) { toast("Please choose an image file"); return; }
  const q = $("#utt").value.trim(); $("#utt").value = "";
  sendFrame(file, file.name, q);
}

// ================================================================================ scenarios ===
// Each scenario is a list of steps. A step with `cap` starts a narration beat: in presenter mode the caption is
// shown (and held) BEFORE that step runs. Steps without `cap` follow the previous one at native timing, so the
// interruption still lands mid-task. `outcome` is the closing caption once everything has settled.
const SCN = [
  { key: "incar", title: "In-car: goal change", desc: "“Navigate to Chennai Airport” → “Actually, gas station first”",
    expect: "epoch 1 route cancelled · epoch 2 planned · 1 set_navigation",
    outcome: "Epoch bump → the old route task was cancelled and its result dropped → one clean route via the gas station.",
    steps: [
      { cap: "A driver asks for a route… then changes their mind while it is still being calculated.", speak: "Navigate to Chennai Airport" },
      { wait: 60 },
      { say: "Actually, gas station first" }] },
  { key: "support_mid", title: "Support: fix before booking", desc: "“Book a flight to Delhi tomorrow” → “Actually, next week instead”",
    expect: "plan patched · reads reused · 1 booking, 1 charge",
    outcome: "Correction → epoch bump → the plan was patched, not restarted → exactly one booking and one charge.",
    steps: [
      { cap: "A customer asks for a flight, then corrects the date while the search is still running.", say: "Book a flight to Delhi tomorrow" },
      { wait: 90 },
      { say: "Actually, next week instead" }] },
  { key: "support_after", title: "Support: fix after booking", desc: "The first booking has already committed, then the date changes",
    expect: "stale booking cancelled by compensation · 1 live booking",
    outcome: "Epoch bump → the stale booking was cancelled by a compensating action through the same ledger → one new booking, one charge.",
    steps: [
      { cap: "A customer books a flight. The write commits, guarded by the epoch fence and the idempotency ledger.", say: "Book a flight to Delhi tomorrow" },
      { waitFor: "action_result" }, { wait: 500 },
      { cap: "Now they change their mind, after the write has already committed.", say: "Actually, next week instead" }] },
  { key: "field", title: "Field: camera + question", desc: "A camera frame with “What's wrong with this machine?”",
    expect: "ack → Moondream description → grounded diagnosis",
    outcome: "Moondream described the frame and the LLM diagnosed with the knowledge-base read tool: grounded in what the camera sees.",
    steps: [
      { cap: "A technician sends a camera frame with a question.", sample: "panel_disconnected_cable.png", text: "What's wrong with this machine?" }] },
  { key: "access", title: "Accessibility: hesitation", desc: "“I want to boo… um… actually, book a table for 4”",
    expect: "hesitation not acted on · self-correction · 1 reservation",
    outcome: "The hesitation was never acted on; the self-correction produced exactly one reservation.",
    steps: [
      { cap: "The user hesitates mid-word, then corrects themselves.", partial: "I want to boo…" },
      { wait: 450 }, { partial: "I want to boo… um…" }, { wait: 500 },
      { say: "actually, book a table for 4" }] },
];
let playing = null;
function buildScenarioCards() {
  const list = $("#scn-list");
  SCN.forEach((s, i) => {
    const c = el("div", "scn"); c.dataset.scn = s.key; c.title = `${s.desc}
Expect: ${s.expect}`;
    const btn = el("button", "btn sm play"); btn.append(icon("play"), "Play"); btn.setAttribute("aria-label", `Play scenario ${i + 1}: ${s.title}`);
    btn.onclick = () => { S.chain++; runScenario(s.key); };
    const exp = el("div", "exp"); exp.append(bold("Expect"), s.expect);
    c.append(el("div", "num", String(i + 1)), el("div", "ttl", s.title), btn, el("div", "dsc", s.desc), exp, el("div", "step", ""));
    list.appendChild(c);
  });
}
function markScenario(key, stepText) {
  $$(".scn").forEach(c => {
    const on = c.dataset.scn === key;
    c.classList.toggle("active", on);
    if (on) $(".step", c).textContent = stepText || "";
  });
}
function setCaption(text, i, n) {
  const cap = $("#caption");
  if (!text) { cap.hidden = true; return; }
  $("#cap-step").textContent = `Step ${i}/${n}`; $("#cap-text").textContent = text;
  cap.hidden = !document.documentElement.classList.contains("present");
  cap.dataset.text = text;
}
function idleFor(ms, cap = 25000) {
  return new Promise(res => {
    const t0 = performance.now();
    const tick = () => {
      const busy = [...S.tasks.values()].some(t => !t.ended);
      if ((!busy && performance.now() - S.lastActivity > ms) || performance.now() - t0 > cap) res(); else setTimeout(tick, 120);
    };
    tick();
  });
}
async function runScenario(key, { fresh = true } = {}) {
  const sc = SCN.find(x => x.key === key); if (!sc) return;
  if (fresh && S.events > 0) { startSession(newId()); await sleep(250); }
  for (let i = 0; i < 60 && !(S.ws && S.ws.readyState === WebSocket.OPEN); i++) await sleep(50);
  const token = ++S.run, present = document.documentElement.classList.contains("present");
  const beats = sc.steps.filter(x => x.cap).length + 1;                 // narration beats + the outcome
  let beat = 0;
  markScenario(key, `beat 0/${beats}`);
  for (const st of sc.steps) {
    if (token !== S.run) return;
    if ("wait" in st) { await sleep(st.wait); continue; }
    if ("waitFor" in st) { if (!(await awaitMessage(st.waitFor, 20000))) return; continue; }
    if (st.cap) {
      beat++; markScenario(key, `beat ${beat}/${beats}`); setCaption(st.cap, beat, beats);
      if (present) await sleep(2600);                                   // read it before it happens
      if (token !== S.run) return;
    }
    if ("say" in st) sayNow(st.say);
    else if ("speak" in st) await speak(st.speak, 70, true);
    else if ("partial" in st) { S.speech++; setLive(true, st.partial); send("transcript_partial", { text: st.partial }); }
    else if ("sample" in st) await sendSample(st.sample, st.text);
  }
  setLive(false, "");
  await idleFor(900);
  if (token !== S.run) return;
  markScenario(key, `done ✓ · ${sc.expect}`);
  setCaption(sc.outcome, beats, beats);
  return true;
}
async function playAll() {
  const chain = ++S.chain;                           // any interrupt / manual play bumps S.chain and stops this
  for (const sc of SCN) {
    if (chain !== S.chain) return;
    const ok = await runScenario(sc.key, { fresh: true });
    if (ok !== true || chain !== S.chain) return;
    await sleep(document.documentElement.classList.contains("present") ? 3200 : 1200);
  }
}

// ==================================================================== theme / presenter ===
function setTheme(t) {
  document.documentElement.dataset.theme = t; store.set("chronos-theme", t);
  const b = $("#theme"); b.replaceChildren(icon(t === "dark" ? "sun" : "moon"));
  b.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
  tlDirty();
}
function setPresent(on) {
  document.documentElement.classList.toggle("present", on);
  $("#present").setAttribute("aria-pressed", String(on));
  const cap = $("#caption"); cap.hidden = !on || !cap.dataset.text;
  const q = new URLSearchParams(location.search); if (on) q.set("present", "1"); else q.delete("present");
  history.replaceState(null, "", "?" + q.toString());
  setTimeout(() => { tlDirty(); }, 60);
}

// =========================================================================== live timeline ===
const LANES = [["perception", "Perception"], ["fast", "Fast path"], ["slow", "Slow path"], ["coordination", "Coordination"], ["tools", "Tools"]];
const TL = { bars: new Map(), ticks: [], bumps: [], first: null, last: 0, raf: 0, dirty: true, epochs: new Set() };
function tlReset() { TL.bars.clear(); TL.ticks = []; TL.bumps = []; TL.first = null; TL.last = 0; TL.epochs.clear(); $("#tl-legend").replaceChildren(); tlDirty(); }
function laneOfTask(t) { return /^slow:/.test(t) ? "slow" : /^(read:|write:|kb\+vision|compensate)/.test(t) ? "tools" : "coordination"; }
function shortTask(t) { return t.replace(/^slow:/, "").replace(/^read:/, "").replace(/^write:/, "").replace("kb+vision", "kb + vision"); }
function tlAdd(r) {
  const t = r.t_ms / 1000;
  if (TL.first === null) TL.first = t;
  TL.last = Math.max(TL.last, t);
  if (!TL.epochs.has(r.epoch)) {
    TL.epochs.add(r.epoch);
    const i = el("i"); i.style.background = epVar(r.epoch); const s = el("span"); s.append(i, "epoch " + r.epoch); $("#tl-legend").appendChild(s);
  }
  switch (r.event) {
    case "task_started": TL.bars.set(r.task_id, { lane: laneOfTask(r.task), epoch: r.epoch, t0: t, t1: null, label: shortTask(r.task), cancelled: false }); break;
    case "task_finished": { const b = TL.bars.get(r.task_id); if (b) b.t1 = t; break; }
    case "task_cancelled": { const b = TL.bars.get(r.task_id); if (b) { b.t1 = t; b.cancelled = true; } break; }
    case "epoch_bumped": TL.bumps.push({ t, epoch: r.epoch }); break;
    default: TL.ticks.push({ lane: LANES.some(l => l[0] === r.component) ? r.component : "coordination", t, epoch: r.epoch, event: r.event });
  }
  tlDirty();
}
function tlDirty() { TL.dirty = true; if (!TL.raf) TL.raf = requestAnimationFrame(tlLoop); }
function tlLoop() {
  TL.raf = 0;
  tlDraw();
  const open = [...TL.bars.values()].some(b => b.t1 === null);
  if (open || performance.now() - S.lastActivity < 3500) TL.raf = requestAnimationFrame(tlLoop);
}
function tlDraw() {
  const c = $("#tl"), box = c.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1, W = Math.max(50, box.width), H = Math.max(50, box.height);
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const mono = cssv("--font-mono") || "monospace";
  const fg = cssv("--fg"), muted = cssv("--muted"), faint = cssv("--faint"), line = cssv("--line-soft"), panel2 = cssv("--panel-2"), bad = cssv("--bad");
  const G = 96, AX = 18, laneH = (H - AX) / LANES.length, plotW = W - G - 8;
  // lanes
  ctx.font = `600 11px ${cssv("--font-display") || "sans-serif"}`; ctx.textBaseline = "middle";
  LANES.forEach(([, name], i) => {
    const y = i * laneH;
    if (i % 2 === 0) { ctx.fillStyle = panel2; ctx.fillRect(0, y, W, laneH); }
    ctx.fillStyle = muted; ctx.fillText(name.toUpperCase(), 10, y + laneH / 2);
  });
  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(G, 0); ctx.lineTo(G, H - AX); ctx.stroke();
  if (TL.first === null) { ctx.fillStyle = faint; ctx.font = `400 12px ${mono}`; ctx.fillText("Events appear here live as the agent works.", G + 14, (H - AX) / 2); return; }
  // time window: grows with the session up to 24 s, then scrolls
  const now = S.clock ? (S.clock.ms + (performance.now() - S.clock.at)) / 1000 : TL.last;
  const openBar = [...TL.bars.values()].some(b => b.t1 === null);
  const end = openBar ? now : Math.min(now, TL.last + 0.6);
  const dur = end - TL.first, span = Math.min(24, Math.max(5, dur + 1));
  const start = dur + 1 <= span ? TL.first - 0.35 : end - span + 0.15;          // grow from the left, then scroll
  const X = t => G + ((t - start) / span) * plotW;
  // axis
  const stepT = span > 16 ? 2 : span > 8 ? 1 : 0.5;
  ctx.font = `400 10px ${mono}`; ctx.fillStyle = faint; ctx.textBaseline = "alphabetic"; ctx.strokeStyle = line;
  for (let t = Math.ceil(start / stepT) * stepT; t < start + span; t += stepT) {
    const x = X(t); if (x < G || t < 0) continue;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - AX); ctx.stroke();
    ctx.fillText("+" + t.toFixed(stepT < 1 ? 1 : 0) + "s", x + 3, H - 5);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(G + 1, 0, plotW + 8, H - AX); ctx.clip();
  // bars, stacked into sub-rows inside a lane when they overlap
  const byLane = {}; TL.bars.forEach(b => (byLane[b.lane] = byLane[b.lane] || []).push(b));
  Object.entries(byLane).forEach(([lane, bars]) => {
    const li = LANES.findIndex(l => l[0] === lane); if (li < 0) return;
    bars.sort((a, b) => a.t0 - b.t0);
    const rows = [];
    bars.forEach(b => { b.end = b.t1 === null ? now : b.t1; let r = rows.findIndex(e => e <= b.t0 + 0.005); if (r < 0) { r = rows.length; rows.push(0); } rows[r] = b.end; b.row = r; });
    const n = Math.min(3, rows.length), bh = Math.min(16, (laneH - 6) / n);
    bars.forEach(b => {
      const x0 = X(b.t0), x1 = Math.max(x0 + 3, X(b.end)), y = li * laneH + 3 + Math.min(b.row, 2) * bh + (laneH - 6 - n * bh) / 2, col = cssv("--e" + (epIdx(b.epoch) + 1));
      ctx.globalAlpha = b.cancelled ? 0.55 : 1;
      ctx.fillStyle = col; ctx.globalAlpha *= b.cancelled ? 0.28 : 0.85; ctx.fillRect(x0, y, x1 - x0, bh - 2);
      ctx.globalAlpha = b.cancelled ? 0.9 : 1; ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.strokeRect(x0 + .5, y + .5, x1 - x0 - 1, bh - 3);
      if (b.cancelled) {                                   // hatched + struck through
        ctx.save(); ctx.beginPath(); ctx.rect(x0, y, x1 - x0, bh - 2); ctx.clip(); ctx.lineWidth = 1;
        for (let hx = x0 - bh; hx < x1; hx += 5) { ctx.beginPath(); ctx.moveTo(hx, y + bh); ctx.lineTo(hx + bh, y); ctx.stroke(); }
        ctx.restore();
        ctx.strokeStyle = bad; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x0, y + (bh - 2) / 2); ctx.lineTo(x1, y + (bh - 2) / 2); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (x1 - x0 > 46 && bh > 10) {
        ctx.save(); ctx.beginPath(); ctx.rect(x0 + 2, y, x1 - x0 - 4, bh - 2); ctx.clip();
        ctx.font = `500 10px ${mono}`; ctx.fillStyle = b.cancelled ? muted : cssv("--on-bar"); ctx.textBaseline = "middle";
        ctx.fillText(b.label, x0 + 4, y + (bh - 2) / 2 + .5); ctx.restore();
      }
    });
  });
  // ticks
  TL.ticks.forEach(k => {
    const li = LANES.findIndex(l => l[0] === k.lane), x = X(k.t); if (li < 0 || x < G - 4) return;
    const y = li * laneH + laneH / 2, col = cssv("--e" + (epIdx(k.epoch) + 1)); ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.6;
    if (k.event === "stale_result_dropped" || /^write_blocked/.test(k.event)) { ctx.strokeStyle = bad; ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke(); }
    else if (k.event === "ack_sent") { ctx.beginPath(); ctx.arc(x, y, 4, 0, 6.29); ctx.fill(); }
    else if (k.event === "write_committed") ctx.fillRect(x - 4, y - 4, 8, 8);
    else { ctx.beginPath(); ctx.moveTo(x, y - 3.5); ctx.lineTo(x + 3.5, y); ctx.lineTo(x, y + 3.5); ctx.lineTo(x - 3.5, y); ctx.closePath(); ctx.fill(); }
  });
  // epoch bumps
  TL.bumps.forEach(b => {
    const x = X(b.t), col = cssv("--e" + (epIdx(b.epoch) + 1));
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash([5, 3]); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - AX); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.font = `700 10px ${mono}`; ctx.textBaseline = "top"; ctx.fillText("⟳E" + b.epoch, x + 4, 2);
  });
  ctx.restore();
}

// ===================================================================================== boot ===
function renderHealth(h) {
  S.health = h || null;
  const mode = h ? h.llm_mode : "?", ol = (h && h.ollama) || {}, models = ol.models || {};
  const needs = mode === "ollama", ready = ol.reachable && Object.values(models).every(Boolean);
  setPill("pill-llm", h ? "ok" : "bad", h ? `LLM: ${needs ? mode : "local"}` : "LLM: ?", needs ? "Real local models" : "Simulated by the deterministic mock LLM (CHRONOS_LLM=mock, no GPU needed)");
  const one = (id, name) => {
    const sim = !!h && !needs;  // mock mode: show the model as running (simulated by the mock LLM)
    const ok = sim || !!models[name];
    setPill(id, ok ? "ok" : "bad", `${name} ${ok ? "✓" : "✗"}`, sim ? "Simulated by the mock LLM (no Ollama needed)" : ok ? "Model available" : "Model missing");
  };
  one("pill-llama", "llama3.2:3b"); one("pill-moon", "moondream");
  const b = $("#banner");
  if (needs && !ready) {
    b.className = "err"; b.hidden = false; h.bannerShown = true;
    b.replaceChildren();
    const t = el("b"); t.append(icon("alert"), ol.reachable ? "Some Ollama models are missing" : "Ollama is not reachable");
    const p = el("div"); p.append(ol.reachable ? "Pull them with " : `CHRONOS is in ollama mode but ${ol.url || "Ollama"} does not answer. Start it, or run in mock mode with `,
      el("code", "", ol.reachable ? "ollama pull llama3.2:3b && ollama pull moondream" : "CHRONOS_LLM=mock"), ". Meanwhile CHRONOS falls back to deterministic answers, so it never freezes.");
    b.append(t, p);
  } else { b.hidden = true; if (h) h.bannerShown = false; }
}
async function pollHealth() {
  try { renderHealth(await (await fetch("/health")).json()); } catch { renderHealth(null); }
  setTimeout(pollHealth, 15000);
}

function wire() {
  $("#send").onclick = sendUtterance;
  $("#utt").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); sendUtterance(); } });
  $("#interrupt").onclick = interrupt;
  $("#speed").oninput = e => { $("#speed-v").textContent = e.target.value + " ms/word"; };
  $("#img").onchange = e => { pickFile(e.target.files[0]); e.target.value = ""; };
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => pickFile(e.dataTransfer.files[0]));
  drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#img").click(); } });
  $$("[data-sample]").forEach(b => b.onclick = () => { const q = $("#utt").value.trim(); $("#utt").value = ""; sendSample(b.dataset.sample, q); });
  $("#newsess").onclick = () => startSession(newId());
  $("#copy").onclick = async () => {
    try { await navigator.clipboard.writeText(S.id); } catch { const r = document.createRange(); r.selectNodeContents($("#sid")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand("copy"); s.removeAllRanges(); }
    toast("Session id copied");
  };
  $("#theme").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  $("#present").onclick = () => setPresent(!document.documentElement.classList.contains("present"));
  $("#playall").onclick = playAll;
  $("#clear").onclick = () => { const id = S.id; resetUI(); S.id = id; };
  $("#jump").onclick = () => { toBottom(); };
  feed().addEventListener("scroll", jumpCheck, { passive: true });
  window.addEventListener("resize", tlDirty);
  if (window.ResizeObserver) new ResizeObserver(tlDirty).observe($("#tl").parentElement);
  setInterval(tickElapsed, 100);

  document.addEventListener("keydown", e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && e.target.type !== "checkbox" && e.target.type !== "range";
    if (e.key === "Escape") { e.preventDefault(); interrupt(); return; }
    if (typing) return;
    const k = e.key.toLowerCase();
    if (k >= "1" && k <= "5") { e.preventDefault(); S.chain++; runScenario(SCN[+k - 1].key); }
    else if (k === "n") { e.preventDefault(); startSession(newId()); }
    else if (k === "t") { e.preventDefault(); $("#theme").click(); }
    else if (k === "p") { e.preventDefault(); $("#present").click(); }
    else if (k === "/") { e.preventDefault(); $("#utt").focus(); }
  });
}

function boot() {
  buildScenarioCards(); wire();
  setTheme(store.get("chronos-theme") === "light" ? "light" : "dark");
  const q = new URLSearchParams(location.search);
  if (q.get("present") === "1") setPresent(true);
  const wanted = q.get("session") || "";
  startSession(ID_RE.test(wanted) ? wanted : newId());
  pollHealth();
  // shareable links: /?session=demo1&run=incar (also support_mid, support_after, field, access) [&present=1]
  const run = q.get("run");
  if (run && SCN.some(s => s.key === run)) runScenario(run, { fresh: false });
  if (q.get("run") === "all") playAll();
  window.__chronos = { S, onTrace, onOutput, runScenario, startSession };     // handy for tests and debugging
}
boot();
})();
