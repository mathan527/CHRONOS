/* Shared nav + theme for the CHRONOS pages. Builds the DOM with createElement only. */
(() => {
  const PAGES = [
    ["/", "Home"], ["/demo", "Live Demo"], ["/how-it-works", "How it works"],
    ["/results", "Results"], ["/timeline", "Timeline"],
  ];
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const root = document.documentElement;
  root.dataset.theme = store.get("chronos-theme") || "dark";

  const NS = "http://www.w3.org/2000/svg";
  const logo = document.createElementNS(NS, "svg");
  logo.setAttribute("viewBox", "0 0 32 32");
  logo.setAttribute("aria-hidden", "true");
  const c = document.createElementNS(NS, "circle");
  for (const [k, v] of Object.entries({ cx: 16, cy: 16, r: 13, fill: "none", stroke: "#d9a441", "stroke-width": 3 })) c.setAttribute(k, v);
  const h = document.createElementNS(NS, "path");
  for (const [k, v] of Object.entries({ d: "M16 8v8l5 3", fill: "none", stroke: "#8fb0c4", "stroke-width": 3, "stroke-linecap": "round" })) h.setAttribute(k, v);
  logo.append(c, h);

  const nav = document.createElement("nav");
  nav.className = "nav";
  nav.setAttribute("aria-label", "Main");
  const wrap = document.createElement("div");
  wrap.className = "wrap";
  const brand = document.createElement("a");
  brand.className = "brand"; brand.href = "/";
  brand.append(logo, "CHRONOS");
  const links = document.createElement("div");
  links.className = "links";
  for (const [href, label] of PAGES) {
    const a = document.createElement("a");
    a.href = href; a.textContent = label;
    if (location.pathname === href) a.setAttribute("aria-current", "page");
    links.appendChild(a);
  }
  const cta = document.createElement("a");
  cta.className = "cta"; cta.href = "/demo"; cta.textContent = "Try the demo";
  const theme = document.createElement("button");
  theme.className = "icon-btn"; theme.type = "button";
  const sync = () => {
    const dark = root.dataset.theme === "dark";
    theme.textContent = dark ? "☀" : "☾";
    theme.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  };
  theme.onclick = () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    store.set("chronos-theme", root.dataset.theme); sync();
  };
  sync();
  wrap.append(brand, links, cta, theme);
  nav.appendChild(wrap);
  document.body.prepend(nav);
})();
