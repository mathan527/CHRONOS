"""WCAG AA (4.5:1) for the demo UI's text colours, computed from the CSS variables themselves."""
import re
from pathlib import Path

import pytest

CSS = (Path(__file__).resolve().parent.parent / "web" / "app.css").read_text(encoding="utf-8")


def _block(marker: str) -> dict[str, str]:
    i = CSS.index(marker)
    return dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", CSS[i:CSS.index("}", i)]))


DARK = _block(":root {")
LIGHT = {**DARK, **_block(':root[data-theme="light"]')}


def _lum(h: str) -> float:
    def f(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _mix(fg: str, bg: str, pct: float) -> str:  # color-mix(in srgb, fg pct%, bg)
    def f(i: int) -> int:
        return round(int(fg[i:i + 2], 16) * pct + int(bg[i:i + 2], 16) * (1 - pct))
    return f"#{f(1):02x}{f(3):02x}{f(5):02x}"


TEXT = ("--fg", "--muted", "--faint", "--accent", "--ok", "--bad", "--warn", "--info",
        "--e1", "--e2", "--e3", "--e4")
STATUS = ("--ok", "--bad", "--warn", "--info", "--e1", "--e2", "--e3", "--e4")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_colours_meet_wcag_aa_on_their_surfaces(theme):
    t = DARK if theme == "dark" else LIGHT
    failures = []
    for c in TEXT:
        for surface in ("--bg", "--panel", "--panel-2"):
            if (r := ratio(t[c], t[surface])) < 4.5:
                failures.append(f"{c} on {surface}: {r:.2f}")
    for c in STATUS:  # status chips and badges: the colour over a 16% tint of itself
        if (r := ratio(t[c], _mix(t[c], t["--panel-2"], 0.16))) < 4.5:
            failures.append(f"{c} on its tint: {r:.2f}")
    if (r := ratio(t["--on-accent"], t["--accent"])) < 4.5:
        failures.append(f"primary button: {r:.2f}")
    if (r := ratio(t["--on-bad"], t["--bad"])) < 4.5:
        failures.append(f"hot Interrupt button: {r:.2f}")
    for c in ("--e1", "--e2", "--e3", "--e4"):
        if (r := ratio(t["--on-bar"], t[c])) < 4.5:
            failures.append(f"timeline label on {c}: {r:.2f}")
    assert failures == []
