"""Drives the demo UI in a real browser and checks what a viewer would see: keyboard shortcuts,
classification chips, epoch dividers, agent-brain panels, camera flows and presenter mode.

    python scripts/ui_e2e.py                    # against http://localhost:8000 with Microsoft Edge
    python scripts/ui_e2e.py --base http://localhost:8000 --browser chrome

Needs the server running (`make run`) and Playwright (`pip install -e .[ui]`).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
FRAME = str(ROOT / "scenarios" / "images" / "panel_disconnected_cable.png")
RESULTS: list[bool] = []


def check(name: str, cond: object, extra: str = "") -> None:
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--browser", default="msedge", help="msedge | chrome | chromium")
    a = ap.parse_args()
    async with async_playwright() as p:
        kw = {} if a.browser == "chromium" else {"channel": a.browser}
        b = await p.chromium.launch(headless=True, **kw)
        page = await (await b.new_context(viewport={"width": 1920, "height": 1080})).new_page()
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        # a unique session each run: the server keeps sessions (and their ledger) alive
        await page.goto(f"{a.base}/demo?session=e2e-{time.time_ns() % 1000000}")
        await page.wait_for_selector("#pill-ws.ok")
        check("empty state visible", await page.is_visible("#empty"))

        await page.uncheck("#sim")  # send words at once: faster than streaming them
        await page.fill("#utt", "Book a flight to Delhi tomorrow")
        await page.press("#utt", "Enter")
        await page.wait_for_selector(".msg.agent .badge.done", timeout=8000)
        check("empty state hidden after the first message", not await page.is_visible("#empty"))
        check("user bubble has a classification chip", await page.locator(".msg.user .chip.acted").count() >= 1)
        check("ack latency chip present", await page.locator(".chip.ok:has-text('ack')").count() >= 1)
        when = await page.locator(".msg .when").first.inner_text()
        check("relative timestamp (+x.xx s)", when.startswith("+") and when.endswith(" s"), when)
        title = await page.locator(".msg .when").first.get_attribute("title")
        check("absolute time on hover", bool(title) and ":" in title, title or "")

        await page.fill("#utt", "Book a table for 2")
        await page.press("#utt", "Enter")
        await page.wait_for_timeout(120)
        await page.fill("#utt", "okay")
        await page.press("#utt", "Enter")
        await page.wait_for_selector(".chip.muted:has-text('BACKCHANNEL')", timeout=5000)
        check("BACKCHANNEL chip is muted and says 'ignored, not acted on'",
              await page.locator(".chip.muted:has-text('ignored, not acted on')").count() >= 1)

        await page.press("#utt", "Escape")
        await page.wait_for_timeout(250)
        check("Esc creates an interrupt bubble", await page.locator(".msg.user:has-text('Interrupt')").count() >= 1)

        await page.click("#feed", position={"x": 5, "y": 5})
        await page.keyboard.press("t")
        theme = await page.evaluate("document.documentElement.dataset.theme")
        check("T toggles the theme", theme == "light", theme)
        await page.keyboard.press("t")
        old = await page.inner_text("#sid")
        await page.keyboard.press("n")
        await page.wait_for_timeout(600)
        check("N starts a new session", (await page.inner_text("#sid")) != old and await page.is_visible("#empty"))

        await page.keyboard.press("1")
        await page.wait_for_timeout(900)
        check("1 highlights the scenario card", await page.locator(".scn.active[data-scn='incar']").count() == 1)
        await page.wait_for_selector(".bumpline", timeout=12000)
        txt = await page.inner_text(".bumpline")
        check("epoch divider shows epochs, reason and counts",
              "Epoch 1 → 2" in txt and "GOAL CHANGE" in txt and "cancelled" in txt and "reused" in txt, txt)
        await page.wait_for_timeout(2500)
        check("old work is struck through", await page.locator(".msg.gone").count() >= 1)
        check("epoch rail has two entries", await page.locator("#epoch-rail li:not(.none)").count() == 2)
        check("plan panel has READ and WRITE tags",
              await page.locator("#plan .rw.read").count() >= 1 and await page.locator("#plan .rw.write").count() >= 1)
        check("a changed slot is highlighted", await page.locator("#slots .slot.changed").count() >= 1)
        check("ledger has rows", await page.locator("#ledger li:not(.none)").count() >= 1)
        check("metrics bar shows epoch 2", (await page.inner_text("#m-epoch [data-v]")).strip() == "2")

        for i in range(6):
            await page.fill("#utt", f"Book a table for {i + 2}")
            await page.press("#utt", "Enter")
            await page.wait_for_timeout(100)
        await page.wait_for_timeout(1500)
        await page.evaluate("document.querySelector('#feed').scrollTop = 0")
        await page.wait_for_timeout(300)
        await page.fill("#utt", "Book a table for 9")
        await page.press("#utt", "Enter")
        await page.wait_for_timeout(1200)
        check("'jump to latest' shows when scrolled up", await page.is_visible("#jump"))
        await page.click("#jump")
        await page.wait_for_timeout(800)
        check("'jump to latest' hides after use", not await page.is_visible("#jump"))

        await page.keyboard.press("Escape")
        await page.click("#newsess")
        await page.wait_for_timeout(600)
        await page.fill("#utt", "why is it not powering on")
        await page.click("[data-sample='breaker_tripped.png']")
        await page.wait_for_selector(".vision", timeout=8000)
        check("frame + typed question -> vision card", await page.locator(".vision .vsec").count() >= 3)
        check("vision card shows the frame", await page.locator(".vision .shot img").count() == 1)
        await page.click("[data-sample='machine_overheating_fan.png']")
        await page.wait_for_function("document.querySelectorAll('.vision').length === 2", timeout=8000)
        check("frame with an EMPTY text box is answered",
              await page.locator(".msg.user:has-text('wrong with this?')").count() >= 1)
        await page.set_input_files("#img", FRAME)
        await page.wait_for_function("document.querySelectorAll('.vision').length === 3", timeout=8000)
        check("an uploaded file is answered", True)

        await page.evaluate("window.__chronos.onTrace({event:'write_blocked_duplicate', component:'coordination',"
                            " epoch:1, t_ms: 9999, tool:'reserve_table', key:'abcdef123456'})")
        await page.wait_for_timeout(250)
        check("a duplicate-blocked row (red) appears in the ledger", await page.locator("#ledger li.dupe").count() == 1)
        check("the duplicate metric ticks to 1", (await page.inner_text("#m-dup [data-v]")).strip() == "1")

        await page.click("#feed", position={"x": 5, "y": 5})
        await page.keyboard.press("p")
        await page.wait_for_timeout(200)
        check("presenter mode on", await page.evaluate("document.documentElement.classList.contains('present')"))
        check("presenter mode hides the JSON viewers", await page.evaluate(
            "[...document.querySelectorAll('details.data')].every(d => getComputedStyle(d).display === 'none')"))
        await page.keyboard.press("p")

        check("no console or page errors", not errs, "; ".join(errs[:3]))
        await b.close()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
