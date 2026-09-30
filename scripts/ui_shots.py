"""Screenshots of the demo UI: every flow in dark, light and presenter mode at 1920x1080.

    python scripts/ui_shots.py                       # all flows x 3 modes -> docs/screenshots/
    python scripts/ui_shots.py --flows incar field --modes dark
    python scripts/ui_shots.py --base http://localhost:8000 --browser chrome

Needs the server running (`make run`) and Playwright (`pip install -e .[ui]`). `--browser msedge`
or `chrome` uses the installed browser; `chromium` needs `playwright install chromium`.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = {"incar": "incar", "support_mid": "support_mid", "support_after": "support_after",
             "field": "field", "access": "access"}
FLOWS = [*SCENARIOS, "camera"]  # camera = a sample frame sent with an empty text box


async def shoot(page: Page, base: str, flow: str, mode: str, out: Path) -> Path:
    sid = f"shot-{flow}-{mode}-{time.time_ns() % 100000}"
    await page.add_init_script(
        f"try {{ localStorage.setItem('chronos-theme', '{'light' if mode == 'light' else 'dark'}') }} catch (e) {{}}")
    await page.goto(f"{base}/demo?session={sid}")
    await page.wait_for_selector("#pill-ws.ok", timeout=10000)
    if mode == "present":
        await page.click("#feed", position={"x": 5, "y": 5})
        await page.keyboard.press("p")
    if flow == "camera":
        await page.click("[data-sample='panel_disconnected_cable.png']")
        await page.wait_for_selector(".vision", timeout=15000)
    else:
        await page.click(f".scn[data-scn='{SCENARIOS[flow]}'] .play")
        await page.wait_for_function(
            "document.querySelector('.scn.active .step')?.textContent.startsWith('done')", timeout=60000)
    await page.wait_for_timeout(700)  # let the last animations settle
    await page.evaluate("document.querySelector('#col-left').scrollTop = 0")
    await page.wait_for_timeout(200)
    path = out / f"{flow}-{mode}.png"
    await page.screenshot(path=str(path))
    return path


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--flows", nargs="*", default=FLOWS, choices=FLOWS)
    ap.add_argument("--modes", nargs="*", default=["dark", "light", "present"],
                    choices=["dark", "light", "present"])
    ap.add_argument("--browser", default="msedge", help="msedge | chrome | chromium")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        kw = {} if a.browser == "chromium" else {"channel": a.browser}
        browser = await p.chromium.launch(headless=True, **kw)
        for mode in a.modes:
            for flow in a.flows:
                ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
                page = await ctx.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                path = await shoot(page, a.base, flow, mode, out)
                print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
                      f"{'  ERRORS: ' + '; '.join(errors) if errors else ''}", flush=True)
                await ctx.close()
        # a 1280-wide check (the smallest layout we design for)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()
        await page.goto(f"{a.base}/demo?session=shot-1280-{time.time_ns() % 100000}")
        await page.wait_for_selector("#pill-ws.ok")
        await page.click(".scn[data-scn='support_after'] .play")
        await page.wait_for_function(
            "document.querySelector('.scn.active .step')?.textContent.startsWith('done')", timeout=60000)
        await page.wait_for_timeout(600)
        await page.screenshot(path=str(out / "support_after-1280.png"))
        print(out / "support_after-1280.png")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
