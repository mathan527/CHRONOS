"""Robustness of the demo UI: the "Ollama unreachable" card and WebSocket reconnect with backoff.

    python scripts/ui_robust.py [--browser msedge|chrome|chromium]

Starts its own throwaway servers on ports 8011 and 8012, so it does not disturb `make run`.
Needs Playwright (`pip install -e .[ui]`). Also writes docs/screenshots/{ollama-unreachable,reconnecting}.png.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
RESULTS: list[bool] = []


def check(name: str, cond: object, extra: str = "") -> None:
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)


def start(port: int, **env: str) -> subprocess.Popen[bytes]:
    e = {**os.environ, "PYTHONIOENCODING": "utf-8", "CHRONOS_TRACE_DIR": str(ROOT / "traces"),
         "CHRONOS_DB": ":memory:", **env}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chronos.api.server:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_up(port: int, secs: float = 15) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a system proxy
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            opener.open(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:  # noqa: BLE001 - still starting
            time.sleep(0.3)
    return False


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--browser", default="msedge", help="msedge | chrome | chromium")
    a = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        kw = {} if a.browser == "chromium" else {"channel": a.browser}
        b = await p.chromium.launch(headless=True, **kw)

        # 1. ollama mode with nothing listening: a friendly card, red pills, and it still answers
        srv = start(8011, CHRONOS_LLM="ollama", OLLAMA_HOST="127.0.0.1:59999")
        check("ollama-mode server starts", wait_up(8011))
        page = await (await b.new_context(viewport={"width": 1920, "height": 1080})).new_page()
        await page.goto("http://localhost:8011/demo?session=ollama-down")
        await page.wait_for_selector("#banner.err:not([hidden])", timeout=8000)
        banner = await page.inner_text("#banner")
        check("card says Ollama is not reachable", "not reachable" in banner)
        check("card explains the fallback", "never freezes" in banner)
        check("model pill is red in ollama mode", await page.locator("#pill-llama.bad").count() == 1)
        check("LLM pill says ollama", "ollama" in (await page.inner_text("#pill-llm")))
        await page.screenshot(path=str(SHOTS / "ollama-unreachable.png"))
        await page.uncheck("#sim")
        await page.fill("#utt", "Navigate to Chennai Airport")
        await page.press("#utt", "Enter")
        await page.wait_for_selector(".result", timeout=15000)
        check("the agent still answers with Ollama down", True)
        srv.terminate()
        srv.wait(10)

        # 2. websocket reconnect with backoff
        srv = start(8012)
        check("second server starts", wait_up(8012))
        page = await (await b.new_context(viewport={"width": 1920, "height": 1080})).new_page()
        await page.goto("http://localhost:8012/demo?session=reconnect-1")
        await page.wait_for_selector("#pill-ws.ok", timeout=8000)
        srv.terminate()
        srv.wait(10)
        await page.wait_for_selector("#pill-ws.warn", timeout=8000)
        txt = await page.inner_text("#pill-ws")
        check("pill shows 'reconnecting…' while the server is down", "reconnecting" in txt, txt)
        await page.screenshot(path=str(SHOTS / "reconnecting.png"))
        await page.wait_for_timeout(2500)  # a couple of backoff rounds
        srv = start(8012)
        check("server restarted", wait_up(8012))
        await page.wait_for_selector("#pill-ws.ok", timeout=15000)
        check("pill returns to connected by itself", True)
        await page.uncheck("#sim")
        await page.fill("#utt", "Book a table for 4")
        await page.press("#utt", "Enter")
        await page.wait_for_selector(".result", timeout=10000)
        check("the session works after the reconnect", True)
        srv.terminate()
        await b.close()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
