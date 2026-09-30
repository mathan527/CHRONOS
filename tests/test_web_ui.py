"""Smoke tests for the demo client: it is served, its parts exist, and app.js only reaches for
elements that index.html actually has (a typo there would silently break the whole page)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from test_api import make_app

WEB = Path(__file__).resolve().parent.parent / "web"

KEY_IDS = [
    # header + status pills
    "sid", "copy", "newsess", "pill-ws", "pill-llm", "pill-llama", "pill-moon", "trace-link",
    "state-link", "present", "theme",
    # metrics bar
    "m-epoch", "m-ack", "m-p50", "m-stale", "m-dup", "m-commit", "m-comp",
    # left column
    "live", "utt", "send", "sim", "speed", "withfinal", "interrupt", "drop", "img", "scn-list", "playall",
    # centre + right
    "feed", "empty", "jump", "banner", "epoch-rail", "plan", "inflight", "ledger", "slots",
    # bottom strip + presenter caption
    "tl", "tl-link", "caption", "cap-text",
]


@pytest.fixture
def client(fast_settings):
    with TestClient(make_app(fast_settings)) as c:
        yield c


def html_ids(html: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', html))


def test_the_page_and_its_assets_are_served(client):
    page = client.get("/demo")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/assets/app.css" in page.text and "/assets/app.js" in page.text
    css, js = client.get("/assets/app.css"), client.get("/assets/app.js")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    assert client.get("/samples/panel_disconnected_cable.png").status_code == 200


def test_assets_route_only_serves_whitelisted_names(client):
    assert client.get("/assets/nope.js").status_code == 404
    assert client.get("/assets/index.html").status_code == 404      # not an asset type
    assert client.get("/assets/..%2Fserver.py").status_code in (404, 422)
    assert client.get("/assets/app.py").status_code == 404


def test_every_key_element_exists(client):
    ids = html_ids(client.get("/demo").text)
    assert [i for i in KEY_IDS if i not in ids] == []


def test_the_page_has_the_five_lanes_of_controls_and_the_keyboard_hint(client):
    text = client.get("/demo").text
    for word in ("Agent brain", "Conversation", "Live timeline", "Idempotency ledger",
                 "State snapshot", "In-flight tool calls", "Interrupt"):
        assert word in text
    assert 'aria-live="polite"' in text and 'role="log"' in text
    assert re.search(r"<kbd>Esc</kbd>\s*interrupt", text) and "<kbd>N</kbd>" in text
    scripts = re.findall(r"<script[^>]*src=\"([^\"]+)\"", text)
    assert scripts == ["/assets/app.js"]  # no CDN scripts: only Google Fonts may be remote
    assert not re.findall(r"(?:src|href)=\"https?://(?!fonts\.g)", text)


def test_app_js_only_uses_ids_that_exist_in_the_html():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"', js))
    used |= set(re.findall(r"\$\(\"#([A-Za-z0-9_-]+) ", js))
    # ids the script creates itself are not in the static HTML
    assert used and sorted(used - html_ids(html)) == []


def test_app_js_has_no_external_network_calls():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert not re.findall(r"(?:fetch|WebSocket|import)\(?\s*[\"'`]https?://", js)


def test_css_defines_the_four_epoch_colours_in_both_themes():
    css = (WEB / "app.css").read_text(encoding="utf-8")
    light = css.split(':root[data-theme="light"]')[1].split("}")[0]
    dark = css.split(":root {")[1].split("}")[0]
    for n in range(1, 5):
        assert f"--e{n}:" in dark and f"--e{n}:" in light
    assert "prefers-reduced-motion" in css


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_app_js_parses():
    r = subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
