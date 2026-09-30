import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from chronos.slowpath.llm import LLMError, LLMInvalidJSON, LLMTimeout
from chronos.slowpath.vision import (
    CANNED,
    UNKNOWN_IMAGE,
    Frame,
    FrameStore,
    MockVision,
    OllamaVision,
)

IMAGES = Path(__file__).parent.parent / "scenarios" / "images"
SAMPLES = ["panel_disconnected_cable.png", "machine_overheating_fan.png", "breaker_tripped.png"]


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_images_are_real_pngs_with_canned_descriptions(name):
    path = IMAGES / name
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    with Image.open(path) as im:
        assert im.size == (640, 480)
    assert name in CANNED


def test_frame_from_path_and_b64_payload():
    f = Frame.from_path(IMAGES / SAMPLES[0])
    assert f.name == SAMPLES[0] and f.data[:4] == b"\x89PNG"
    g = Frame.from_payload({"b64": f.b64, "name": "cam.png"})
    assert g.data == f.data and g.name == "cam.png" and g.sha == f.sha
    assert Frame.from_payload({"path": str(IMAGES / SAMPLES[0])}).sha == f.sha
    with pytest.raises(ValueError):
        Frame.from_payload({})


def test_frame_store_keeps_only_latest():
    s = FrameStore()
    assert s.latest() is None
    a, b = Frame("a.png", b"a"), Frame("b.png", b"b")
    s.put(a)
    s.put(b)
    assert s.latest() is b and s.received == 2


async def test_mock_vision_canned_by_filename():
    v = MockVision()
    d = await v.describe(Frame.from_path(IMAGES / "panel_disconnected_cable.png"))
    assert "cable" in d and "loose" in d
    assert "fan" in await v.describe(Frame.from_path(IMAGES / "machine_overheating_fan.png"))
    assert "breaker" in await v.describe(Frame.from_path(IMAGES / "breaker_tripped.png"))
    assert await v.describe(Frame("mystery.jpg", b"x")) == UNKNOWN_IMAGE
    assert await v.describe(Frame("PANEL_DISCONNECTED_CABLE.PNG", b"x")) == CANNED[
        "panel_disconnected_cable.png"]  # case-insensitive
    assert len(v.calls) == 5


async def test_mock_vision_timeout():
    with pytest.raises(LLMTimeout):
        await MockVision(latency_s=1).describe(Frame("x.png", b"x"), timeout=0.05)


def ollama(handler, **kw) -> OllamaVision:
    return OllamaVision(transport=httpx.MockTransport(handler), **kw)


async def test_ollama_vision_request_shape():
    seen = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"response": "  A loose cable.  "})

    frame = Frame.from_path(IMAGES / SAMPLES[0])
    assert await ollama(h).describe(frame) == "A loose cable."
    assert seen["url"].endswith("/api/generate")
    b = seen["body"]
    assert b["model"] == "moondream" and b["stream"] is False
    assert b["images"] == [frame.b64] and base64.b64decode(b["images"][0]) == frame.data
    assert "cable" in b["prompt"].lower()


async def test_ollama_vision_typed_failures():
    f = Frame("x.png", b"x")
    with pytest.raises(LLMInvalidJSON):
        await ollama(lambda _r: httpx.Response(200, json={"nope": 1})).describe(f)
    with pytest.raises(LLMInvalidJSON):
        await ollama(lambda _r: httpx.Response(200, json={"response": "  "})).describe(f)
    with pytest.raises(LLMError):
        await ollama(lambda _r: httpx.Response(503)).describe(f)

    def conn(_r):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        await ollama(conn).describe(f)


async def test_ollama_vision_hard_timeout():
    async def slow(_r):
        await asyncio.sleep(2)
        return httpx.Response(200, json={"response": "late"})

    with pytest.raises(LLMTimeout):
        await ollama(slow, timeout_s=0.05).describe(Frame("x.png", b"x"))
