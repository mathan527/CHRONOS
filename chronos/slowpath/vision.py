"""Vision grounding: describe the latest camera frame. Moondream via Ollama, or a canned mock."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from chronos.config import Settings
from chronos.slowpath.llm import LLMError, LLMInvalidJSON, LLMTimeout

PROMPT = ("Describe what you see on this machine or panel in one or two sentences. Focus on "
          "cables and plugs, indicator lights, fans and vents, switches, damage and dust.")


@dataclass(frozen=True)
class Frame:
    name: str
    data: bytes = field(repr=False)

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @classmethod
    def from_path(cls, path: str | Path) -> Frame:
        p = Path(path)
        return cls(p.name, p.read_bytes())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Frame:
        """CAMERA_FRAME event payload: {"path": ...} or {"b64": ..., "name": ...}."""
        if "path" in payload:
            return cls.from_path(payload["path"])
        if "b64" in payload:
            return cls(str(payload.get("name", "frame")), base64.b64decode(payload["b64"]))
        raise ValueError("camera payload needs 'path' or 'b64'")


class FrameStore:
    """Keeps only the latest frame of a session."""

    def __init__(self) -> None:
        self._latest: Frame | None = None
        self._at = 0.0
        self.received = 0

    def put(self, frame: Frame) -> None:
        self._latest, self._at, self.received = frame, time.perf_counter(), self.received + 1

    def latest(self) -> Frame | None:
        return self._latest

    def age_s(self) -> float | None:
        """Seconds since the latest frame arrived (None if there is none)."""
        return None if self._latest is None else time.perf_counter() - self._at


class Vision(Protocol):
    name: str

    async def describe(self, frame: Frame, *, timeout: float | None = None) -> str: ...

    async def aclose(self) -> None: ...


class OllamaVision:
    """Moondream through /api/generate with images=[base64]. Hard wall-clock timeout."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "moondream",
                 timeout_s: float = 20.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name = f"ollama:{model}"
        self.model, self.timeout_s = model, timeout_s
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=None)

    async def describe(self, frame: Frame, *, timeout: float | None = None) -> str:
        payload = {"model": self.model, "prompt": PROMPT, "images": [frame.b64], "stream": False,
                   "options": {"temperature": 0}}
        try:
            async with asyncio.timeout(timeout or self.timeout_s):
                resp = await self._client.post("/api/generate", json=payload)
        except (TimeoutError, httpx.TimeoutException) as e:
            raise LLMTimeout("vision: timed out") from e
        except httpx.HTTPError as e:
            raise LLMError(f"vision: {type(e).__name__}: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"vision: HTTP {resp.status_code}")
        try:
            text = str(resp.json()["response"]).strip()
        except (KeyError, TypeError, ValueError) as e:
            raise LLMInvalidJSON("vision: unparseable response") from e
        if not text:
            raise LLMInvalidJSON("vision: empty description")
        return text

    async def aclose(self) -> None:
        await self._client.aclose()


CANNED: dict[str, str] = {
    "panel_disconnected_cable.png": (
        "A grey control panel with a power cable hanging loose, unplugged from its socket. "
        "The green status LED is dark."),
    "machine_overheating_fan.png": (
        "A machine whose cooling fan has stopped, with dust blocking the vents and a red "
        "temperature warning light lit."),
    "breaker_tripped.png": (
        "An electrical panel where one circuit breaker switch has tripped to the middle "
        "position while the others are on."),
}
UNKNOWN_IMAGE = "An image that is too unclear to describe."


class MockVision:
    """Canned descriptions keyed by image filename. `calls` records every describe()."""

    name = "mock-vision"

    def __init__(self, canned: dict[str, str] | None = None, latency_s: float = 0.0) -> None:
        self.canned = {k.lower(): v for k, v in {**CANNED, **(canned or {})}.items()}
        self.latency_s = latency_s
        self.calls: list[str] = []

    async def describe(self, frame: Frame, *, timeout: float | None = None) -> str:
        self.calls.append(frame.name)
        try:
            async with asyncio.timeout(timeout):
                if self.latency_s:
                    await asyncio.sleep(self.latency_s)
        except TimeoutError as e:
            raise LLMTimeout("vision: timed out") from e
        return self.canned.get(Path(frame.name).name.lower(), UNKNOWN_IMAGE)

    async def aclose(self) -> None:
        return None


def make_vision(settings: Settings) -> Vision:
    if settings.llm_mode == "ollama":
        return OllamaVision(settings.ollama_url, settings.vision_model)
    return MockVision()
