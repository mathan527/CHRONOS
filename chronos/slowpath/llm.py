"""LLM client interface + implementations.

Every call takes a `task` tag (e.g. "intent", "bargein"), a system prompt and a user string, and
returns a parsed JSON object. Failures are typed so callers can apply deterministic fallbacks:
LLMTimeout (hard time limit hit), LLMInvalidJSON (model answered, but not with a JSON object),
LLMError (transport / HTTP failure).
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from chronos.config import Settings
from chronos.slowpath.diagnosis import deterministic_diagnosis


class LLMError(Exception):
    pass


class LLMTimeout(LLMError):
    pass


class LLMInvalidJSON(LLMError):
    pass


class LLM(Protocol):
    name: str

    async def chat_json(self, task: str, system: str, user: str, *,
                        timeout: float | None = None) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


# ------------------------------------------------------------------------- Ollama -----------
class OllamaLLM:
    """Ollama /api/chat with format=json. The timeout is enforced as a hard wall-clock bound
    (asyncio.timeout) on top of httpx's own, so a stalled model can never freeze the agent."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b",
                 timeout_s: float = 8.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name = f"ollama:{model}"
        self.model, self.timeout_s = model, timeout_s
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=None)

    async def chat_json(self, task: str, system: str, user: str, *,
                        timeout: float | None = None) -> dict[str, Any]:
        payload = {"model": self.model, "stream": False, "format": "json",
                   "options": {"temperature": 0},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        try:
            async with asyncio.timeout(timeout or self.timeout_s):
                resp = await self._client.post("/api/chat", json=payload)
        except (TimeoutError, httpx.TimeoutException) as e:
            raise LLMTimeout(f"{task}: timed out") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{task}: {type(e).__name__}: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"{task}: HTTP {resp.status_code}")
        try:
            content = resp.json()["message"]["content"]
            obj = json.loads(content)
        except (KeyError, TypeError, ValueError) as e:
            raise LLMInvalidJSON(f"{task}: unparseable response") from e
        if not isinstance(obj, dict):
            raise LLMInvalidJSON(f"{task}: expected a JSON object")
        return obj

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- Mock -----------
Handler = Callable[[dict[str, Any]], dict[str, Any]]


def _mock_bargein(p: dict[str, Any]) -> dict[str, Any]:
    """Deliberately coarse keyword scoring: a stand-in for a model's fuzzy judgement on the
    utterances the precise tier-1 rules abstain on."""
    t = p["utterance"].lower()
    toks = set(re.findall(r"[a-z']+", t))
    if toks & {"stop", "cancel", "nevermind", "forget", "abort"} or "never mind" in t:
        label = "cancel"
    elif t.rstrip().endswith("?") or toks & {"what", "which", "when", "where", "why", "how", "who"}:
        label = "clarification_question"
    elif toks & {"instead", "rather", "change", "switch", "meant", "wrong", "different"} \
            or "make it" in t:
        label = "correction"
    elif toks & {"also", "add", "plus", "another", "extra", "too"} or t.startswith("and "):
        label = "addition"
    elif toks & {"first", "before", "new", "actually", "now"}:
        label = "goal_change"
    elif len(toks) <= 2 and toks & {"ok", "okay", "yes", "yeah", "right", "sure", "fine"}:
        label = "backchannel"
    else:
        label = "hesitation"
    return {"label": label, "confidence": 0.6}


def _mock_intent(p: dict[str, Any]) -> dict[str, Any]:
    t = p["utterance"].lower()
    slots: dict[str, Any] = {}
    m = re.search(r"\bto\s+([a-z ]+?)(?:\s+(?:on|at|by|for|tomorrow|today)\b|$)", t)
    if re.search(r"\b(flight|fly|ticket)", t):
        intent = "cancel_booking" if "cancel" in t else "book_flight"
        if m and intent == "book_flight":
            slots["dest"] = m.group(1).strip().title()
    elif re.search(r"\b(table|restaurant|dinner|lunch|breakfast)\b", t):
        intent = "reserve_table"
    elif re.search(r"\b(navigate|route|drive|directions|map|road)\b", t):
        intent = "navigate"
        if m:
            slots["destination"] = m.group(1).strip().title()
    elif re.search(r"\b(machine|broken|fault|noise|error|wrong)\b", t):
        intent, slots = "troubleshoot", {"symptom": p["utterance"]}
    else:
        intent = "smalltalk"
    return {"intent": intent, "slots": slots, "confidence": 0.6}


def _mock_diagnose(p: dict[str, Any]) -> dict[str, Any]:
    d = deterministic_diagnosis(p.get("question", ""), p.get("frame_description"),
                                p.get("kb_matches") or [])
    return {"diagnosis": d.diagnosis, "likely_cause": d.likely_cause, "steps": d.steps}


class MockLLM:
    """Deterministic, rule-based, no network. `handlers` override/extend per-task behaviour;
    `latency_s` simulates a slow model (used to test timeouts)."""

    name = "mock"

    def __init__(self, handlers: dict[str, Handler] | None = None, latency_s: float = 0.0) -> None:
        self.handlers: dict[str, Handler] = {"bargein": _mock_bargein, "intent": _mock_intent,
                                         "diagnose": _mock_diagnose}
        self.handlers.update(handlers or {})
        self.latency_s = latency_s
        self.calls: list[tuple[str, str]] = []

    async def chat_json(self, task: str, system: str, user: str, *,
                        timeout: float | None = None) -> dict[str, Any]:
        self.calls.append((task, user))
        try:
            async with asyncio.timeout(timeout):
                if self.latency_s:
                    await asyncio.sleep(self.latency_s)
                handler = self.handlers.get(task)
                if handler is None:
                    raise LLMError(f"mock has no handler for task {task!r}")
                try:
                    payload = json.loads(user)
                except ValueError:
                    payload = {"utterance": user}
                return handler(payload)
        except TimeoutError as e:
            raise LLMTimeout(f"{task}: timed out") from e

    async def aclose(self) -> None:
        return None


def make_llm(settings: Settings) -> LLM:
    if settings.llm_mode == "ollama":
        return OllamaLLM(settings.ollama_url, settings.llm_model, settings.llm_timeout_s)
    return MockLLM()
