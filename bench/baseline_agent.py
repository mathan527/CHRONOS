"""A naive, sequential, half-duplex agent: the baseline CHRONOS is measured against.

It gets EXACTLY the same building blocks as CHRONOS: the same world and tools (via the same
registry and argument schemas), the same intent extractor and slot parser, the same LLM and vision
clients, the same diagnosis code, the same rule-based utterance labelling. What it does not have
is any coordination:

  * it waits for a complete utterance (a FINAL, or a silence timeout on partials) and acts on
    nothing earlier: no speculation, no read cache;
  * it plans, executes, and only then speaks: there is no acknowledgment;
  * an interrupt cancels whatever is in flight and RESTARTS the goal from scratch, with nothing
    protecting a write that is already dispatching: no epochs, no fencing token, no idempotency
    ledger, and no compensation of stale writes;
  * new input is only heard between handling steps (one event at a time, FIFO, no priorities).

Two concessions keep it from being a strawman (both are visible in results.md):
  * `filter_noise=True` (default) lets it ignore "okay"/"um" using the same rules CHRONOS uses;
    the strict variant (`filter_noise=False`) treats every input as an interrupt and restarts.
  * "cancel that" / "stop" does an explicit undo of the writes it remembers from the CURRENT
    goal (calling the compensating tool directly). It has no notion of undoing a *stale* write
    when the user changes their mind: a restart simply forgets earlier effects.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import ValidationError

from bench.instrument import WriteTap
from chronos.agent import responses as R
from chronos.perception.bargein import BargeInContext, tier1
from chronos.perception.intent import Intent, IntentExtractor
from chronos.protocol import (
    BargeInType,
    Event,
    EventType,
    OutputMessage,
    OutputStatus,
    OutputType,
    ToolResult,
)
from chronos.slowpath.diagnose import diagnose_with_llm
from chronos.slowpath.llm import LLM, LLMError
from chronos.slowpath.vision import Frame, FrameStore, Vision
from chronos.tools.registry import REGISTRY, ToolRegistry
from chronos.tools.world import ToolError, World

B = BargeInType


class BaselineAgent:
    name = "baseline"

    def __init__(self, session_id: str, *, world: World, llm: LLM, vision: Vision,
                 today: date | None = None, silence_ms: int = 300, filter_noise: bool = True,
                 tap: WriteTap | None = None, registry: ToolRegistry = REGISTRY,
                 llm_timeout_s: float = 8.0, vision_timeout_s: float = 20.0) -> None:
        self.session_id, self.world, self.llm, self.vision = session_id, world, llm, vision
        self.today, self.filter_noise, self.tap, self.registry = today, filter_noise, tap, registry
        self.llm_timeout_s, self.vision_timeout_s = llm_timeout_s, vision_timeout_s
        self.silence_s = silence_ms / 1000
        self.extractor = IntentExtractor(llm, today=today, timeout_s=llm_timeout_s)
        self.frames = FrameStore()
        self.outputs: list[OutputMessage] = []
        self._subs: list[Callable[[OutputMessage], None]] = []
        self.queue: asyncio.Queue[Event] = asyncio.Queue()  # plain FIFO: no priorities
        self.intent: Intent | None = None
        self.slots: dict[str, Any] = {}
        self.done_writes: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self._task: asyncio.Task[None] | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._timer: asyncio.Task[None] | None = None
        self._partial = ""
        self._handling = False

    # ---------------------------------------------------------------- plumbing ---------------
    def subscribe(self, cb: Callable[[OutputMessage], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run())

    async def submit(self, event: Event) -> Event:
        await self.queue.put(event)
        return event

    def _say(self, type_: OutputType, status: OutputStatus, text: str,
             data: dict[str, Any] | None = None) -> None:
        msg = OutputMessage(type=type_, session_id=self.session_id, epoch=0, status=status,
                            text=text, data=data or {})
        self.outputs.append(msg)
        for cb in list(self._subs):
            with contextlib.suppress(Exception):
                cb(msg)

    @property
    def idle(self) -> bool:
        return (self.queue.empty() and not self._handling
                and not (self._timer and not self._timer.done())
                and not (self._task and not self._task.done()))

    async def wait_idle(self, timeout: float = 10.0) -> None:
        deadline = time.perf_counter() + timeout
        while True:
            if self.idle:
                await asyncio.sleep(0.02)
                if self.idle:
                    return
            if time.perf_counter() > deadline:
                raise TimeoutError("baseline did not become idle")
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        for t in (self._loop_task, self._timer, self._task):
            if t:
                t.cancel()
        await asyncio.gather(*(t for t in (self._loop_task, self._timer, self._task) if t),
                             return_exceptions=True)
        for closer in (self.llm.aclose, self.vision.aclose):
            with contextlib.suppress(Exception):
                await closer()

    # ------------------------------------------------------------------ the loop -------------
    async def _run(self) -> None:
        while True:
            event = await self.queue.get()
            self._handling = True
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a naive agent still must not die on one event
                self._say(OutputType.ERROR, OutputStatus.FAILED, "Sorry, something went wrong.")
            finally:
                self._handling = False

    async def _handle(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        t = event.type
        if t is EventType.TRANSCRIPT_PARTIAL:
            self._partial = text  # naive endpointing: act when the speaker goes quiet
            self._cancel_timer()
            self._timer = asyncio.create_task(self._silence_then_commit(text))
        elif t in (EventType.TRANSCRIPT_FINAL, EventType.TEXT):
            self._cancel_timer()
            self._partial = ""
            if text:
                await self._utterance(text)
        elif t is EventType.CAMERA_FRAME:
            self.frames.put(await asyncio.to_thread(Frame.from_payload, dict(event.payload)))
            if text:
                await self._utterance(text)
        elif t is EventType.INTERRUPT:
            if text:
                await self._utterance(text)
            elif event.payload.get("action") == "cancel":
                await self._stop()

    def _cancel_timer(self) -> None:
        if self._timer and not self._timer.done() and self._timer is not asyncio.current_task():
            self._timer.cancel()
        self._timer = None

    async def _silence_then_commit(self, text: str) -> None:
        await asyncio.sleep(self.silence_s)
        self._timer = None
        await self.queue.put(Event(session_id=self.session_id, type=EventType.TRANSCRIPT_FINAL,
                                   payload={"text": text, "committed": True}))

    # ----------------------------------------------------------------- understanding ---------
    async def _utterance(self, text: str) -> None:
        ctx = BargeInContext(self.intent) if self.intent is not None else None
        d = tier1(text, final=True, context=ctx, today=self.today)
        label = d.label if d else None
        noise = label in (B.BACKCHANNEL, B.HESITATION)
        if noise and self.filter_noise:
            return
        if label is B.CANCEL:
            await self._stop()
            return
        if noise:  # strict/naive variant: any input is an interrupt, so restart what we had
            if self.intent is None:
                self._say(OutputType.RESPONSE, OutputStatus.PENDING, "Sorry, I didn't catch that.")
                return
            await self._cancel_task()
            self._launch()
            return
        res = await self.extractor.extract(text)  # half-duplex: this blocks the loop
        intent, slots = self._merge(label, res.intent, res.slots, res.source, text)
        if intent is None:
            self._say(OutputType.RESPONSE, OutputStatus.PENDING, "Sorry, I didn't catch that.")
            return
        await self._cancel_task()  # RESTART: kills whatever is in flight, nothing fences a write
        self.intent, self.slots = intent, slots
        self._launch()

    def _merge(self, label: BargeInType | None, new_intent: Intent, new_slots: dict[str, Any],
               source: str, text: str) -> tuple[Intent | None, dict[str, Any]]:
        cur = self.intent
        if cur is not None and label in (B.CORRECTION, B.ADDITION):
            updates = self.extractor.slot_updates(text, cur)
            return cur, {**self.slots, **updates}
        if new_intent is Intent.ADD_STOP:  # "X first": the same merge CHRONOS performs
            slots: dict[str, Any] = {"via": new_slots["stop"]} if "stop" in new_slots else {}
            if cur is Intent.NAVIGATE and "destination" in self.slots:
                slots["destination"] = self.slots["destination"]
            return Intent.NAVIGATE, slots
        if new_intent is Intent.SMALLTALK and source == "fallback":
            return None, {}
        return new_intent, dict(new_slots)

    # ------------------------------------------------------------------------ doing ----------
    def _launch(self) -> None:
        self.done_writes = []  # a restart forgets what earlier attempts already did
        assert self.intent is not None
        self._task = asyncio.create_task(self._pipeline(self.intent, dict(self.slots)))

    async def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _stop(self) -> None:
        """'stop' / 'cancel that': stop what is running and undo what THIS attempt remembers."""
        await self._cancel_task()
        undone = 0
        for tool, args, result in self.done_writes:
            spec = self.registry.get(tool)
            comp = spec.compensator if spec else None
            if comp is None:
                continue
            cspec = self.registry.get(comp.tool)
            try:
                parsed = cspec.schema.model_validate(comp.build_args(args, result))
                await self.world.delay(comp.tool)
                await cspec.fn(self.world, parsed)
                undone += 1
            except (ToolError, ValidationError):
                pass
        self.done_writes, self.intent, self.slots = [], None, {}
        self._say(OutputType.RESPONSE, OutputStatus.CANCELLED,
                  "Okay, I've stopped that." + (" I undid what I had done." if undone else ""))

    async def _call(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        spec = self.registry.get(tool)
        try:
            parsed = spec.schema.model_validate(args)
            await self.world.delay(tool)
            return await spec.fn(self.world, parsed)
        except (ToolError, ValidationError):
            return None

    async def _write(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch a write with no commit gate, no fence and no ledger."""
        spec = self.registry.get(tool)
        try:
            parsed = spec.schema.model_validate(args)
        except ValidationError:
            return None
        clean = parsed.model_dump(mode="json")
        attempt = self.tap.begin(tool, clean) if self.tap else None
        try:
            await self.world.delay(tool)
            data = await spec.fn(self.world, parsed)
        except asyncio.CancelledError:
            if attempt:
                self.tap.end(attempt, "cancelled")  # type: ignore[union-attr]
            raise
        except ToolError:
            if attempt:
                self.tap.end(attempt, "failed")  # type: ignore[union-attr]
            return None
        if attempt:
            self.tap.end(attempt, "committed")  # type: ignore[union-attr]
        self.done_writes.append((tool, clean, data))
        return data

    @staticmethod
    def _pick(slots: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: slots[k] for k in keys if k in slots}

    async def _pipeline(self, intent: Intent, slots: dict[str, Any]) -> None:
        """understand (done) -> plan -> execute -> speak, strictly in that order."""
        say = self._say
        if intent is Intent.SMALLTALK:
            say(OutputType.RESPONSE, OutputStatus.DONE, "Hello! How can I help?")
        elif intent is Intent.TROUBLESHOOT:
            await self._diagnose(str(slots.get("symptom") or ""))
        elif intent is Intent.BOOK_FLIGHT:
            if "dest" not in slots or "date" not in slots:
                say(OutputType.RESPONSE, OutputStatus.PENDING,
                    R.missing_prompt(tuple(k for k in ("dest", "date") if k not in slots)))
                return
            found = await self._call("search_flights", self._pick(slots, ("origin", "dest", "date")))
            flights = list((found or {}).get("flights", []))
            if slots.get("time"):
                flights = [f for f in flights if f["time"] == slots["time"]]
            if not flights:
                say(OutputType.RESPONSE, OutputStatus.FAILED, R.blocked_text("no_flights"))
                return
            args = {**self._pick(slots, ("origin", "dest", "date", "seat_pref", "passenger")),
                    "time": slots.get("time") or flights[0]["time"]}
            await self._finish("book_flight", args)
        elif intent is Intent.NAVIGATE:
            if "destination" not in slots:
                say(OutputType.RESPONSE, OutputStatus.PENDING, R.missing_prompt(("destination",)))
                return
            route = await self._call("get_route", self._pick(slots, ("destination", "via")))
            if route is None:
                say(OutputType.RESPONSE, OutputStatus.FAILED, R.blocked_text("route_failed"))
                return
            await self._finish("set_navigation", self._pick(slots, ("destination", "via")))
        elif intent is Intent.RESERVE_TABLE:
            if "party_size" not in slots:
                say(OutputType.RESPONSE, OutputStatus.PENDING, R.missing_prompt(("party_size",)))
                return
            keys = ("party_size", "date", "time", "restaurant")
            avail = await self._call("check_table_availability", self._pick(slots, keys))
            if not (avail or {}).get("count"):
                say(OutputType.RESPONSE, OutputStatus.FAILED, R.blocked_text("no_table"))
                return
            await self._finish("reserve_table", self._pick(slots, keys))
        else:
            say(OutputType.RESPONSE, OutputStatus.FAILED, R.blocked_text("no_write_for_intent"))

    async def _finish(self, tool: str, args: dict[str, Any]) -> None:
        data = await self._write(tool, args)
        if data is None:
            self._say(OutputType.RESPONSE, OutputStatus.FAILED, "Sorry, that didn't work.")
            return
        result = ToolResult(call_id="baseline", tool=tool, data=data)
        self._say(OutputType.ACTION_RESULT, OutputStatus.DONE, R.write_result_text(result),
                  {"tool": tool, **data})

    async def _diagnose(self, question: str) -> None:
        desc = None
        frame = self.frames.latest()
        if frame is not None:  # strictly sequential: look, then look things up, then think
            try:
                desc = await self.vision.describe(frame, timeout=self.vision_timeout_s)
            except (LLMError, TimeoutError):
                desc = None
        kb = await self._call("lookup_troubleshooting_kb",
                              {"query": question, "observations": desc or ""})
        matches = list((kb or {}).get("matches", []))
        diag = await diagnose_with_llm(self.llm, question, desc, matches, self.llm_timeout_s)
        self._say(OutputType.RESPONSE, OutputStatus.DONE, R.diagnosis_text(diag.model_dump()),
                  {"diagnosis": diag.model_dump()})
