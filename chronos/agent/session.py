"""AgentSession: one consumer loop that wires perception, fast path, coordination and slow path.

    event --> perception (classify) --> [epoch bump + cancel, synchronous] --> fast-path ack
          --> slow-path task registered under the current epoch --> results emitted ONLY through
              CancellationManager.emit_if_current --> writes ONLY through planner.execute ->
              ToolExecutor.execute_write (commit gate + epoch fence + idempotency ledger)

The loop itself never awaits the slow path: everything slow (LLM intent fallback, tool reads,
writes, compensation, vision) runs in tasks. The only await on the model inside the loop is the
barge-in classifier's tier 2, which has a hard 250 ms timeout and a deterministic fallback.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from chronos.agent import responses as R
from chronos.config import Settings
from chronos.coordination.cancellation import CancellationManager
from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger
from chronos.coordination.snapshot import SnapshotStore, thaw
from chronos.events.queue import EventQueue
from chronos.fastpath.ack import AckBuilder, fmt_time
from chronos.perception.bargein import BargeInClassifier, BargeInContext, tier1
from chronos.perception.eou import EndOfUtteranceDetector
from chronos.perception.intent import Intent, IntentExtractor, norm
from chronos.protocol import (
    BargeInType,
    Event,
    EventType,
    OutputMessage,
    OutputStatus,
    OutputType,
    PlanStatus,
)
from chronos.slowpath.llm import LLM, make_llm
from chronos.slowpath.planner import Plan, PlannerPolicy, PlanOutcome, SpeculativePlanner
from chronos.slowpath.vision import Frame, Vision, make_vision
from chronos.tools.registry import ToolExecutor
from chronos.tools.world import World
from chronos.trace.logger import TraceLogger

B = BargeInType
_AFFIRM = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|go ahead|do it|"
                     r"proceed|please do|haan|theek hai)(?: please)?$")
_DUP_WINDOW_S = 10.0
DEFAULT_FRAME_QUESTION = "What's wrong with this?"


@dataclass
class SessionMetrics:
    events: int = 0
    outputs: int = 0
    ack_latencies_ms: list[float] = field(default_factory=list)
    labels: Counter = field(default_factory=Counter)
    tiers: Counter = field(default_factory=Counter)
    ignored: Counter = field(default_factory=Counter)
    stale_emitted: int = 0  # must stay 0: an output whose epoch was not current when sent


class AgentSession:
    def __init__(self, session_id: str, *, settings: Settings, tracer: TraceLogger,
                 epochs: EpochManager, cancellation: CancellationManager,
                 snapshots: SnapshotStore, ledger: IdempotencyLedger, world: World,
                 executor: ToolExecutor, extractor: IntentExtractor,
                 classifier: BargeInClassifier, planner: SpeculativePlanner, ack: AckBuilder,
                 llm: LLM, vision: Vision, today: date | None = None) -> None:
        self.session_id, self.settings, self.tracer = session_id, settings, tracer
        self.epochs, self.cancellation, self.snapshots = epochs, cancellation, snapshots
        self.ledger, self.world, self.executor = ledger, world, executor
        self.extractor, self.classifier, self.planner, self.ack = (
            extractor, classifier, planner, ack)
        self.llm, self.vision = llm, vision
        self.queue = EventQueue()
        self.metrics = SessionMetrics()
        self.outputs: list[OutputMessage] = []
        self._subs: list[Callable[[OutputMessage], None]] = []
        self._lock = asyncio.Lock()  # serialises planner mutations across slow-path tasks
        self._loop_task: asyncio.Task[None] | None = None
        self._handling = False
        self._draft_open = False  # the current partial stream is (re)drafting a goal
        self._last_eou: tuple[str, float] | None = None
        self._last_hesitation = ""
        self._frame_timer: asyncio.Task[None] | None = None  # default question for a bare frame
        self._perception = tracer.bind("perception")
        self._fast = tracer.bind("fast")
        self._slow_trace = tracer.bind("slow")
        self.eou = EndOfUtteranceDetector(
            self._eou_commit, silence_ms=settings.eou_silence_ms, on_hold=self._eou_hold,
            context=self._barge_context)

    # ------------------------------------------------------------------ lifecycle ------------
    @classmethod
    async def create(cls, session_id: str, settings: Settings | None = None, *,
                     llm: LLM | None = None, vision: Vision | None = None,
                     world: World | None = None, policy: PlannerPolicy | None = None,
                     today: date | None = None) -> AgentSession:
        settings = settings or Settings.from_env()
        tracer = TraceLogger(session_id, settings.trace_dir)
        epochs = EpochManager(session_id, tracer.bind("coordination"), start=1)  # 1-based
        cancellation = CancellationManager(epochs, tracer.bind("coordination"))
        ledger = await IdempotencyLedger.open(settings.db_path, tracer.bind("coordination"))
        world = world or await World.create(":memory:", latency_ms=settings.tool_latency_ms,
                                            today=today)
        llm = llm or make_llm(settings)
        vision = vision or make_vision(settings)
        snapshots = SnapshotStore(session_id)
        executor = ToolExecutor(session_id, world, epochs, ledger, trace=tracer.bind("tools"))
        extractor = IntentExtractor(llm, today=today, timeout_s=settings.llm_timeout_s,
                                    trace=tracer.bind("perception"))
        classifier = BargeInClassifier(llm, timeout_s=settings.classifier_llm_timeout_s,
                                       today=today, trace=tracer.bind("perception"))
        planner = SpeculativePlanner(
            session_id, epochs=epochs, cancellation=cancellation, snapshots=snapshots,
            executor=executor, extractor=extractor, llm=llm, vision=vision,
            trace=tracer.bind("slow"), policy=policy, llm_timeout_s=settings.llm_timeout_s)
        ack = AckBuilder((lambda: today) if today else None)
        return cls(session_id, settings=settings, tracer=tracer, epochs=epochs,
                   cancellation=cancellation, snapshots=snapshots, ledger=ledger, world=world,
                   executor=executor, extractor=extractor, classifier=classifier,
                   planner=planner, ack=ack, llm=llm, vision=vision, today=today)

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run(), name=f"loop:{self.session_id}")

    async def aclose(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        self.eou.reset()
        self._cancel_frame_timer()
        tasks = self.cancellation.active()
        for t in tasks:  # shutdown: cancel everything, protected writes included
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.executor.aclose()
        await self.planner.aclose()
        for closer in (self.llm.aclose, self.vision.aclose, self.ledger.close, self.world.close):
            with contextlib.suppress(Exception):  # best-effort teardown
                await closer()
        self.tracer.close()

    # --------------------------------------------------------------------- I/O -----------
    def subscribe(self, cb: Callable[[OutputMessage], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def subscribe_trace(self, cb: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Live trace rows (opt-in stream for UIs; outputs are unaffected)."""
        return self.tracer.subscribe(cb)

    async def submit(self, event: Event) -> Event:
        """Enqueue a client event (stamped with the epoch it arrived in)."""
        if event.session_id != self.session_id:
            raise ValueError(f"event for session {event.session_id!r} sent to {self.session_id!r}")
        stamped = event.model_copy(update={"epoch": self.epochs.current()})
        self.metrics.events += 1
        self._perception.emit("event_received", epoch=stamped.epoch, type=stamped.type.value,
                              event_id=stamped.event_id, text=stamped.payload.get("text"))
        await self.queue.put(stamped)
        return stamped

    def _sink(self, msg: OutputMessage, t_recv: float | None) -> None:
        """The last step before a message leaves the agent. Reached only via emit_if_current."""
        if msg.epoch != self.epochs.current():  # unreachable by construction; counted, not hidden
            self.metrics.stale_emitted += 1
        self.metrics.outputs += 1
        self.outputs.append(msg)
        if msg.type is OutputType.ACK:
            lat = (time.perf_counter() - t_recv) * 1000 if t_recv is not None else 0.0
            self.metrics.ack_latencies_ms.append(lat)
            self._fast.emit("ack_sent", epoch=msg.epoch, text=msg.text, latency_ms=round(lat, 3))
        else:
            self._slow_trace.emit("response_sent", epoch=msg.epoch, type=msg.type.value,
                                  status=msg.status.value, text=msg.text)
        for cb in list(self._subs):
            with contextlib.suppress(Exception):  # a broken subscriber must not break the agent
                cb(msg)

    def _emit(self, msg: OutputMessage, *, t_recv: float | None = None) -> bool:
        return self.cancellation.emit_if_current(
            msg.epoch, msg, lambda m: self._sink(m, t_recv), what=f"{msg.type.value}:{msg.status.value}")

    def _say(self, type_: OutputType, status: OutputStatus, text: str, *, epoch: int,
             data: dict[str, Any] | None = None) -> bool:
        return self._emit(OutputMessage(type=type_, session_id=self.session_id, epoch=epoch,
                                        status=status, text=text, data=data or {}))

    # ------------------------------------------------------------------- the loop ----------
    async def _run(self) -> None:
        while True:
            event = await self.queue.get()
            self._handling = True
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the loop must survive any single event
                self._perception.emit("loop_error", epoch=self.epochs.current(),
                                      error=f"{type(e).__name__}: {e}", event_type=event.type.value)
                self._say(OutputType.ERROR, OutputStatus.FAILED, "Sorry, something went wrong.",
                          epoch=self.epochs.current(), data={"event_id": event.event_id})
            finally:
                self._handling = False

    async def _absorb_frames(self, event: Event) -> None:
        """Restore causal order: strict priority lets a transcript overtake a camera frame the
        client sent BEFORE it, which would make the agent answer 'what is wrong with this?'
        without the picture. Process those earlier frames first."""
        for frame_event in self.queue.take_where(
                lambda e: e.type is EventType.CAMERA_FRAME
                and e.ts_monotonic <= event.ts_monotonic):
            await self._on_camera(frame_event)

    async def _handle(self, event: Event) -> None:
        if event.type is not EventType.CAMERA_FRAME:
            await self._absorb_frames(event)
        t = event.type
        if t is not EventType.CAMERA_FRAME and (
                t is not EventType.INTERRUPT or event.payload.get("text")):
            self._cancel_frame_timer()  # the user said something: no default question needed
        if t is EventType.TRANSCRIPT_PARTIAL:
            await self._on_partial(event)
        elif t in (EventType.TRANSCRIPT_FINAL, EventType.TEXT):
            await self._on_final(event)
        elif t is EventType.CAMERA_FRAME:
            await self._on_camera(event)
        elif t is EventType.INTERRUPT:
            await self._on_interrupt(event)

    # ---------------------------------------------------------------- event handlers -------
    def _may_speculate(self) -> bool:
        """Partials may (re)draft a goal only if that cannot orphan a goal we could still undo."""
        g = self.planner.goal
        if g is None or g.status in ("cancelled", "failed"):
            return True
        if g.status == "done":
            return not self.planner.within_undo_window(g)
        return g.status == "draft" and self._draft_open

    async def _on_partial(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        t0 = time.perf_counter()
        cue = tier1(text, final=False, context=self._barge_context())
        if (cue is not None and cue.label is B.HESITATION
                and cue.reason in ("trailing_ellipsis", "filler_only")
                and text != self._last_hesitation):  # an explicit "um..." / "boo…": trace it once
            self._last_hesitation = text
            self.metrics.ignored["hesitation_partial"] += 1
            self._utt(text, B.HESITATION.value, "rules", (time.perf_counter() - t0) * 1000,
                      acted=False, note=cue.reason, partial=True)
        if self._may_speculate():
            await self.planner.on_partial(text)  # rules only, no await on anything slow
            g = self.planner.goal
            self._draft_open = bool(g and g.status == "draft")
        await self.eou.feed_partial(text)

    async def _eou_commit(self, text: str) -> None:
        self._last_eou = (self._key(text), time.perf_counter())
        await self.queue.put(Event(session_id=self.session_id, type=EventType.TRANSCRIPT_FINAL,
                                   payload={"text": text, "committed": True, "source": "eou"},
                                   epoch=self.epochs.current()))

    async def _eou_hold(self, text: str) -> None:
        self.metrics.ignored["hesitation_held"] += 1
        self._perception.emit("hesitation_held", epoch=self.epochs.current(), text=text)

    @staticmethod
    def _key(text: str) -> str:
        return re.sub("[^a-z0-9 ]", "", norm(text)).strip()

    async def _on_final(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        self.eou.reset()  # a final supersedes any buffered partial
        if not text:
            return
        if not event.payload.get("committed") and self._last_eou and (
                self._key(text) == self._last_eou[0]
                and time.perf_counter() - self._last_eou[1] < _DUP_WINDOW_S):
            self.metrics.ignored["duplicate_final"] += 1
            self._perception.emit("duplicate_final_ignored", epoch=self.epochs.current(),
                                  text=text)
            return
        await self._utterance(text, event.ts_monotonic)

    async def _on_camera(self, event: Event) -> None:
        try:
            frame: Frame = await asyncio.to_thread(Frame.from_payload, dict(event.payload))
        except Exception as e:  # noqa: BLE001 - bad client payload must not kill the loop
            self._say(OutputType.ERROR, OutputStatus.FAILED,
                      "I couldn't read that camera frame.", epoch=self.epochs.current(),
                      data={"error": f"{type(e).__name__}: {e}"})
            return
        self.planner.on_camera_frame(frame)
        text = str(event.payload.get("text", "")).strip()
        if text:  # frame + question in one event (use case 3)
            self._cancel_frame_timer()
            await self._utterance(text, event.ts_monotonic)
        elif not self._goal_running():
            self._arm_frame_timer()  # a picture with no words: ask the obvious question

    # A frame sent without a question means "what's wrong with this?". Words may still follow
    # (the user points the camera, then speaks), so wait one end-of-utterance silence first.
    def _arm_frame_timer(self) -> None:
        self._cancel_frame_timer()
        self._frame_timer = asyncio.create_task(self._frame_default_question(),
                                                name="frame-default-question")

    def _cancel_frame_timer(self) -> None:
        t, self._frame_timer = self._frame_timer, None
        if t is not None and not t.done() and t is not asyncio.current_task():
            t.cancel()

    async def _frame_default_question(self) -> None:
        await asyncio.sleep(self.settings.eou_silence_ms / 1000)
        self._frame_timer = None
        self._perception.emit("frame_default_question", epoch=self.epochs.current(),
                              text=DEFAULT_FRAME_QUESTION)
        await self.queue.put(Event(
            session_id=self.session_id, type=EventType.TRANSCRIPT_FINAL,
            payload={"text": DEFAULT_FRAME_QUESTION, "committed": True, "source": "frame_default"},
            epoch=self.epochs.current()))

    async def _on_interrupt(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if text:
            await self._utterance(text, event.ts_monotonic)
        elif event.payload.get("action") == "cancel":
            await self._process(B.CANCEL, "(interrupt: cancel)", event.ts_monotonic)
        else:  # e.g. speech onset: a signal only, no state change
            self._perception.emit("interrupt_signal", epoch=self.epochs.current(),
                                  payload=dict(event.payload))

    # ------------------------------------------------------------------ utterances ---------
    def _barge_context(self) -> BargeInContext | None:
        g = self.planner.goal
        if g is None or g.status in ("cancelled", "failed"):
            return None
        return BargeInContext(g.intent)

    def _awaiting_slots(self) -> bool:
        p, g = self.planner.plan, self.planner.goal
        return bool(g and g.status == "draft" and p and p.missing)

    def _awaiting_confirmation(self) -> bool:
        p, g = self.planner.plan, self.planner.goal
        return bool(g and g.status == "draft" and p and not p.missing
                    and p.status is PlanStatus.DRAFT and p.writes)

    def _slot_fill(self, text: str) -> dict[str, Any] | None:
        """If the agent just asked for a missing slot, an answer like 'four' or 'tomorrow' fills
        it in (no interruption: it is neither a correction nor a new goal)."""
        g, p = self.planner.goal, self.planner.plan
        if g is None or p is None:
            return None
        d = tier1(text, final=True)
        if d is not None and d.label in (B.BACKCHANNEL, B.HESITATION, B.CANCEL):
            return None  # "okay" / "um" / "stop" are not answers to a slot question
        return self.extractor.slot_updates(text, g.intent, p.missing) or None

    def _utt(self, text: str, label: str, tier: str, ms: float, *, acted: bool,
             note: str = "", partial: bool = False) -> None:
        """One trace row per utterance: what it was taken to be, by which tier, how fast, and
        whether the agent acted on it (the demo UI's classification chip)."""
        self._perception.emit("utterance", epoch=self.epochs.current(), text=text, label=label,
                              tier=tier, classify_ms=round(ms, 3), acted=acted, note=note,
                              partial=partial)

    async def _utterance(self, text: str, t_recv: float) -> None:
        continuing = self._draft_open  # this final closes the partial stream that drafted a goal
        # 1. answers to something we asked (a missing slot / an explicit confirmation)
        if self._awaiting_confirmation() and _AFFIRM.match(norm(text).strip(" .!?")):
            self._utt(text, "confirm", "rules", 0.0, acted=True)
            self._draft_open = False
            await self._begin("confirm", B.BACKCHANNEL, text, t_recv, ack_text="Okay — going ahead…")
            return
        if self._awaiting_slots() and self._slot_fill(text):
            self._utt(text, "addition", "rules", 0.0, acted=True, note="slot_fill")
            self._draft_open = False
            await self._process(B.ADDITION, text, t_recv, note="slot_fill")
            return
        if continuing and self.planner.goal is not None and self.planner.goal.status == "draft":
            d = tier1(text, final=True, context=self._barge_context())
            if d is not None and d.label in (B.BACKCHANNEL, B.HESITATION):
                # A stray "okay" in the middle of a sentence neither ends the request nor
                # interrupts it: the stream is still open.
                self.metrics.ignored[d.label.value] += 1
                self._utt(text, d.label.value, "rules", 0.0, acted=False,
                          note="noise_during_speech")
                self._perception.emit("ignored", epoch=self.epochs.current(),
                                      reason="noise_during_speech", text=text)
                return
            self._utt(text, "new_request", "rules", 0.0, acted=True, note="end_of_stream")
            self._draft_open = False
            self.metrics.labels["new_request"] += 1  # the end of the request being drafted,
            await self._process(None, text, t_recv)  # not an interruption of it
            return
        self._draft_open = False
        # 2. classify. Tier 1 rules; tier 2 (LLM, hard 250 ms) only if there is a live goal that
        #    could be interrupted. When idle, only noise is filtered ("um", "okay", a stray "stop").
        running = self._goal_running()
        res = await self.classifier.classify(text, final=True, context=self._barge_context(),
                                             epoch=self.epochs.current(), allow_llm=running)
        self.metrics.tiers[res.tier] += 1
        if res.tier == 0 and res.reason == "fallback:llm_disabled":
            self._utt(text, "new_request", "rules", res.latency_ms, acted=True)
            self.metrics.labels["new_request"] += 1
            await self._process(None, text, t_recv)  # idle + no noise cue: a fresh request
            return
        self.metrics.labels[res.label.value] += 1
        ignored = res.label in (B.BACKCHANNEL, B.HESITATION)
        self._utt(text, res.label.value, {2: "llm"}.get(res.tier, "rules"), res.latency_ms,
                  acted=not ignored, note=res.reason)
        if ignored:
            self.metrics.ignored[res.label.value] += 1  # traced by the classifier; no ack, no work
            return
        await self._process(res.label, text, t_recv)

    def _goal_running(self) -> bool:
        """Only a goal that is still being worked on can be interrupted. A finished goal can be
        corrected/cancelled (cues still route to it), but an utterance the rules cannot place is
        then a new request, not something to ask the LLM to reinterpret as an interruption."""
        g = self.planner.goal
        if g is None:
            return False
        if g.status == "committed":
            return True
        # A draft is only "running" while speculative work is in flight. A draft that is waiting
        # for the user (blocked, or missing a slot) cannot be interrupted, only continued.
        return g.status == "draft" and bool(self.cancellation.active())

    async def _process(self, label: BargeInType | None, text: str, t_recv: float, *,
                       note: str = "") -> None:
        """`label=None` means a fresh request (no barge-in semantics)."""
        planner = self.planner
        if label is None:
            await self._begin("new", B.GOAL_CHANGE, text, t_recv,
                              ack_text=self._ack_text(None, text, False), tag="new")
            return
        parsed = self.extractor.from_rules(text)
        hint = parsed.intent if parsed else None
        target = planner.has_target(label, hint)
        if label is B.CANCEL and not target:
            self.metrics.ignored["cancel_nothing"] += 1
            self._perception.emit("ignored", epoch=self.epochs.current(), reason="nothing_to_cancel")
            return
        bumped = planner.supersedes(label, hint)
        # Synchronous: cancels every older-epoch task before this loop turn ends.
        epoch = self.epochs.bump(label.value) if bumped else self.epochs.current()
        if label is B.CLARIFICATION_QUESTION and target:
            mode = "question"
        else:
            mode = "bargein" if target else "new"
        # Name the task by what it really is: a request the classifier merely mislabelled (e.g.
        # "different_intent" -> GOAL_CHANGE) with nothing to act on is a *new* request.
        tag = note or ("" if target else "new")
        await self._begin(mode, label, text, t_recv, bumped=bumped, epoch=epoch,
                          ack_text=self._ack_text(label, text, target), note=note, tag=tag)

    async def _begin(self, mode: str, label: BargeInType, text: str, t_recv: float, *,
                     ack_text: str | None, bumped: bool = False, epoch: int | None = None,
                     note: str = "", tag: str = "") -> None:
        epoch = self.epochs.current() if epoch is None else epoch
        tag = tag or label.value
        if ack_text:  # fast path: deterministic, sent (awaited) BEFORE any slow work starts
            self._emit(OutputMessage(type=OutputType.ACK, session_id=self.session_id, epoch=epoch,
                                     status=OutputStatus.PENDING, text=ack_text,
                                     data={"label": tag, "mode": mode}), t_recv=t_recv)
        self.cancellation.spawn(epoch, self._slow(mode, label, text, epoch, bumped),
                                name=f"slow:{mode}:{tag}")

    def _ack_text(self, label: BargeInType | None, text: str, target: bool) -> str | None:
        g = self.planner.goal
        if label is not None and target and g is not None:
            if label is B.CANCEL:
                return self.ack.build(None, B.CANCEL,
                                      has_writes=bool(g.write_keys) or g.status == "committed")
            if label in (B.CORRECTION, B.ADDITION):
                plan = self.planner.plan
                changed = self.extractor.slot_updates(text, g.intent,
                                                      plan.missing if plan else ())
                return self.ack.build(g.intent, label, thaw(self.snapshots.current().slots),
                                      changed=changed)
            if label is B.GOAL_CHANGE:
                parsed = self.extractor.from_rules(text)
                return self.ack.build(parsed.intent if parsed else None, B.GOAL_CHANGE,
                                      parsed.slots if parsed else {})
            if label is B.CLARIFICATION_QUESTION:
                return self.ack.build(None, B.CLARIFICATION_QUESTION)
        parsed = self.extractor.from_rules(text)  # a fresh request
        intent = parsed.intent if parsed else (
            Intent.TROUBLESHOOT if self.planner.is_frame_question(text) else None)
        return self.ack.build(intent, None, parsed.slots if parsed else {})

    # -------------------------------------------------------------------- slow path --------
    async def _slow(self, mode: str, label: BargeInType, text: str, epoch: int,
                    bumped: bool) -> None:
        try:
            if mode == "question":  # read-only: answer from state without waiting for the lock
                out = await self.planner.handle_bargein(label, text, final=True)
                await self._deliver(out)
                return
            async with self._lock:
                if not self.epochs.is_current(epoch):
                    return  # superseded while queued behind another task
                if mode == "new":
                    out = await self.planner.on_final(text)
                elif mode == "confirm":
                    out = await self.planner.confirm()
                else:
                    out = await self.planner.handle_bargein(label, text, final=True,
                                                            bumped=bumped)
                await self._deliver(out)
        except asyncio.CancelledError:
            raise  # superseded: already traced as task_cancelled by the CancellationManager
        except Exception as e:  # noqa: BLE001 - report, never crash the session
            self._slow_trace.emit("slow_error", epoch=self.epochs.current(),
                                  error=f"{type(e).__name__}: {e}")
            self._say(OutputType.ERROR, OutputStatus.FAILED, "Sorry, I hit a problem with that.",
                      epoch=self.epochs.current(), data={"error": f"{type(e).__name__}: {e}"})

    async def _deliver(self, out: PlanOutcome) -> None:
        """Turn a planner outcome into client messages. Every message is epoch-gated."""
        for r in out.compensated:
            if r.ok and not r.data.get("already_compensated"):
                self._say(OutputType.ACTION_RESULT, OutputStatus.CANCELLED,
                          R.compensation_text(r), epoch=out.epoch,
                          data={"tool": r.tool, **r.data})
        main = out.commit if out.commit is not None else out
        ep = main.plan.epoch if main.plan is not None else main.epoch
        a = main.action
        if out.action == "cancelled":
            self._say(OutputType.RESPONSE, OutputStatus.CANCELLED, "Okay, I've stopped that.",
                      epoch=out.epoch)
        elif a == "committed" and main.plan is not None:
            await self._run_plan(main.plan)
        elif a == "needs_slots" and main.plan is not None:
            self._say(OutputType.RESPONSE, OutputStatus.PENDING, R.missing_prompt(main.plan.missing),
                      epoch=ep, data={"missing": list(main.plan.missing)})
        elif a == "needs_confirmation" and main.plan is not None:
            self._say(OutputType.RESPONSE, OutputStatus.PENDING,
                      "Shall I go ahead? Say yes to confirm.", epoch=ep,
                      data={"awaiting_confirmation": main.note})
        elif a == "blocked":
            self._say(OutputType.RESPONSE, OutputStatus.FAILED, R.blocked_text(main.note),
                      epoch=ep, data={"reason": main.note})
        elif a == "diagnosis":
            d = main.data["diagnosis"]
            self._say(OutputType.RESPONSE, OutputStatus.DONE, R.diagnosis_text(d), epoch=ep,
                      data={"diagnosis": d})
        elif a == "smalltalk":
            self._say(OutputType.RESPONSE, OutputStatus.DONE, "Hello! How can I help?", epoch=ep)
        elif a == "busy":
            self._say(OutputType.RESPONSE, OutputStatus.PENDING,
                      "I'm still working on your last request.", epoch=ep)
        elif a == "answer_question":
            plan = self.planner.plan
            g = self.planner.goal
            w = plan.writes[0].args if plan and plan.writes else None
            self._say(OutputType.RESPONSE, OutputStatus.DONE,
                      R.answer_question(g.intent if g else None, g.status if g else "",
                                        thaw(self.snapshots.current().slots), w), epoch=ep)
        elif a == "addition_needs_replan":
            self._say(OutputType.RESPONSE, OutputStatus.PENDING,
                      "That's already in progress. If you want it changed, tell me what to "
                      "change and I'll redo it.", epoch=ep, data={"updates": out.updates})
        elif a == "clarify" or out.note == "no_slot_change_understood":
            self._say(OutputType.RESPONSE, OutputStatus.PENDING,
                      "Sorry, I didn't catch the change. Could you say that again?", epoch=ep)
        # stale / noop / drafted / added-without-commit: nothing to say

    async def _run_plan(self, plan: Plan) -> None:
        write = plan.writes[0] if plan.writes else None
        if write is not None:
            self._say(OutputType.PROGRESS, OutputStatus.COMMITTED, self._progress_text(plan),
                      epoch=plan.epoch, data={"tool": write.tool, "args": write.args,
                                              "inferred": list(plan.inferred)})
        res = await self.planner.execute(plan)
        for wr in res.writes:
            if wr.ok:
                self._say(OutputType.ACTION_RESULT, OutputStatus.DONE, R.write_result_text(wr),
                          epoch=plan.epoch, data={"tool": wr.tool, **wr.data})
            elif wr.data.get("blocked"):
                continue  # duplicate / stale: correctly not executed, nothing to tell the user
            else:
                self._say(OutputType.ERROR, OutputStatus.FAILED,
                          f"Sorry, that didn't work: {wr.error}", epoch=plan.epoch,
                          data={"tool": wr.tool, "error": wr.error})

    @staticmethod
    def _progress_text(plan: Plan) -> str:
        w = plan.writes[0]
        a = w.args
        if w.tool == "book_flight":
            note = " (earliest available)" if "time" in plan.inferred else ""
            return f"Booking the {fmt_time(a['time'])} flight{note}…"
        if w.tool == "set_navigation":
            via = f" via {a['via']}" if a.get("via") else ""
            return f"Starting navigation to {a['destination']}{via}…"
        if w.tool == "reserve_table":
            return f"Reserving a table for {a['party_size']}…"
        if w.tool == "cancel_booking":
            return f"Cancelling booking {a['booking_id']}…"
        return "Working on it…"

    # ------------------------------------------------------------------- introspection -----
    async def wait_idle(self, timeout: float = 10.0) -> None:
        """Wait until nothing is queued, being handled, or running (tests / HTTP `wait=true`)."""
        deadline = time.perf_counter() + timeout

        def idle() -> bool:
            return (self.queue.empty() and not self._handling and not self.eou.timer_active
                    and self._frame_timer is None and not self.cancellation.active())

        while True:
            if idle():
                await asyncio.sleep(0.02)
                if idle():
                    return
            if time.perf_counter() > deadline:
                raise TimeoutError("session did not become idle")
            await asyncio.sleep(0.01)

    async def describe(self) -> dict[str, Any]:
        g, p, snap = self.planner.goal, self.planner.plan, self.snapshots.current()
        lat = self.metrics.ack_latencies_ms
        return {
            "session_id": self.session_id,
            "epoch": self.epochs.current(),
            "goal": {"intent": g.intent.value, "status": g.status,
                     "writes": len(g.write_keys)} if g else None,
            "plan": {"status": p.status.value, "epoch": p.epoch, "missing": list(p.missing),
                     "steps": [{"tool": s.tool, "args": s.args, "epoch": s.epoch,
                                "write": s.is_write} for s in p.steps]} if p else None,
            "slots": thaw(snap.slots),
            "world": {"bookings": await self.world.active_bookings(),
                      "charges": await self.world.charges(),
                      "reservations": await self.world.active_reservations(),
                      "navigation": await self.world.active_navigation(),
                      "invariant_violations": await self.world.check_invariants()},
            "ledger": await self.ledger.all_rows(self.session_id),
            "metrics": {"events": self.metrics.events, "outputs": self.metrics.outputs,
                        "labels": dict(self.metrics.labels), "tiers": dict(self.metrics.tiers),
                        "ignored": dict(self.metrics.ignored),
                        "stale_dropped": self.cancellation.dropped_stale,
                        "stale_emitted": self.metrics.stale_emitted,
                        "ack_ms": {"n": len(lat), "max": round(max(lat), 2) if lat else None}},
            "queue": {"size": self.queue.qsize(), "max_depth": self.queue.metrics.max_depth},
            "outputs": [m.model_dump(mode="json") for m in self.outputs[-50:]],
        }
