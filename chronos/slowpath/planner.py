"""Speculative planner.

Reads can be speculative; writes must be committed.

* Partial transcripts -> a DRAFT plan; read-only tool calls start immediately.
* A plan is COMMITTED only on a final transcript with all required slots (or after an explicit
  `confirm()` for tools listed in `PlannerPolicy.confirm_writes`). Only COMMITTED plans can reach
  a write, and every write still passes the executor's epoch fence + idempotency ledger.
* CORRECTION patches the previous snapshot (reads whose slot dependencies still hold are reused,
  only affected reads are re-issued); GOAL_CHANGE starts a new goal; CANCEL abandons the goal.
  Any committed writes of the superseded goal are undone through the ledger (compensation).

Plans are assembled from intent + slots with deterministic templates: a 3B model is unreliable
at emitting tool graphs, and the rules make behaviour testable. The LLM is used for intent
fallback (perception) and for writing the grounded troubleshooting diagnosis.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chronos.coordination.cancellation import CancellationManager
from chronos.coordination.canonical import call_key
from chronos.coordination.epoch import EpochManager
from chronos.coordination.snapshot import ReadEntry, SnapshotStore, thaw
from chronos.perception.intent import Intent, IntentExtractor
from chronos.protocol import BargeInType, PlanStatus, ToolCall, ToolResult
from chronos.slowpath.diagnose import diagnose_with_llm
from chronos.slowpath.diagnosis import Diagnosis
from chronos.slowpath.llm import LLM, LLMError
from chronos.slowpath.vision import Frame, FrameStore, Vision
from chronos.tools.registry import ToolExecutor
from chronos.trace.logger import ComponentTrace

B = BargeInType

# Slots a plan needs before it may be committed.
_COMMIT_REQUIRED: dict[Intent, tuple[str, ...]] = {
    Intent.BOOK_FLIGHT: ("dest", "date"),
    Intent.NAVIGATE: ("destination",),
    Intent.RESERVE_TABLE: ("party_size",),
    Intent.CANCEL_BOOKING: ("booking_id",),
    Intent.CHANGE_BOOKING: ("booking_id",),
}
_TERMINAL = ("done", "cancelled", "failed")


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal_id: str
    epoch: int
    intent: Intent
    slots: dict[str, Any] = Field(default_factory=dict)
    steps: tuple[ToolCall, ...] = ()
    status: PlanStatus = PlanStatus.DRAFT
    missing: tuple[str, ...] = ()
    inferred: tuple[str, ...] = ()  # slots the agent filled in itself (e.g. flight time)

    @property
    def reads(self) -> tuple[ToolCall, ...]:
        return tuple(s for s in self.steps if not s.is_write)

    @property
    def writes(self) -> tuple[ToolCall, ...]:
        return tuple(s for s in self.steps if s.is_write)


@dataclass
class Goal:
    goal_id: str
    intent: Intent
    status: str = "draft"  # draft | committed | done | cancelled | failed
    write_keys: list[str] = field(default_factory=list)  # committed writes belonging to this goal
    done_at: float | None = None  # monotonic time the goal finished (for the undo window)

    def finish(self) -> None:
        self.status, self.done_at = "done", time.perf_counter()


@dataclass(frozen=True)
class PlannerPolicy:
    confirm_writes: frozenset[str] = frozenset()  # tools that need an explicit confirm()
    # A finished goal can still be corrected/cancelled for this long ("make it 6pm", "cancel
    # that" right after hearing the result). Later, a stray "stop" must not undo old work.
    undo_window_s: float = 60.0


@dataclass
class PlanOutcome:
    action: str
    plan: Plan | None = None
    epoch: int = 0
    note: str = ""
    compensated: list[ToolResult] = field(default_factory=list)
    reads_reused: int = 0
    reads_started: int = 0
    replanned: list[str] = field(default_factory=list)  # tools re-issued for this change
    updates: dict[str, Any] = field(default_factory=dict)  # slots changed by the barge-in
    data: dict[str, Any] = field(default_factory=dict)
    commit: PlanOutcome | None = None  # commit attempt made as part of this barge-in

    @property
    def committed(self) -> bool:
        return self.action == "committed" or bool(self.commit and self.commit.committed)


@dataclass
class ExecutionResult:
    writes: list[ToolResult] = field(default_factory=list)
    ok: bool = False
    stale: bool = False
    note: str = ""


class SpeculativePlanner:
    def __init__(self, session_id: str, *, epochs: EpochManager, cancellation: CancellationManager,
                 snapshots: SnapshotStore, executor: ToolExecutor, extractor: IntentExtractor,
                 llm: LLM | None = None, vision: Vision | None = None,
                 frames: FrameStore | None = None, trace: ComponentTrace | None = None,
                 policy: PlannerPolicy | None = None, llm_timeout_s: float = 6.0,
                 vision_timeout_s: float = 20.0) -> None:
        self.session_id = session_id
        self.epochs, self.cancellation, self.snapshots = epochs, cancellation, snapshots
        self.executor, self.extractor, self.llm = executor, extractor, llm
        self.vision, self.frames = vision, frames or FrameStore()
        self._trace, self.policy = trace, policy or PlannerPolicy()
        self.llm_timeout_s, self.vision_timeout_s = llm_timeout_s, vision_timeout_s
        self.goal: Goal | None = None
        self.plan: Plan | None = None
        self._confirmed = False
        self._warm: dict[Any, asyncio.Task[Any]] = {}  # speculative tasks of the current epoch
        self._desc_tasks: dict[str, asyncio.Task[str | None]] = {}

    # ------------------------------------------------------------------ helpers -------------
    def _t(self, event: str, epoch: int | None = None, **fields: Any) -> None:
        if self._trace:
            self._trace.emit(event, epoch=self.epochs.current() if epoch is None else epoch,
                             **fields)

    def _slots(self) -> dict[str, Any]:
        return thaw(self.snapshots.current().slots)

    def _norm(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
        spec = self.executor.registry.get(tool)
        if spec is None:
            return None
        try:
            return spec.schema.model_validate(dict(args)).model_dump(mode="json")
        except ValidationError:
            return None

    def _call(self, tool: str, args: Mapping[str, Any], epoch: int) -> ToolCall | None:
        nargs = self._norm(tool, args)
        return None if nargs is None else self.executor.registry.make_call(tool, nargs, epoch)

    @staticmethod
    def _deps(call: ToolCall, slots: Mapping[str, Any]) -> dict[str, Any]:
        """Slot values a read depends on: only the args actually specified (schema defaults
        normalise to None and must not count, or an unrelated slot change would invalidate it)."""
        return {k: slots[k] for k, v in call.args.items() if v is not None and k in slots}

    @staticmethod
    def _pick(slots: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: slots[k] for k in keys if k in slots}

    def _new_goal(self, intent: Intent) -> Goal:
        self.goal = Goal(uuid.uuid4().hex, intent)
        self._confirmed = False
        self._warm.clear()
        return self.goal

    def _apply(self, goal: Goal, intent: Intent, *, slots: Mapping[str, Any] | None = None,
               replace: bool = False, status: PlanStatus = PlanStatus.DRAFT) -> None:
        """Write the goal's state into the snapshot store (incremental unless replace)."""
        diff: dict[str, Any] = dict(slots or {})
        if replace:
            diff.update({k: None for k in self.snapshots.current().slots if k not in diff})
        self.snapshots.apply(
            {"intent": intent.value, "slots": diff,
             "plan": {"status": status.value, "goal_id": goal.goal_id, "intent": intent.value}},
            epoch=self.epochs.current())

    def _set_plan_status(self, status: PlanStatus) -> None:
        goal = self.goal
        self.snapshots.apply({"plan": {"status": status.value,
                                       "goal_id": goal.goal_id if goal else None,
                                       "intent": goal.intent.value if goal else None}},
                             epoch=self.epochs.current())

    # -------------------------------------------------------------- plan building ----------
    def _reads_for(self, intent: Intent, slots: Mapping[str, Any]) -> list[tuple[str, dict]]:
        p = self._pick
        if intent is Intent.BOOK_FLIGHT and "dest" in slots:
            # `time` is deliberately NOT an argument: the planner picks the departure from the
            # results, so a "make it 6pm" correction reuses this read instead of repeating it.
            return [("search_flights", p(slots, ("origin", "dest", "date")))]
        if intent is Intent.NAVIGATE and "destination" in slots:
            return [("get_route", p(slots, ("destination", "via")))]
        if intent is Intent.RESERVE_TABLE and "party_size" in slots:
            return [("check_table_availability",
                     p(slots, ("party_size", "date", "time", "restaurant")))]
        if intent in (Intent.CANCEL_BOOKING, Intent.CHANGE_BOOKING) and "booking_id" in slots:
            return [("get_booking", {"booking_id": slots["booking_id"]})]
        return []

    def _make_plan(self, goal: Goal, intent: Intent, slots: Mapping[str, Any],
                   epoch: int) -> Plan:
        calls = [c for tool, args in self._reads_for(intent, slots)
                 if (c := self._call(tool, args, epoch)) is not None]
        missing = tuple(k for k in _COMMIT_REQUIRED.get(intent, ()) if k not in slots)
        return Plan(goal_id=goal.goal_id, epoch=epoch, intent=intent, slots=dict(slots),
                    steps=tuple(calls), status=PlanStatus.DRAFT, missing=missing)

    def _write_for(self, plan: Plan, results: list[ToolResult]
                   ) -> tuple[ToolCall | None, str, tuple[str, ...]]:
        """Build the write step from the (now known) read results. -> (call, note, inferred)."""
        s, e, intent = plan.slots, plan.epoch, plan.intent
        if intent is Intent.BOOK_FLIGHT:
            r = results[0]
            if not r.ok:
                return None, "search_failed", ()
            flights = list(r.data.get("flights", []))
            wanted = s.get("time")
            if wanted:
                flights = [f for f in flights if f["time"] == wanted]
            if not flights:
                return None, "no_flight_at_time" if wanted else "no_flights", ()
            inferred = () if wanted else ("time",)
            args = {**self._pick(s, ("origin", "dest", "date", "seat_pref", "passenger")),
                    "time": wanted or flights[0]["time"]}
            return self._call("book_flight", args, e), "", inferred
        if intent is Intent.NAVIGATE:
            if not results[0].ok:
                return None, "route_failed", ()
            return self._call("set_navigation", self._pick(s, ("destination", "via")), e), "", ()
        if intent is Intent.RESERVE_TABLE:
            r = results[0]
            if not r.ok or not r.data.get("count"):
                return None, "no_table", ()
            args = self._pick(s, ("party_size", "date", "time", "restaurant"))
            return self._call("reserve_table", args, e), "", ()
        if intent is Intent.CANCEL_BOOKING:
            r = results[0]
            if not r.ok:
                return None, "booking_not_found", ()
            if r.data.get("status") == "CANCELLED":
                return None, "already_cancelled", ()
            return self._call("cancel_booking", {"booking_id": s["booking_id"]}, e), "", ()
        if intent is Intent.CHANGE_BOOKING:
            return None, "change_booking_not_supported", ()
        return None, "no_write_for_intent", ()

    # --------------------------------------------------------- speculative reads -----------
    def _launch_reads(self, plan: Plan) -> tuple[int, list[str]]:
        """Start every read the plan needs that is neither cached nor already running.
        -> (reads reused from the snapshot, tools started)."""
        reused, started = 0, []
        snap = self.snapshots.current()
        for call in plan.reads:
            if snap.cached(call.tool, call.args) is not None:
                reused += 1
                self._t("read_reused", tool=call.tool, args=call.args, source="snapshot")
                continue
            key = call_key(call.tool, call.args)
            task = self._warm.get(key)
            if task is not None and not task.done():
                continue
            deps = self._deps(call, snap.slots)
            self._warm[key] = self.cancellation.spawn(
                plan.epoch, self._warm_read(call, deps), name=f"read:{call.tool}")
            started.append(call.tool)
        return reused, started

    async def _warm_read(self, call: ToolCall, deps: Mapping[str, Any]) -> None:
        res = await self.executor.execute_read(call)
        self._record(call, res, deps)

    def _record(self, call: ToolCall, res: ToolResult, deps: Mapping[str, Any]) -> None:
        """Remember a finished read in the snapshot cache if it is still valid for the CURRENT
        slots, even when it finished in an older epoch (a read is a pure function of its args).
        Otherwise it is obsolete: drop it and trace it."""
        if not res.ok:
            return
        entry = ReadEntry(call.tool, call.args, res.data, deps)
        snap = self.snapshots.current()
        if entry.valid_for(snap.slots):
            if not self.epochs.is_current(call.epoch):
                self._t("read_carried_over", tool=call.tool, from_epoch=call.epoch,
                        to_epoch=self.epochs.current())
            self.snapshots.apply({"cache_add": [entry]}, epoch=snap.epoch)
        elif not self.epochs.is_current(call.epoch):
            self.cancellation.emit_if_current(call.epoch, res, what=f"read:{call.tool}")

    async def _read(self, call: ToolCall) -> ToolResult:
        entry = self.snapshots.current().cached(call.tool, call.args)
        if entry is not None:
            return ToolResult(call_id=call.call_id, tool=call.tool, data=thaw(entry.result),
                              epoch=call.epoch, cached=True)
        res = await self.executor.execute_read(call)
        snap = self.snapshots.current()
        self._record(call, res, self._deps(call, snap.slots))
        return res

    # -------------------------------------------------------------------- vision ----------
    def on_camera_frame(self, payload: Mapping[str, Any] | Frame) -> Frame:
        frame = payload if isinstance(payload, Frame) else Frame.from_payload(dict(payload))
        self.frames.put(frame)
        self._t("frame_received", frame=frame.name, bytes=len(frame.data))
        g = self.goal  # the frame may arrive after the words: start describing it right away
        if g is not None and g.intent is Intent.TROUBLESHOOT and g.status == "draft":
            self._speculate_frame(self.epochs.current(), self._slots())
        return frame

    def _speculate_frame(self, epoch: int, slots: Mapping[str, Any]) -> bool:
        """Start describing the latest frame + the KB lookup for a troubleshooting draft."""
        frame = self.frames.latest()
        if frame is None or "symptom" not in slots or (
                self.plan is not None and self.plan.intent is not Intent.TROUBLESHOOT):
            return False
        key = ("kb", slots["symptom"], frame.sha)
        if key in self._warm:
            return False
        self._warm[key] = self.cancellation.spawn(
            epoch, self._warm_troubleshoot(str(slots["symptom"]), epoch), name="kb+vision")
        return True

    async def _describe_frame(self) -> str | None:
        frame = self.frames.latest()
        if frame is None or self.vision is None:
            return None
        task = self._desc_tasks.get(frame.sha)
        if task is None:  # shared + shielded: a cancelled epoch must not kill a slow model call
            task = asyncio.create_task(self._describe_once(frame))
            self._desc_tasks[frame.sha] = task
        return await asyncio.shield(task)

    async def _describe_once(self, frame: Frame) -> str | None:
        assert self.vision is not None
        try:
            desc = await self.vision.describe(frame, timeout=self.vision_timeout_s)
        except (LLMError, TimeoutError):
            self._desc_tasks.pop(frame.sha, None)  # allow a later retry
            return None
        self._t("frame_described", frame=frame.name, description=desc)
        return desc

    def _kb_call(self, question: str, desc: str | None, epoch: int) -> ToolCall | None:
        return self._call("lookup_troubleshooting_kb",
                          {"query": question, "observations": desc or ""}, epoch)

    async def _warm_troubleshoot(self, question: str, epoch: int) -> None:
        desc = await self._describe_frame()
        call = self._kb_call(question, desc, epoch)
        if call is not None:
            res = await self.executor.execute_read(call)
            self._record(call, res, {"symptom": question})

    async def _llm_diagnose(self, question: str, desc: str | None,
                            matches: list[dict[str, Any]]) -> Diagnosis:
        return await diagnose_with_llm(self.llm, question, desc, matches, self.llm_timeout_s)

    async def _diagnose(self, plan: Plan) -> PlanOutcome:
        question = str(plan.slots.get("symptom") or "")
        desc = await self._describe_frame()
        call = self._kb_call(question, desc, plan.epoch)
        res = await self._read(call) if call else None
        matches = list(res.data.get("matches", [])) if res and res.ok else []
        diag = await self._llm_diagnose(question, desc, matches)
        if not self.epochs.is_current(plan.epoch):
            self.cancellation.emit_if_current(plan.epoch, diag, what="diagnosis")
            return PlanOutcome("stale", plan, self.epochs.current(), note="epoch_changed")
        done = plan.model_copy(update={"status": PlanStatus.DONE,
                                       "steps": plan.steps + ((call,) if call else ())})
        self.plan = done
        if self.goal:
            self.goal.finish()
        self._set_plan_status(PlanStatus.DONE)
        self._t("diagnosis_ready", grounded=diag.grounded, source=diag.source,
                kb_ids=diag.kb_ids)
        return PlanOutcome("diagnosis", done, plan.epoch, data={"diagnosis": diag.model_dump()})

    # ------------------------------------------------------------------- drafting ---------
    @staticmethod
    def _canon(intent: Intent, slots: dict[str, Any]) -> tuple[Intent, dict[str, Any]]:
        if intent is Intent.ADD_STOP:  # "X first" with no active route = a route with a via
            return Intent.NAVIGATE, {"via": slots["stop"]} if "stop" in slots else {}
        return intent, slots

    def _draft(self, intent: Intent, slots: dict[str, Any]) -> tuple[int, list[str]] | None:
        """Create/refresh the DRAFT goal+plan and launch speculative reads.
        -> (reads reused, tools started), or None if a committed goal is running."""
        intent, slots = self._canon(intent, slots)
        goal = self.goal
        replace = False
        if goal is not None and goal.status == "committed":
            return None  # never replace a running goal: its writes must stay tracked
        if goal is None or goal.status in _TERMINAL or goal.intent is not intent:
            goal, replace = self._new_goal(intent), True
        self._apply(goal, intent, slots=slots, replace=replace)
        epoch, all_slots = self.epochs.current(), self._slots()
        plan = self._make_plan(goal, intent, all_slots, epoch)
        self.plan = plan
        reused, started = self._launch_reads(plan)
        if self._speculate_frame(epoch, all_slots):
            started.append("describe_frame")
        self._t("plan_drafted", plan_id=plan.plan_id, intent=intent.value,
                steps=[c.tool for c in plan.steps], missing=list(plan.missing),
                reused=reused, started=started)
        return reused, started

    async def on_partial(self, text: str) -> Plan | None:
        """Streaming partial transcript: rules only (no LLM, no awaiting), then speculate."""
        parsed = self.extractor.from_rules(text)
        if parsed is not None:
            self._draft(parsed.intent, dict(parsed.slots))
        return self.plan

    FRAME_QUESTION_WINDOW_S = 120.0  # a picture stays "the thing we are talking about" this long

    def is_frame_question(self, text: str) -> bool:
        """Words nothing else claimed, said about a camera frame that just arrived, are a
        question about that picture ("why is it not powering on?"), not small talk."""
        age = self.frames.age_s()
        if age is None or age > self.FRAME_QUESTION_WINDOW_S:
            return False
        words = text.split()
        return len(words) >= 2 or text.strip().endswith("?")

    async def on_final(self, text: str) -> PlanOutcome:
        """A complete utterance that starts (or completes) a goal."""
        res = await self.extractor.extract(text, epoch=self.epochs.current())
        intent, slots = res.intent, dict(res.slots)
        if (intent is Intent.SMALLTALK and res.source != "rules"  # nobody understood it...
                and self.is_frame_question(text)):  # ...but there is a picture to ask about
            intent, slots = Intent.TROUBLESHOOT, {"symptom": text.strip()}
            self._t("frame_question", text=text)
        drafted = self._draft(intent, slots)
        if drafted is None:
            return PlanOutcome("busy", self.plan, self.epochs.current(),
                               note="a committed goal is running; use handle_bargein")
        out = await self._try_commit()
        out.reads_reused += drafted[0]
        out.reads_started += len(drafted[1])
        return out

    # ------------------------------------------------------------------ committing --------
    async def _try_commit(self) -> PlanOutcome:
        plan, goal = self.plan, self.goal
        epoch = self.epochs.current()
        if plan is None or goal is None:
            return PlanOutcome("noop", None, epoch)
        if plan.missing:
            return PlanOutcome("needs_slots", plan, epoch, note=",".join(plan.missing))
        if plan.intent is Intent.SMALLTALK:
            goal.finish()
            return PlanOutcome("smalltalk", plan, epoch)
        if plan.intent is Intent.TROUBLESHOOT:
            return await self._diagnose(plan)
        results = list(await asyncio.gather(*(self._read(c) for c in plan.reads)))
        if not self.epochs.is_current(plan.epoch):  # an interruption landed while we waited
            self.cancellation.emit_if_current(plan.epoch, plan, what="plan_commit")
            return PlanOutcome("stale", plan, self.epochs.current(), note="epoch_changed")
        write, note, inferred = self._write_for(plan, results)
        if write is None:
            return PlanOutcome("blocked", plan, epoch, note=note)
        full = plan.model_copy(update={"steps": plan.reads + (write,), "inferred": inferred})
        if write.tool in self.policy.confirm_writes and not self._confirmed:
            self.plan = full
            return PlanOutcome("needs_confirmation", full, epoch, note=write.tool)
        committed = full.model_copy(update={"status": PlanStatus.COMMITTED})
        self.plan, goal.status = committed, "committed"
        self._set_plan_status(PlanStatus.COMMITTED)
        self._t("plan_committed", plan_id=committed.plan_id,
                steps=[c.tool for c in committed.steps], inferred=list(inferred))
        return PlanOutcome("committed", committed, epoch)

    async def confirm(self) -> PlanOutcome:
        """Explicit user confirmation for a plan that was waiting on it."""
        self._confirmed = True
        return await self._try_commit()

    # ------------------------------------------------------------------ execution ---------
    def _on_write_done(self, goal: Goal, task: asyncio.Task[ToolResult]) -> None:
        if task.cancelled() or task.exception() is not None:
            return
        res = task.result()
        if res.ok and res.idempotency_key:
            goal.write_keys.append(res.idempotency_key)

    async def execute(self, plan: Plan) -> ExecutionResult:
        """Run a COMMITTED plan's writes. Each write runs as a *protected* task so an epoch bump
        can never cancel it half-way; the executor's fence + compensation decide its fate."""
        goal = self.goal
        if plan.status is not PlanStatus.COMMITTED or goal is None:
            return ExecutionResult(note="not_committed")
        if not self.epochs.is_current(plan.epoch):
            self.cancellation.emit_if_current(plan.epoch, plan, what="plan_execute")
            return ExecutionResult(stale=True, note="stale_epoch")
        snap = self.snapshots.current()
        results: list[ToolResult] = []
        for call in plan.writes:
            task = self.cancellation.spawn(
                plan.epoch, self.executor.execute_write(call, plan.epoch, snap),
                name=f"write:{call.tool}", protected=True)
            task.add_done_callback(partial(self._on_write_done, goal))
            res = await asyncio.shield(task)
            results.append(res)
            if not res.ok:
                break
        stale = any(r.data.get("reason") == "stale_epoch" or r.error == "stale_after_commit"
                    for r in results)
        ok = all(r.ok for r in results)
        if goal is self.goal and self.epochs.is_current(plan.epoch):
            if ok:
                goal.finish()
                self.plan = plan.model_copy(update={"status": PlanStatus.DONE})
                self._set_plan_status(PlanStatus.DONE)
            elif not any(r.data.get("blocked") for r in results):
                goal.status = "failed"
                self.plan = plan.model_copy(update={"status": PlanStatus.FAILED})
                self._set_plan_status(PlanStatus.FAILED)
        return ExecutionResult(results, ok=ok, stale=stale)

    # ------------------------------------------------------------------- barge-in ---------
    async def _flush_write_callbacks(self) -> None:
        await asyncio.sleep(0)  # let done-callbacks of just-finished writes record their keys

    async def _undo_goal(self, goal: Goal) -> list[ToolResult]:
        await self._flush_write_callbacks()
        keys, goal.write_keys = list(goal.write_keys), []
        if not keys:
            return []
        # Protected + shielded: a later epoch bump must never cancel a compensation half-way.
        task = self.cancellation.spawn(self.epochs.current(),
                                       self.executor.compensate_goal(keys),
                                       name="compensate", protected=True)
        return await asyncio.shield(task)

    def within_undo_window(self, goal: Goal) -> bool:
        return goal.done_at is not None and (
            time.perf_counter() - goal.done_at <= self.policy.undo_window_s)

    def has_target(self, label: BargeInType, new_intent: Intent | None = None) -> bool:
        """Is there a goal this barge-in acts on? If not, the utterance is a new request.
        `new_intent` is what the rules read from the utterance (if anything)."""
        g = self.goal
        if g is None or g.status in ("cancelled", "failed") or label in (
                B.BACKCHANNEL, B.HESITATION):
            return False
        if label is B.CLARIFICATION_QUESTION:
            # A question is about a goal that is still being worked on. Once it is finished, the
            # same words ("What's wrong with this machine?") are far more likely a new request.
            return g.status in ("draft", "committed")
        if g.status == "done":
            if label is B.GOAL_CHANGE:
                # A different objective after completion is a new request ("reserve a table"
                # after a flight). But "gas station first" after the route was set changes THAT
                # route: it is not a new task, and the destination must carry over.
                return (new_intent is Intent.ADD_STOP and g.intent is Intent.NAVIGATE
                        and self.within_undo_window(g))
            return self.within_undo_window(g)
        return True

    def supersedes(self, label: BargeInType, new_intent: Intent | None = None) -> bool:
        """Should this barge-in bump the epoch? (Only if it changes/cancels a live goal.)"""
        return label.bumps_epoch and self.has_target(label, new_intent)

    def _bump(self, reason: str, already: bool) -> int:
        return self.epochs.current() if already else self.epochs.bump(reason)

    async def handle_bargein(self, label: BargeInType, text: str, *,
                             final: bool = True, bumped: bool = False) -> PlanOutcome:
        """Apply a classified interruption. BACKCHANNEL / HESITATION never touch state or the
        epoch. CORRECTION / GOAL_CHANGE / CANCEL bump the epoch (cancelling older work) when
        there is a goal to supersede. `bumped=True` means the caller (the session) already
        bumped the epoch synchronously, so cancellation was not delayed by this coroutine."""
        epoch = self.epochs.current()
        if label in (B.BACKCHANNEL, B.HESITATION):
            return PlanOutcome("noop", self.plan, epoch, note=label.value)
        if label is B.CLARIFICATION_QUESTION:
            return PlanOutcome("answer_question", self.plan, epoch, note=text)
        goal = self.goal
        parsed = self.extractor.from_rules(text)
        if goal is None or not self.has_target(label, parsed.intent if parsed else None):
            if label is B.CANCEL:
                return PlanOutcome("noop", self.plan, epoch, note="nothing_to_cancel")
            # nothing to supersede: this is simply the start of a request (no epoch bump)
            if final:
                return await self.on_final(text)
            await self.on_partial(text)
            return PlanOutcome("drafted", self.plan, epoch)
        if label is B.ADDITION:
            return await self._addition(goal, text, final)
        if label is B.CANCEL:
            return await self._cancel(goal, bumped)
        if label is B.CORRECTION:
            return await self._correction(goal, text, final, bumped)
        return await self._goal_change(goal, text, final, bumped)

    async def _replan(self, goal: Goal, intent: Intent, epoch: int
                      ) -> tuple[Plan, int, list[str]]:
        self._warm.clear()
        plan = self._make_plan(goal, intent, self._slots(), epoch)
        self.plan = plan
        reused, started = self._launch_reads(plan)
        return plan, reused, started

    async def _correction(self, goal: Goal, text: str, final: bool,
                          bumped: bool = False) -> PlanOutcome:
        epoch = self._bump("correction", bumped)  # cancels every older-epoch task, synchronously
        undo = asyncio.create_task(self._undo_goal(goal))
        updates = self.extractor.slot_updates(text, goal.intent)
        goal.status, self._confirmed = "draft", False
        self._apply(goal, goal.intent, slots=updates)
        plan, reused, started = await self._replan(goal, goal.intent, epoch)
        self._t("plan_patched", plan_id=plan.plan_id, updates=updates, reused=reused,
                replanned=started)
        out = PlanOutcome("corrected", plan, epoch, reads_reused=reused, reads_started=len(started),
                          replanned=started, updates=updates,
                          note="" if updates else "no_slot_change_understood")
        if final and updates:
            out.commit = await self._try_commit()
            out.plan = self.plan
        out.compensated = await undo
        return out

    async def _goal_change(self, goal: Goal, text: str, final: bool,
                           bumped: bool = False) -> PlanOutcome:
        epoch = self._bump("goal_change", bumped)
        old_slots = self._slots()
        ext = await self.extractor.extract(text, epoch=epoch)
        if (ext.source == "fallback" and ext.confidence < 0.3) or ext.intent is Intent.SMALLTALK:
            # We could not tell what the new goal is: keep the old one (re-stamped to this epoch).
            plan, reused, started = await self._replan(goal, goal.intent, epoch)
            return PlanOutcome("clarify", plan, epoch, note="new_goal_not_understood",
                               reads_reused=reused, reads_started=len(started), replanned=started)
        intent, slots = self._merge_goal_change(goal.intent, old_slots, ext.intent, ext.slots)
        undo = asyncio.create_task(self._undo_goal(goal))
        goal.status = "cancelled"
        new_goal = self._new_goal(intent)
        self._apply(new_goal, intent, slots=slots, replace=True)
        plan, reused, started = await self._replan(new_goal, intent, epoch)
        self._t("plan_drafted", plan_id=plan.plan_id, intent=intent.value, reason="goal_change",
                steps=[c.tool for c in plan.steps], missing=list(plan.missing), reused=reused,
                started=started)
        out = PlanOutcome("goal_changed", plan, epoch, reads_reused=reused,
                          reads_started=len(started), replanned=started, updates=slots)
        if final:
            out.commit = await self._try_commit()
            out.plan = self.plan
        out.compensated = await undo
        return out

    @staticmethod
    def _merge_goal_change(old_intent: Intent, old_slots: Mapping[str, Any], new_intent: Intent,
                           new_slots: Mapping[str, Any]) -> tuple[Intent, dict[str, Any]]:
        if new_intent is Intent.ADD_STOP:
            slots: dict[str, Any] = {"via": new_slots["stop"]} if "stop" in new_slots else {}
            if old_intent is Intent.NAVIGATE and "destination" in old_slots:
                slots["destination"] = old_slots["destination"]  # keep going where we were going
            return Intent.NAVIGATE, slots
        return new_intent, dict(new_slots)

    async def _cancel(self, goal: Goal, bumped: bool = False) -> PlanOutcome:
        epoch = self._bump("cancel", bumped)
        compensated = await self._undo_goal(goal)
        goal.status = "cancelled"
        if self.plan is not None:
            self.plan = self.plan.model_copy(update={"status": PlanStatus.CANCELLED})
        self._set_plan_status(PlanStatus.CANCELLED)
        self._warm.clear()
        self._t("plan_cancelled", goal_id=goal.goal_id, compensated=len(compensated))
        return PlanOutcome("cancelled", self.plan, epoch, compensated=compensated)

    async def _addition(self, goal: Goal, text: str, final: bool) -> PlanOutcome:
        epoch = self.epochs.current()
        awaiting = self.plan.missing if self.plan is not None else ()
        updates = self.extractor.slot_updates(text, goal.intent, awaiting)
        if goal.status in ("committed", "done") or goal.write_keys:
            # The write already exists; a different write would be a second booking. Do not
            # touch it: report so the session can offer to change it (a correction).
            return PlanOutcome("addition_needs_replan", self.plan, epoch, updates=updates,
                               note="goal already committed")
        if not updates:
            return PlanOutcome("added", self.plan, epoch, note="nothing_understood")
        self._apply(goal, goal.intent, slots=updates)  # same epoch: no bump for an addition
        plan, reused, started = await self._replan(goal, goal.intent, epoch)
        self._t("plan_patched", plan_id=plan.plan_id, updates=updates, reused=reused,
                replanned=started)
        out = PlanOutcome("added", plan, epoch, reads_reused=reused, reads_started=len(started),
                          replanned=started, updates=updates)
        if final:
            out.commit = await self._try_commit()
            out.plan = self.plan
        return out

    async def aclose(self) -> None:
        for t in list(self._desc_tasks.values()):
            t.cancel()
        self._desc_tasks.clear()
