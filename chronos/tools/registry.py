"""Tool registry + executor. `ToolExecutor.execute_write` is the ONLY path to a write."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from chronos.coordination.canonical import call_key
from chronos.coordination.epoch import EpochManager
from chronos.coordination.ledger import IdempotencyLedger, LedgerStatus
from chronos.coordination.snapshot import StateSnapshot
from chronos.protocol import PlanStatus, ToolCall, ToolResult
from chronos.tools.world import ToolError, World
from chronos.trace.logger import ComponentTrace

ToolFn = Callable[[World, Any], Awaitable[dict[str, Any]]]


class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"


class BlockReason(str, Enum):
    NOT_COMMITTED = "not_committed"
    STALE_EPOCH = "stale_epoch"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class Compensator:
    """How to undo a write: which tool to call and how to build its args from the original
    write's (args, result)."""
    tool: str
    build_args: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    schema: type[BaseModel]
    fn: ToolFn
    compensator: Compensator | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self, kind: ToolKind | None = None) -> list[str]:
        return sorted(n for n, s in self._tools.items() if kind is None or s.kind == kind)

    def make_call(self, tool: str, args: Mapping[str, Any], epoch: int) -> ToolCall:
        spec = self._tools.get(tool)
        return ToolCall(tool=tool, args=dict(args), epoch=epoch,
                        is_write=bool(spec and spec.kind == ToolKind.WRITE))


REGISTRY = ToolRegistry()  # populated at import time by the decorators below


def read_tool(name: str, schema: type[BaseModel],
              registry: ToolRegistry = REGISTRY) -> Callable[[ToolFn], ToolFn]:
    def deco(fn: ToolFn) -> ToolFn:
        registry.add(ToolSpec(name, ToolKind.READ, schema, fn))
        return fn
    return deco


def write_tool(name: str, schema: type[BaseModel], compensator: Compensator | None = None,
               registry: ToolRegistry = REGISTRY) -> Callable[[ToolFn], ToolFn]:
    def deco(fn: ToolFn) -> ToolFn:
        registry.add(ToolSpec(name, ToolKind.WRITE, schema, fn, compensator))
        return fn
    return deco


class _Fenced(Exception):
    """Epoch went stale between ledger claim and dispatch."""

    ledger_status = LedgerStatus.BLOCKED


class ToolExecutor:
    """Per-session executor: read cache + guarded writes."""

    def __init__(self, session_id: str, world: World, epochs: EpochManager,
                 ledger: IdempotencyLedger, registry: ToolRegistry = REGISTRY,
                 trace: ComponentTrace | None = None) -> None:
        self.session_id, self.world, self.epochs = session_id, world, epochs
        self.ledger, self.registry, self._trace = ledger, registry, trace
        self._read_cache: dict[str, asyncio.Task[ToolResult]] = {}

    # ------------------------------------------------------------------ helpers ------------
    def _emit(self, event: str, epoch: int, **f: Any) -> None:
        if self._trace:
            self._trace.emit(event, epoch=epoch, **f)

    @staticmethod
    def _fail(call: ToolCall, epoch: int, error: str, **data: Any) -> ToolResult:
        return ToolResult(call_id=call.call_id, tool=call.tool, ok=False, error=error,
                          data=data, epoch=epoch)

    def _resolve(self, call: ToolCall, kind: ToolKind) -> tuple[ToolSpec, BaseModel] | ToolResult:
        spec = self.registry.get(call.tool)
        if spec is None:
            return self._fail(call, call.epoch, f"unknown_tool: {call.tool}")
        if spec.kind != kind:
            return self._fail(call, call.epoch, f"not_a_{kind.value}_tool: {call.tool}")
        try:
            return spec, spec.schema.model_validate(call.args)
        except ValidationError as e:
            return self._fail(call, call.epoch, "invalid_args", detail=e.errors(
                include_url=False, include_context=False, include_input=False))

    # -------------------------------------------------------------------- reads ------------
    async def execute_read(self, call: ToolCall) -> ToolResult:
        """Speculative-safe. Results cached per (tool, canonical args); identical concurrent
        reads share one execution. Cache is epoch-agnostic: a read is a pure function of its
        args, so an older epoch's result is reusable. Any committed write clears the cache."""
        resolved = self._resolve(call, ToolKind.READ)
        if isinstance(resolved, ToolResult):
            return resolved
        spec, args = resolved
        key = call_key(call.tool, args.model_dump(mode="json"))
        task = self._read_cache.get(key)
        cached = task is not None
        if task is None:
            task = asyncio.create_task(self._run_read(call, spec, args, key))
            self._read_cache[key] = task
        res = await asyncio.shield(task)  # a cancelled caller must not kill a shared read
        out = res.model_copy(update={"call_id": call.call_id, "epoch": call.epoch,
                                     "cached": cached})
        self._emit("tool_read", call.epoch, tool=call.tool, args=args.model_dump(mode="json"),
                   cached=cached, ok=out.ok, latency_ms=0.0 if cached else res.latency_ms)
        return out

    async def _run_read(self, call: ToolCall, spec: ToolSpec, args: BaseModel,
                        key: str) -> ToolResult:
        t0 = time.perf_counter()
        try:
            await self.world.delay(call.tool)
            data = await spec.fn(self.world, args)
            res = ToolResult(call_id=call.call_id, tool=call.tool, data=data, epoch=call.epoch,
                             latency_ms=(time.perf_counter() - t0) * 1000)
        except ToolError as e:
            res = self._fail(call, call.epoch, str(e))
        if not res.ok:
            self._read_cache.pop(key, None)  # never cache failures
        return res

    def invalidate_reads(self) -> None:
        self._read_cache.clear()

    async def aclose(self) -> None:
        """Cancel reads still in flight (they are shared tasks nobody else owns) so none of them
        touches the world after it has been closed."""
        pending = [t for t in self._read_cache.values() if not t.done()]
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._read_cache.clear()

    # ------------------------------------------------------------------- writes ------------
    async def execute_write(self, call: ToolCall, epoch: int,
                            snapshot: StateSnapshot) -> ToolResult:
        """Write guard. A write dispatches only if
          (a) snapshot.plan status is COMMITTED           -> else blocked `not_committed`
          (b) epoch is still current, checked before the ledger claim AND again immediately
              before dispatch (fencing token)             -> else blocked `stale_epoch`
          (c) the idempotency ledger allows the key       -> else blocked `duplicate`
        If the epoch goes stale *after* the world mutation, the write is compensated at once."""
        resolved = self._resolve(call, ToolKind.WRITE)
        if isinstance(resolved, ToolResult):
            return resolved
        spec, args = resolved
        clean = args.model_dump(mode="json")  # normalised: "6pm" and "18:00" share a key

        if snapshot.plan.get("status") != PlanStatus.COMMITTED.value:
            return self._blocked(call, epoch, BlockReason.NOT_COMMITTED)
        if not self.epochs.is_current(epoch):
            return self._blocked(call, epoch, BlockReason.STALE_EPOCH)

        key = self.ledger.make_key(self.session_id, call.tool, clean)
        t0 = time.perf_counter()
        data: dict[str, Any] = {}
        try:
            async with self.ledger.guard(key, session_id=self.session_id, tool=call.tool,
                                         args=clean, epoch=epoch) as ok:
                if not ok:
                    prior = await self.ledger.result(key)
                    return self._fail(call, epoch, BlockReason.DUPLICATE.value, blocked=True,
                                      reason=BlockReason.DUPLICATE.value,
                                      existing_status=ok.existing.value if ok.existing else None,
                                      existing=prior)
                await self.world.delay(call.tool)  # simulated dispatch latency
                if not self.epochs.is_current(epoch):  # fencing check, right before the effect
                    raise _Fenced
                data = await spec.fn(self.world, args)
                ok.set_result(data)
        except _Fenced:  # ledger marked the claim BLOCKED, so the key stays retryable
            return self._blocked(call, epoch, BlockReason.STALE_EPOCH)
        except ToolError as e:
            return self._fail(call, epoch, str(e))

        self.invalidate_reads()
        res = ToolResult(call_id=call.call_id, tool=call.tool, data=data, epoch=epoch,
                         latency_ms=(time.perf_counter() - t0) * 1000, idempotency_key=key)
        if not self.epochs.is_current(epoch):  # went stale while the effect was landing
            comp = await self.compensate({"key": key, "tool": call.tool, "args": clean,
                                          "epoch": epoch, "result": data})
            return self._fail(call, epoch, "stale_after_commit", blocked=True,
                              reason=BlockReason.STALE_EPOCH.value, compensated=comp.ok,
                              compensation=comp.data)
        return res

    def _blocked(self, call: ToolCall, epoch: int, reason: BlockReason) -> ToolResult:
        self._emit("write_blocked", epoch, tool=call.tool, reason=reason.value,
                   current_epoch=self.epochs.current())
        return self._fail(call, epoch, reason.value, blocked=True, reason=reason.value)

    # ------------------------------------------------------------- compensation ------------
    async def compensate(self, committed_write: Mapping[str, Any]) -> ToolResult:
        """Undo a COMMITTED write via its compensating tool, through the same ledger (so the
        compensation itself is idempotent). `committed_write` is a row from ledger.committed()
        (or the equivalent dict). Bypasses the plan/epoch gates: it exists *because* the epoch
        moved on."""
        orig_tool, epoch = committed_write["tool"], committed_write["epoch"]
        stub = ToolCall(tool=orig_tool, epoch=epoch, is_write=True)
        spec = self.registry.get(orig_tool)
        if spec is None or spec.compensator is None:
            return self._fail(stub, epoch, f"no_compensator: {orig_tool}")
        if not committed_write.get("result"):
            return self._fail(stub, epoch, "no_result_to_compensate")
        comp = spec.compensator
        cspec = self.registry.get(comp.tool)
        if cspec is None:
            return self._fail(stub, epoch, f"unknown_tool: {comp.tool}")
        cargs = cspec.schema.model_validate(
            comp.build_args(committed_write["args"], committed_write["result"]))
        clean = cargs.model_dump(mode="json")
        ckey = self.ledger.make_key(self.session_id, comp.tool, clean)
        call = ToolCall(tool=comp.tool, args=clean, epoch=self.epochs.current(), is_write=True)
        data: dict[str, Any] = {}
        try:
            async with self.ledger.guard(ckey, session_id=self.session_id, tool=comp.tool,
                                         args=clean, epoch=self.epochs.current()) as ok:
                if not ok:  # already compensated / being compensated: idempotent success
                    return ToolResult(call_id=call.call_id, tool=comp.tool, ok=True,
                                      data={"already_compensated": True}, epoch=call.epoch)
                await self.world.delay(comp.tool)
                data = await cspec.fn(self.world, cargs)
                ok.set_result(data)
        except ToolError as e:
            return self._fail(call, call.epoch, str(e))
        await self.ledger.mark_compensated(committed_write["key"])
        self.invalidate_reads()
        self._emit("write_compensated", self.epochs.current(), tool=orig_tool,
                   compensator=comp.tool, original_epoch=epoch)
        return ToolResult(call_id=call.call_id, tool=comp.tool, data=data, epoch=call.epoch)

    async def compensate_stale(self, keep_keys: frozenset[str] = frozenset()) -> list[ToolResult]:
        """Compensate every committed write from an older epoch that the new plan doesn't keep."""
        out = []
        for w in await self.ledger.committed(self.session_id):
            spec = self.registry.get(w["tool"])
            if w["epoch"] < self.epochs.current() and w["key"] not in keep_keys \
                    and spec and spec.compensator:
                out.append(await self.compensate(w))
        return out

    async def compensate_goal(self, keys: Iterable[str]) -> list[ToolResult]:
        """Compensate exactly the given committed writes (one goal's writes), leaving every
        other write in the session untouched."""
        wanted = set(keys)
        if not wanted:
            return []
        rows = [w for w in await self.ledger.committed(self.session_id) if w["key"] in wanted]
        return [await self.compensate(w) for w in rows]
