"""Idempotency ledger (SQLite via aiosqlite). A write runs only if its key isn't live."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import aiosqlite

from chronos.coordination.canonical import canonical_json, idempotency_key
from chronos.trace.logger import ComponentTrace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    key TEXT PRIMARY KEY, session_id TEXT NOT NULL, tool TEXT NOT NULL, args_json TEXT NOT NULL,
    status TEXT NOT NULL, epoch INTEGER NOT NULL, result_json TEXT,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
)"""


class LedgerStatus(str, Enum):
    IN_FLIGHT = "IN_FLIGHT"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"  # fenced: the epoch moved on before the effect, nothing was written
    COMPENSATED = "COMPENSATED"


# IN_FLIGHT / COMMITTED block a new execution; FAILED / BLOCKED / COMPENSATED allow a retry.


class GuardTicket:
    """Yielded by `ledger.guard`. Truthy iff the caller may execute the write."""

    def __init__(self, key: str, allowed: bool, existing: LedgerStatus | None) -> None:
        self.key, self.allowed, self.existing = key, allowed, existing
        self.result: Mapping[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.allowed

    def set_result(self, result: Mapping[str, Any]) -> None:
        self.result = result


class IdempotencyLedger:
    def __init__(self, db: aiosqlite.Connection, trace: ComponentTrace | None = None) -> None:
        self._db = db
        self._trace = trace
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: str = ":memory:",
                   trace: ComponentTrace | None = None) -> IdempotencyLedger:
        db = await aiosqlite.connect(path)
        await db.execute(_SCHEMA)
        await db.commit()
        return cls(db, trace)

    async def close(self) -> None:
        await self._db.close()

    @staticmethod
    def make_key(session_id: str, tool: str, args: Mapping[str, Any]) -> str:
        return idempotency_key(session_id, tool, args)

    async def status(self, key: str) -> LedgerStatus | None:
        async with self._db.execute("SELECT status FROM ledger WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return LedgerStatus(row[0]) if row else None

    async def result(self, key: str) -> dict[str, Any] | None:
        async with self._db.execute("SELECT result_json FROM ledger WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return json.loads(row[0]) if row and row[0] else None

    async def _try_claim(self, key: str, session_id: str, tool: str, args: Mapping[str, Any],
                         epoch: int) -> tuple[bool, LedgerStatus | None]:
        """Atomic check-and-set. The asyncio.Lock serialises tasks in this process; the
        conditional upsert keeps it correct even with several connections on the same file."""
        now = time.time()
        async with self._lock:
            cur = await self._db.execute(
                "INSERT INTO ledger (key,session_id,tool,args_json,status,epoch,created_at,"
                "updated_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "status=excluded.status, epoch=excluded.epoch, result_json=NULL, "
                "updated_at=excluded.updated_at WHERE ledger.status IN (?,?,?)",
                (key, session_id, tool, canonical_json(args), LedgerStatus.IN_FLIGHT.value, epoch,
                 now, now, LedgerStatus.FAILED.value, LedgerStatus.BLOCKED.value,
                 LedgerStatus.COMPENSATED.value),
            )
            claimed = cur.rowcount == 1
            await self._db.commit()
            existing = None if claimed else await self.status(key)
        return claimed, existing

    async def _finish(self, key: str, status: LedgerStatus,
                      result: Mapping[str, Any] | None = None) -> None:
        async with self._lock:
            await self._db.execute(
                "UPDATE ledger SET status=?, result_json=?, updated_at=? WHERE key=?",
                (status.value, canonical_json(result) if result is not None else None,
                 time.time(), key))
            await self._db.commit()

    @asynccontextmanager
    async def guard(self, key: str, *, session_id: str = "", tool: str = "",
                    args: Mapping[str, Any] | None = None,
                    epoch: int = 0) -> AsyncIterator[GuardTicket]:
        """Usage::

            async with ledger.guard(key, session_id=..., tool=..., args=..., epoch=...) as ok:
                if ok:
                    ...execute the write...
                    ok.set_result(data)

        Exit marks COMMITTED on success, FAILED on any exception or cancellation (retryable);
        an exception carrying `ledger_status` (fencing) is recorded with that status instead.
        """
        claimed, existing = await self._try_claim(key, session_id, tool, args or {}, epoch)
        ticket = GuardTicket(key, claimed, existing)
        if not claimed:
            if self._trace:
                self._trace.emit("write_blocked_duplicate", epoch=epoch, tool=tool, key=key[:12],
                                 existing=existing.value if existing else None)
            yield ticket
            return
        try:
            yield ticket
        except BaseException as e:
            # an exception may name its own outcome (e.g. a fenced write is BLOCKED, not FAILED)
            status = getattr(e, "ledger_status", LedgerStatus.FAILED)
            await asyncio.shield(self._finish(key, status))
            raise
        await asyncio.shield(self._finish(key, LedgerStatus.COMMITTED, ticket.result))
        if self._trace:
            self._trace.emit("write_committed", epoch=epoch, tool=tool, key=key[:12])

    async def mark_compensated(self, key: str) -> bool:
        """COMMITTED -> COMPENSATED (after a compensating action succeeded)."""
        async with self._lock:
            cur = await self._db.execute(
                "UPDATE ledger SET status=?, updated_at=? WHERE key=? AND status=?",
                (LedgerStatus.COMPENSATED.value, time.time(), key, LedgerStatus.COMMITTED.value))
            await self._db.commit()
            return cur.rowcount == 1

    async def committed(self, session_id: str) -> list[dict[str, Any]]:
        """Committed, not-yet-compensated writes for a session (for stale-write compensation)."""
        async with self._db.execute(
            "SELECT key,tool,args_json,epoch,result_json FROM ledger "
            "WHERE session_id=? AND status=? ORDER BY created_at",
            (session_id, LedgerStatus.COMMITTED.value),
        ) as cur:
            rows = await cur.fetchall()
        return [{"key": k, "tool": t, "args": json.loads(a), "epoch": e,
                 "result": json.loads(r) if r else None} for k, t, a, e, r in rows]

    async def all_rows(self, session_id: str) -> list[dict[str, Any]]:
        """Every ledger row of a session, any status (for the state endpoint / debugging)."""
        async with self._db.execute(
            "SELECT key,tool,args_json,status,epoch FROM ledger WHERE session_id=? "
            "ORDER BY created_at", (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [{"key": k[:12], "tool": t, "args": json.loads(a), "status": st, "epoch": e}
                for k, t, a, st, e in rows]
