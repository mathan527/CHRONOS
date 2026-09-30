"""Single source of truth for every wire/event schema. Edit here to match the official spec."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
    return uuid.uuid4().hex


class EventType(str, Enum):
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    TEXT = "text"
    CAMERA_FRAME = "camera_frame"
    INTERRUPT = "interrupt"


class BargeInType(str, Enum):
    CORRECTION = "correction"
    GOAL_CHANGE = "goal_change"
    ADDITION = "addition"
    CANCEL = "cancel"
    BACKCHANNEL = "backchannel"
    HESITATION = "hesitation"
    CLARIFICATION_QUESTION = "clarification_question"

    @property
    def bumps_epoch(self) -> bool:
        return self in (BargeInType.CORRECTION, BargeInType.GOAL_CHANGE, BargeInType.CANCEL)


class PlanStatus(str, Enum):
    DRAFT = "draft"  # speculative: reads only
    COMMITTED = "committed"  # user finished/confirmed: writes allowed
    CANCELLED = "cancelled"
    DONE = "done"
    FAILED = "failed"


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)

    event_id: str = Field(default_factory=_uid)
    session_id: str
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    # time.perf_counter: monotonic AND high resolution (time.monotonic ticks at ~15.6 ms on Windows)
    ts_monotonic: float = Field(default_factory=time.perf_counter)
    epoch: int = Field(default=0, ge=0)


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str = Field(default_factory=_uid)
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    epoch: int = Field(default=0, ge=0)
    is_write: bool = False
    idempotency_key: str | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    epoch: int = Field(default=0, ge=0)
    latency_ms: float = 0.0
    cached: bool = False
    idempotency_key: str | None = None  # set on a committed write


class OutputType(str, Enum):
    ACK = "ack"
    PROGRESS = "progress"
    RESPONSE = "response"
    ACTION_RESULT = "action_result"
    ERROR = "error"


class OutputStatus(str, Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    DONE = "done"
    FAILED = "failed"


class OutputMessage(BaseModel):
    """Every message sent to the client."""

    type: OutputType
    session_id: str
    epoch: int = Field(ge=0)
    status: OutputStatus
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=_uid)
    ts: float = Field(default_factory=time.time)
