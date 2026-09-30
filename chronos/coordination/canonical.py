"""Canonical JSON + hashing helpers shared by the ledger and the read cache."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _default(obj: Any) -> Any:
    if isinstance(obj, Mapping):  # frozen (MappingProxyType) args must hash like plain dicts
        return dict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, so equal args always hash equal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_default)


def call_key(tool: str, args: Mapping[str, Any]) -> str:
    return f"{tool}:{canonical_json(args)}"


def idempotency_key(session_id: str, tool: str, args: Mapping[str, Any]) -> str:
    # 0x1f separator: prevents ("ab","c") colliding with ("a","bc").
    raw = "\x1f".join((session_id, tool, canonical_json(args)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
