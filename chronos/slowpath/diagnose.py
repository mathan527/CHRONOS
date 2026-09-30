"""The one place the LLM writes a troubleshooting diagnosis (used by the planner AND by the
benchmark baseline, so both agents make the same model call with the same fallback)."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from chronos.slowpath.diagnosis import (
    SYSTEM_PROMPT,
    Diagnosis,
    LLMDiagnosis,
    deterministic_diagnosis,
    is_grounded,
)
from chronos.slowpath.llm import LLM, LLMError


async def diagnose_with_llm(llm: LLM | None, question: str, desc: str | None,
                            matches: list[dict[str, Any]], timeout_s: float) -> Diagnosis:
    """LLM diagnosis validated against a schema; any failure falls back to the deterministic one."""
    if llm is not None:
        payload = {"question": question, "frame_description": desc,
                   "kb_matches": [{k: m.get(k) for k in ("id", "title", "causes", "steps")}
                                  for m in matches]}
        try:
            raw = await llm.chat_json("diagnose", SYSTEM_PROMPT, json.dumps(payload),
                                      timeout=timeout_s)
            d = LLMDiagnosis.model_validate(raw)
            top = matches[0] if matches else None
            return Diagnosis(
                diagnosis=d.diagnosis, likely_cause=d.likely_cause or (top or {}).get("title", ""),
                steps=d.steps, observed=desc, grounded=is_grounded(desc, top),
                kb_ids=[m["id"] for m in matches if m["id"] != "no_match"], source="llm")
        except (LLMError, ValidationError, ValueError, TimeoutError):
            pass
    return deterministic_diagnosis(question, desc, matches)
