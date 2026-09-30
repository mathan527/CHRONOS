"""Grounded troubleshooting diagnosis: what the camera saw + what the KB says.

The LLM writes the diagnosis (JSON), but `deterministic_diagnosis` builds the same structure
without any model, so the agent always has an answer when the LLM is slow, down, or wrong.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMDiagnosis(BaseModel):
    """Schema the LLM must satisfy; anything else triggers the deterministic fallback."""
    model_config = ConfigDict(extra="ignore")

    diagnosis: str = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    likely_cause: str = ""


class Diagnosis(BaseModel):
    diagnosis: str
    likely_cause: str
    steps: list[str]
    observed: str | None = None  # the camera description this answer is grounded in
    grounded: bool = False  # True only if the observation supports the chosen KB match
    kb_ids: list[str] = Field(default_factory=list)
    source: Literal["llm", "fallback"] = "fallback"


SYSTEM_PROMPT = (
    "You are a field-service assistant. Diagnose the machine problem using ONLY the camera "
    "description and the knowledge-base matches provided. Mention what the camera shows. "
    'Reply with ONLY JSON: {"diagnosis": string, "likely_cause": string, "steps": [string, ...]}.')


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def is_grounded(observed: str | None, match: dict[str, Any] | None) -> bool:
    """The observation supports the match if it shares at least one of the match's keywords."""
    if not observed or not match or match.get("id") == "no_match":
        return False
    ow = _words(observed)
    return any(k in ow or any(k in w for w in ow if len(w) > 3) for k in match.get("keywords", []))


def deterministic_diagnosis(question: str, observed: str | None,
                            matches: list[dict[str, Any]]) -> Diagnosis:
    top = matches[0] if matches else None
    if top is None or top.get("id") == "no_match":
        cause = "No known fault matches the symptoms."
        steps = list(top["steps"]) if top else ["Power off safely.", "Escalate to a technician."]
        lead = f"The camera shows: {observed.rstrip('.')}. " if observed else ""
        return Diagnosis(diagnosis=f"{lead}{cause}", likely_cause=cause, steps=steps,
                         observed=observed, grounded=False, kb_ids=[], source="fallback")
    title, causes = top["title"], top["causes"]
    if observed:
        text = (f"The camera shows: {observed.rstrip('.')}. That points to "
                f"{title[0].lower() + title[1:]}. {causes}")
    else:
        text = (f"I have no camera image, so I am going on your description only. Most likely: "
                f"{title[0].lower() + title[1:]}. {causes}")
    return Diagnosis(diagnosis=text, likely_cause=title, steps=list(top["steps"]),
                     observed=observed, grounded=is_grounded(observed, top),
                     kb_ids=[m["id"] for m in matches], source="fallback")
