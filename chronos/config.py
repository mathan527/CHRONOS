"""Environment-driven settings. Immutable; passed explicitly, never mutated globally."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _ollama_url() -> str:
    """OLLAMA_HOST (what the docker-compose file sets) wins over OLLAMA_URL; the scheme is optional."""
    raw = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL") or "http://localhost:11434"
    return raw if "://" in raw else f"http://{raw}"


@dataclass(frozen=True)
class Settings:
    llm_mode: str = "mock"  # "mock" | "ollama"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    vision_model: str = "moondream"
    llm_timeout_s: float = 8.0
    classifier_llm_timeout_s: float = 0.25
    eou_silence_ms: int = 700  # end-of-utterance silence before a partial transcript is committed
    tool_latency_ms: tuple[int, int] = (100, 800)
    trace_dir: str = "traces"
    db_path: str = "chronos.db"

    @classmethod
    def from_env(cls) -> Settings:
        lo, hi = os.getenv("CHRONOS_TOOL_LATENCY_MS", "100,800").split(",")
        return cls(
            llm_mode=os.getenv("CHRONOS_LLM", "mock"),
            ollama_url=_ollama_url(),
            llm_model=os.getenv("CHRONOS_LLM_MODEL", "llama3.2:3b"),
            vision_model=os.getenv("CHRONOS_VISION_MODEL", "moondream"),
            llm_timeout_s=float(os.getenv("CHRONOS_LLM_TIMEOUT", "8")),
            classifier_llm_timeout_s=float(os.getenv("CHRONOS_CLF_TIMEOUT", "0.25")),
            eou_silence_ms=int(os.getenv("CHRONOS_EOU_MS", "700")),
            tool_latency_ms=(int(lo), int(hi)),
            trace_dir=os.getenv("CHRONOS_TRACE_DIR", "traces"),
            db_path=os.getenv("CHRONOS_DB", "chronos.db"),
        )
