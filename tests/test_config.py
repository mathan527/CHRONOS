from chronos.config import Settings


def test_defaults_are_mock(monkeypatch):
    monkeypatch.delenv("CHRONOS_LLM", raising=False)
    s = Settings.from_env()
    assert s.llm_mode == "mock"
    assert s.tool_latency_ms == (100, 800)


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHRONOS_LLM", "ollama")
    monkeypatch.setenv("CHRONOS_TOOL_LATENCY_MS", "1,5")
    s = Settings.from_env()
    assert s.llm_mode == "ollama" and s.tool_latency_ms == (1, 5)


def test_ollama_host_is_accepted_with_or_without_a_scheme(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "ollama:11434")
    assert Settings.from_env().ollama_url == "http://ollama:11434"
    monkeypatch.setenv("OLLAMA_HOST", "https://gpu-box:1234")
    assert Settings.from_env().ollama_url == "https://gpu-box:1234"
    monkeypatch.delenv("OLLAMA_HOST")
    monkeypatch.setenv("OLLAMA_URL", "http://elsewhere:9")
    assert Settings.from_env().ollama_url == "http://elsewhere:9"
