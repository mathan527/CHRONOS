# AI Disclosure

**Project:** CHRONOS, an interruptible real-time AI agent
**Team:** Zenera, SRM Institute of Science and Technology
**Event:** Samsung PRISM Generative AI Hackathon, Theme 05

This file states where AI is used in the product and where AI assisted in building it.

## 1. AI models used inside the product (runtime)

| Model | Used for | Where | Required? |
|---|---|---|---|
| Llama 3.2 3B (`llama3.2:3b`, via Ollama) | planning, slot extraction fallback, barge-in classifier fallback, diagnosis | [chronos/slowpath/llm.py](chronos/slowpath/llm.py) | No. Optional (`CHRONOS_LLM=ollama`). |
| Moondream2 (`moondream`, via Ollama) | describing camera frames for the field-troubleshooting use case | [chronos/slowpath/vision.py](chronos/slowpath/vision.py) | No. Optional. |
| **Mock LLM** (not an AI model) | deterministic, rule-based stand-in used by default, in tests and in the benchmark | [chronos/slowpath/llm.py](chronos/slowpath/llm.py) | Default |

- Both models run **locally** through Ollama. No user data is sent to a cloud AI service.
- Every model call has a timeout and a deterministic fallback, so the agent never depends on a model answering.
- The acknowledgment path ("Got it, switching to the 6pm flight…") is template-based and makes **no** model call.
- Writes (booking, navigation, reservation) are never decided by a model alone. They pass a deterministic guard: committed plan, current epoch, idempotency ledger.

### What was and was not measured

- All committed benchmark results (`bench/results.json`, `bench/results.md`, the Results page) were measured with the **mock LLM**. They show the coordination layer (epochs, cancellation, write guard, ledger), not language-model quality.
- The Ollama pass is implemented but **was not run** for the committed results, because Ollama was not installed on the benchmark machine. No Ollama figures are claimed anywhere in this repository.
- In the demo console, the model status pills read "simulated" in their hover text when the mock LLM is active.

## 2. AI assistance used to build the project (development)

- **Claude Code (Anthropic)** was used as an AI coding assistant during development, including drafting and editing source code, tests, the benchmark harness, the web demo console and site pages, the README and documentation, and debugging.
- The architecture and requirements come from the team's written project brief ([CLAUDE.md](CLAUDE.md)), which the assistant was instructed to follow.
- The test suite (`pytest`, 354 tests at the time of writing) and the benchmark harness are in the repository so the results can be reproduced and checked independently.

## 3. AI-generated or synthetic assets

- **Camera frames** in [scenarios/images/](scenarios/images/) are synthetic placeholder images drawn by a script ([scenarios/make_images.py](scenarios/make_images.py)). They are not photographs of real equipment.
- **Benchmark scenarios** are produced by a seeded generator ([bench/scenarios.py](bench/scenarios.py)), not collected from real users.
- **Tools and the "world"** (flights, bookings, navigation, tables, troubleshooting knowledge base) are mocks backed by SQLite.
- **Speech** is simulated as scripted streaming text. There is no speech recognition in this prototype.
- **Screenshots** in [docs/screenshots/](docs/screenshots/) are captured from the running demo console by [scripts/ui_shots.py](scripts/ui_shots.py).

## 4. Third-party software and models

Python 3.11, FastAPI, Pydantic v2, structlog, aiosqlite, httpx, PyYAML, pytest and Ruff (see [pyproject.toml](pyproject.toml)); Ollama; Llama 3.2 3B (Meta) and Moondream2 under their own licenses. Users are responsible for accepting those model licenses when they pull the models.

## 5. Known limitations relevant to AI use

See the **Limitations** section of the [README](README.md): a 3B model on real phrasing will make more classification and slot errors than the rule-based mock, which the timeouts and fallbacks contain but do not remove.
