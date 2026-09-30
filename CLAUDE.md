# CHRONOS — Project Brief (persisted from the original kickoff message)

You are helping me build CHRONOS, an interruptible real-time AI agent, for the Samsung PRISM
Generative AI Hackathon (Theme 05). Before writing any code, save this entire message as
CLAUDE.md in the repo root so it persists across sessions. Then reply with a short summary of the
plan and the repo layout you will use. Do NOT write feature code yet.

## Problem
Today's voice assistants are half-duplex: listen → think → speak → act, one step at a time.
If the user interrupts ("Wait! Make it the 6pm flight!") the agent either ignores them, freezes,
or throws away all work and restarts. Worse, it may execute stale writes (booking the 8pm flight
AND the 6pm flight = double booking).

## Core principle
"Reads can be speculative; writes must be committed. Every interruption or change creates a new
version (epoch)."

## Architecture (implement exactly these components)
1. Event Sources: user audio (simulated as streaming partial transcripts), text input, camera
   frames (image files / base64), and explicit interrupt/barge-in events.
2. Unified Event Queue: a single asyncio.PriorityQueue; interrupts have highest priority.
   Every event is a Pydantic model with: event_id, session_id, type, payload, ts_monotonic, epoch.
3. Perception Layer:
   - Intent Understanding: extract intent + slots (e.g. book_flight{dest, date, time}).
   - Barge-in Classifier: labels each incoming utterance, while the agent is busy, as one of:
     CORRECTION ("make it 6pm"), GOAL_CHANGE ("actually, gas station first"),
     ADDITION ("and add a window seat"), CANCEL ("stop", "never mind"),
     BACKCHANNEL ("uh-huh", "okay" — must NOT interrupt), HESITATION ("I want to boo...",
     "um" — wait, don't act), CLARIFICATION_QUESTION.
     Two tiers: deterministic rules/regex first (<5 ms), LLM fallback with a hard timeout.
4. Fast Path: deterministic acknowledgment in < 300 ms, template-based, NO LLM call.
   It must NEVER claim a task is done before it is ("Got it, switching to the 6pm flight…",
   never "Booked!").
5. Slow Path:
   - Speculative Planner: builds a plan with the local LLM; starts read-only tool calls
     immediately, even while the user is still talking.
   - Local LLM: Llama 3.2 3B via Ollama (model tag `llama3.2:3b`), JSON-mode output.
   - Vision Grounding: Moondream2 via Ollama (model tag `moondream`) for camera frames.
   - Read-only Tool Calls: run speculatively, results cached per (tool, args).
6. Coordination & Safety Layer:
   - Epoch Manager: monotonic integer per session; bumped on every CORRECTION, GOAL_CHANGE,
     CANCEL (not on BACKCHANNEL/HESITATION).
   - Epoch Versioning: every task, plan, tool call and output is stamped with its epoch.
   - Cancellation Manager: on epoch bump, cancels all asyncio tasks of older epochs
     (task.cancel()) and discards their results. Results arriving with a stale epoch are dropped
     and logged, never emitted.
   - State Snapshot: immutable snapshot of slots/plan per epoch so a correction patches the
     previous state incrementally instead of replanning from scratch. Reuse still-valid
     speculative read results.
   - Idempotency Ledger: every write gets an idempotency key =
     sha256(session_id + tool_name + canonical_json(args)). A write executes only if the key is
     not already in the ledger with status COMMITTED/IN_FLIGHT. The ledger is async-safe
     (asyncio.Lock) and persisted to SQLite.
7. Tool Layer:
   - Read Tools (safe to speculate): search_flights, get_route, check_table_availability,
     get_booking, lookup_troubleshooting_kb.
   - Write Tools (guarded): book_flight, cancel_booking, set_navigation, reserve_table.
   - Write guard: a write may run ONLY if (a) the plan is COMMITTED (user finished speaking /
     confirmed), (b) its epoch == current epoch, checked again immediately before dispatch
     (fencing token), and (c) the idempotency ledger allows it. If an already-committed write
     becomes stale, issue a compensating action (e.g. cancel_booking) through the same ledger.
   - All tools are mocks backed by an in-memory/SQLite "world" with configurable random latency
     (e.g. 100–800 ms) so interruptions actually land mid-task.
8. Action Output: every message to the client is protocol-compliant JSON validated by a Pydantic
   model: {type: ack|progress|response|action_result|error, session_id, epoch, status:
   pending|committed|cancelled|done|failed, text, data, trace_id, ts}. Keep this schema in ONE
   file (chronos/protocol.py) so it can be changed to match the hackathon's official spec.
9. Trace System: every component writes structured JSONL trace events (structlog) to
   traces/<session_id>.jsonl: event_received, classified, ack_sent, epoch_bumped,
   task_started, task_cancelled, stale_result_dropped, tool_read, write_blocked_duplicate,
   write_committed, response_sent — each with epoch and ms timestamps.
   Trace Timeline Visualizer: a single self-contained HTML page (served by FastAPI and also
   exportable offline) that renders the JSONL as swim lanes (Perception / Fast / Slow /
   Coordination / Tools) on a time axis, colored by epoch, with cancelled work shown struck
   through.

## Tech stack (fixed)
Python 3.11 + asyncio · Ollama (llama3.2:3b, moondream) · Pydantic v2 · FastAPI + WebSocket ·
structlog · SQLite (aiosqlite) · Docker + docker-compose · pytest + pytest-asyncio.

## Use cases the demo must show
1. In-car: "Navigate to Chennai Airport" → mid-route-calc "Actually, gas station first" →
   old route task cancelled (epoch 1), new route planned (epoch 2), only one set_navigation.
2. Customer support: "Book a flight to Delhi tomorrow" → "Actually, next week instead" →
   no double booking, no double charge; if the first booking was already committed, it is
   cancelled via compensating action.
3. Field troubleshooting: user sends a camera frame + "What's wrong with this machine?" →
   Moondream describes the frame, LLM diagnoses using the KB read tool, answer grounded in
   what the camera sees.
4. Accessibility: "I want to boo… um… actually, book a table for 4" → hesitation is not acted
   on, self-correction updates the current task incrementally, one reservation.

## Engineering rules
- Fully async; never block the event loop (wrap blocking calls in asyncio.to_thread).
- Every LLM call has a timeout and a deterministic fallback so the agent never freezes.
- The system must run with a MOCK LLM (env CHRONOS_LLM=mock) so tests run without Ollama.
- Type hints everywhere; small modules; no global mutable state outside the session object.
- Write tests alongside each component. Run `pytest -q` after every phase and fix failures
  before reporting done.
- Metrics must be MEASURED by the benchmark harness, never hardcoded.
- Work phase by phase. After each phase: show what was built, test results, and what's next.

## Target repo layout
chronos/
  protocol.py        # Pydantic event + output schemas
  events/queue.py    # unified priority event queue
  perception/intent.py, perception/bargein.py
  fastpath/ack.py
  slowpath/planner.py, slowpath/llm.py, slowpath/vision.py
  coordination/epoch.py, coordination/cancellation.py, coordination/snapshot.py,
  coordination/ledger.py
  tools/registry.py, tools/read_tools.py, tools/write_tools.py, tools/world.py
  agent/session.py   # orchestrator wiring everything together
  trace/logger.py, trace/visualizer.html
  api/server.py      # FastAPI + WebSocket
bench/               # baseline vs chronos benchmark harness
scenarios/           # scripted demo scenarios (YAML)
tests/
web/                 # minimal demo client (text box, mic-sim, image upload, live JSON feed)
Dockerfile, docker-compose.yml, README.md, pyproject.toml
