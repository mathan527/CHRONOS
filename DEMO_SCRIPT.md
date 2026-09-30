# CHRONOS: 5-minute demo video script

Team Zenera, SRM Institute of Science and Technology · Samsung PRISM Generative AI Hackathon, Theme 05

**Format:** screen recording with voice-over. Left half of the screen: the web client
(<http://localhost:8000>). Right half: the trace timeline (`/traces/<session>/view`), refreshed
after each scenario. Keep the "Interrupt" and "Scripted scenarios" buttons visible.

**Before you record** (2 minutes, off camera)

1. `docker compose up` (or `make run` for mock mode). Wait until `/health` shows both models ✓ (or
   "LLM mode: mock").
2. Open the client, note the session id, and open the timeline in a second window.
3. Use a fresh session per scenario (**New session** button) so each timeline is short and readable.
4. Set the speech-speed slider so "Navigate to Chennai Airport" takes about 2 seconds to type out.
5. Do one silent dry run of all four; latency in mock mode is 100-800 ms per tool call, so every
   interruption lands mid-task.

---

## 0:00 – 0:30 · The problem and the idea

**On screen:** title card, then a slide with two lines: "Listen → think → speak → act" vs. CHRONOS.

**Say:**
"Voice assistants today are half-duplex. They listen, think, speak, then act, one step at a time.
When you interrupt, one of three things happens: they ignore you, they freeze, or they throw away
everything and start over. And the worst case: they run the stale request *and* the new one. You
say 'book the 8pm flight', then 'make it 6pm', and you get both flights and both charges.
CHRONOS is our answer. One principle: **reads can be speculative, writes must be committed, and every
interruption creates a new version, an epoch.**"

## 0:30 – 1:15 · Architecture in 45 seconds

**On screen:** the Mermaid diagram from the README (screenshot or the rendered GitHub page). Point
at each block as you name it.

**Say:**
"Events come in through one priority queue, and interrupts jump the line. Perception extracts intent
and classifies barge-ins with rules first, under 5 milliseconds, and an LLM fallback with a hard
timeout. A fast path acknowledges in under 300 milliseconds from a template. No LLM, and it never
claims a task is done before it is. The slow path plans with a local Llama 3.2 3B, grounds camera
frames with Moondream, and starts read-only tool calls immediately. The coordination layer is the heart
of it: epochs, cancellation, state snapshots, and an idempotency ledger. A write only runs if the plan
is committed, its epoch is still current, checked again right before dispatch, and the ledger hasn't
seen it."

## 1:15 – 2:00 · Scenario 1: in-car, change of goal

**Do:** New session. Click **1 · In-car**. (Or type "Navigate to Chennai Airport", wait about a second,
then "Actually, gas station first".)

**Say while it plays:**
"I'm driving. 'Navigate to Chennai Airport'... and before the route finishes, 'Actually, gas station
first.' Watch the feed. The ack for the new request arrives instantly, and it says 'changing plans',
not 'done'. Now the timeline."

**Switch to timeline. Point at:** the epoch 1 route task struck through in the Slow lane, the epoch bump
marker in Coordination, epoch 2 in a new colour.

**Say:**
"Epoch 1 was cancelled mid-flight: struck through. Epoch 2 planned the route via the gas station and
reused the destination from the snapshot instead of starting over. Exactly one `set_navigation`
reached the tools lane."

## 2:00 – 3:00 · Scenario 2: customer support, the double-booking case

**Do, part A (before booking):** New session. Click **2a · Support: change the flight before it is booked**.

**Say:**
"'Book a flight to Delhi tomorrow', then 'actually, next week instead.' The correction lands
during the search. The plan is patched, and only next week's flight is booked. One booking, one charge."

**Do, part B (after booking):** New session. Click **2b · change the flight after it is booked**.

**Say:**
"Now the hard case. The first booking has already committed. Here's the correction."
**Point at the feed:** the `cancelled` action_result, then the new booking.
"CHRONOS didn't ignore the first booking and didn't double-book. It issued a compensating
`cancel_booking` through the same idempotency ledger, refunded it, and booked the new date. Let's
look at the ledger."

**Open** `/sessions/<id>/state` in a tab and scroll to `ledger`: one `book_flight` COMPENSATED, one
COMMITTED. **Say:** "One live booking, one active charge, and the audit trail shows why."

## 3:00 – 3:45 · Scenario 3: field troubleshooting with vision

**Do:** New session. Click **Sample: stopped fan** (or upload a photo), with the text box set to
"What's wrong with this machine?".

**Say:**
"A technician points a camera at a machine and asks what's wrong. Moondream describes the frame:
a stopped cooling fan, dust blocking the vents, a red temperature light. The LLM then diagnoses it using
the troubleshooting knowledge-base read tool, and the answer is grounded in what the camera actually
sees, not a generic guess. Steps: power off, clear the vents, check the fan, restart."

**Point at the timeline:** `frame_described` in the Slow lane, then `lookup_troubleshooting_kb` in the
Tools lane, then `diagnosis_ready` and the response. "Reads only, no writes, so there was nothing to
commit and nothing to undo."

## 3:45 – 4:15 · Scenario 4: accessibility, hesitation and self-correction

**Do:** New session. Click **4 · Accessibility**. (Partial "I want to boo…", pause, "I want to boo…
um…", pause, then "actually, book a table for 4".)

**Say:**
"Speech isn't clean. 'I want to boo… um…' is a hesitation. CHRONOS waits and doesn't act, and it
doesn't bump the epoch. Then the self-correction, 'actually, book a table for 4', updates the current
task in place. One reservation. Not a half-finished booking, and not two."

**Point at the timeline:** no plan and no epoch bump while the hesitant partials arrive, then the
correction producing one plan, one `check_table_availability` read, and a single `write_committed` for
`reserve_table` in the Coordination lane, all in epoch 1.
Also show the stray-"okay" case: type "okay" during any task. "Backchannels are traced and ignored."

## 4:15 – 5:00 · Numbers, honesty, and wrap-up

**On screen:** `bench/results.md` headline table (or `bench/charts/mock_correctness.png`).

**Say:**
"We benchmarked CHRONOS against a naive half-duplex agent on the same tools and language model, over 200
randomised interruption scenarios. Every number is measured, not hardcoded. Final state consistent with
what the user last asked for: CHRONOS 200 out of 200, baseline 182 out of 200. Duplicate live writes: zero
versus 9 percent. Double charges: zero versus 4.5 percent. And time to first response: about
1 millisecond versus 106, which comes from the fast path acknowledgment. To be clear, that is
acknowledgment latency, not faster task completion.
Limits, stated plainly: tools and speech are simulated, this run used the deterministic mock LLM,
and the local-model pass is implemented but wasn't part of these committed numbers.
Everything runs with one command, `docker compose up`, or in mock mode with no GPU. Thank you. We are
Team Zenera from SRM Institute of Science and Technology."

---

### Shot checklist

- [ ] Health line at the top of the client shows the intended mode (models ✓ or "mock").
- [ ] A fresh session for each scenario; timeline refreshed after each one.
- [ ] Timeline zoomed so struck-through tasks and epoch colours are legible at 1080p.
- [ ] Ledger view open for scenario 2b.
- [ ] Results table on screen for the last 45 seconds; the ack-latency caveat spoken aloud.

### If something goes wrong on camera

- **A tool call finishes before the interrupt lands** (the timing is random): click the scenario again;
  each scenario button uses fixed waits, so it is usually reproducible. In mock mode you can also set
  `CHRONOS_TOOL_LATENCY_MS=300,600` to widen the window.
- **Ollama is slow on first use** (model load): send one throwaway request before recording. The agent
  falls back deterministically if the LLM times out, but a warm model looks better.
