"""Barge-in classifier: what did the user just do to a busy agent?

Tier 1: deterministic rules (microseconds). Tier 2: LLM with a hard timeout, only when tier 1
abstains. If tier 2 also fails, the fallback is HESITATION: the one label that neither bumps the
epoch nor commits anything, so a classifier failure can never destroy work or trigger a write.

Label semantics
  CORRECTION  tweaks a parameter of the current task ("make it 6pm", "next week instead")
  GOAL_CHANGE swaps in a different objective ("actually, gas station first")
  ADDITION    extends the current task ("and add a window seat")
  CANCEL      abandons it ("stop", "never mind")
  BACKCHANNEL acknowledgement, must not interrupt ("uh-huh", "okay")
  HESITATION  unfinished / filler, wait and do not act ("I want to boo...", "um")
  CLARIFICATION_QUESTION  the user asks something about what the agent is doing
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import date

from chronos.perception.intent import Intent, find_slots, norm, parse_rules
from chronos.protocol import BargeInType
from chronos.slowpath.llm import LLM, LLMError
from chronos.trace.logger import ComponentTrace

L = BargeInType


@dataclass(frozen=True)
class BargeInContext:
    """What the agent is currently doing (lets us tell CORRECTION from GOAL_CHANGE)."""
    current_intent: Intent | None = None


@dataclass(frozen=True)
class Tier1Decision:
    label: BargeInType
    confidence: float
    reason: str


@dataclass(frozen=True)
class BargeInResult:
    label: BargeInType
    tier: int  # 1 rules, 2 LLM, 0 fallback default
    confidence: float
    latency_ms: float
    reason: str
    text: str

    @property
    def bumps_epoch(self) -> bool:
        return self.label.bumps_epoch


# ------------------------------------------------------------------------ vocabulary --------
_BACKCHANNEL = (
    r"uh huh|mm hmm|mmhmm|mm hm|mhm|mm|hmm|okay|ok|k|kk|yeah|yep|yes|yup|right|sure|alright|all right|fine|"
    r"cool|i see|got it|understood|good|great|nice|haan|han|ha|hanji|achha|accha|acha|"
    r"theek hai|thik hai|ji|ya|yaa|hm|oh|ah|aha|oh okay|i understand|makes sense|correct")
_BACKCHANNEL_RE = re.compile(rf"(?:{_BACKCHANNEL})(?: (?:{_BACKCHANNEL}))*")
_FILLERS = {"um", "uh", "er", "erm", "hmm", "hm", "ah", "eh", "mmm", "umm", "uhh", "err", "ahh"}
_PURE_HESITATION_RE = re.compile(
    r"(?:(?:um+|uh+|er+m?|hmm+|ah+|eh|mmm+|hm|like|so|well|actually|wait|hold on|hang on|"
    r"let me think|let me see|one second|one sec|just a (?:second|moment|sec)|give me a (?:second|moment)|"
    r"how do i say|what'?s (?:it called|the word|that word)|what is the word|what do i say|okay so|ok so|and|but|or|then|see|you know|"
    r"just|basically|i mean|i think|umm+|uhh+|ek minute|ruko zara|ek second)[ ,]*)+")
_DANGLERS = {"to", "the", "a", "an", "for", "at", "on", "in", "and", "my", "with", "from", "of",
             "i", "i'm", "im", "wanna", "gonna", "can", "could", "then", "but", "or",
             "so", "like", "want", "need", "would", "will", "should", "if", "when", "how", "i'd",
             "i'll", "by", "into", "our", "your", "actually", "also", "make"}
_TRAILING_FILLERS = _FILLERS | {"like", "so", "actually", "basically", "you know"}
_PREFIX_VOCAB = ("book", "reserve", "navigate", "cancel", "change", "table", "flight", "delhi",
                 "mumbai", "chennai", "bengaluru", "bangalore", "tomorrow", "airport", "restaurant",
                 "station", "direction", "destination", "instead", "diagnose", "machine", "search",
                 "booking", "reservation", "window", "aisle", "seat", "petrol")
_LEAD_MARKERS = re.compile(
    r"^(?:(?:wait|hold on|hang on|hey|listen|sorry|oh|arre|arey|oops|um+|uh+|so|ok|okay|please|"
    r"yaar|ji|no wait|wait wait|excuse me|one second|one sec|hmm+|ah+|er+m?)[ ,]+)+")

_CANCEL_STRONG = re.compile(
    r"\b(?:never ?mind|forget (?:it|that|about it|about that)|scrap (?:it|that)|drop (?:it|that)|"
    r"leave it|abort|cancel (?:it|that|this|everything|all|the whole thing)|"
    r"stop (?:it|that|this|everything|now|right now|please|please stop|talking|doing)|"
    r"cancel my (?:booking|flight|reservation|order|trip|ticket|navigation)|"
    r"cancel the (?:booking|flight|reservation|order|trip|ticket|navigation)|"
    r"(?:don'?t|do not) (?:do|book|go|proceed|reserve|navigate|send)(?: that| it| anything)?|"
    r"no need|not needed|ruko|band karo|rehne do|chhod do|mat karo)\b")
_CANCEL_WEAK_TOKENS = {"stop", "cancel", "halt", "enough", "quit", "abort", "ruko", "nevermind"}
_NEGATED_CANCEL = re.compile(r"\b(?:don'?t|do not|no need to|never)\s+(?:cancel|stop)\b")

_ADD_LEAD = re.compile(
    r"^(?:and|also|plus|oh and|and also|one more thing|another thing|additionally|one more|"
    r"also one more thing|aur|and one more thing)\b")
_ADD_INLINE = re.compile(
    r"\badd\b|\bas well\b|\balong with\b|\bin addition\b|\bon top of that\b|"
    r"\bwith (?:a|an|the|one)? ?(?:window|aisle|extra|vegetarian|veg|non-veg|child|baby|wheelchair)\b|"
    r"\b(?:too|also)$")
_CORRECTION = re.compile(
    r"\b(?:make it|change it to|change (?:the |my )?(?:time|date|day|flight|seat|booking|"
    r"destination|table|number|count)|switch (?:it )?to|move it to|instead|rather|i meant|"
    r"i mean|on second thought|(?:it )?should be|shall be|not (?:the )?\w+(?: \w+)?,? (?:but|make|it'?s)|"
    r"(?:can|could|would) you (?:make|change|switch|move)|let'?s (?:say|make it|do)|"
    r"make that|change that|make (?:the|my) (?:time|date|day|number|count|seats?|booking)|correction|the other one|i said|,\s*not\s+(?:the\s+)?\S+(?:\s+\S+)?$|wrong (?:time|date|day|one|flight)|"
    r"nahi(?:n)?[, ]+(?:make|it|the|6|7|8|\w+ (?:am|pm))|no[, ]+(?:not|make|it'?s|it should))\b")
_QUESTION_LEAD = re.compile(
    r"^(?:what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|"
    r"will|would|shall|should|have|has|kya|kitna|kitne|kab|kahan|kaun|any|tell me|show me)\b")
_GOAL_CUES = re.compile(r"\bfirst\b|\bbefore (?:that|this|we|going)\b|\binstead of\b|"
                        r"\bnew (?:destination|plan|task)\b|\bdifferent (?:place|destination|task)\b|"
                        r"\blet'?s (?:go|do|stop|head)\b|\bforget (?:that|it),?\b")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9:'-]+", text)


def _clean(text: str) -> str:
    t = re.sub(r"\b([ap])\.m\.?", r"\1m", norm(text))  # p.m. -> pm
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)  # uh-huh -> uh huh
    t = re.sub(r"[^\w\s':.-]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # dots that are not decimal/time separators
    return re.sub(r"\s+", " ", t).strip(" -:")


def _ends_open(raw: str) -> bool:
    return bool(re.search(r"(?:\.{2,}|…|-|–|—|,)\s*$", raw.strip()))


def _is_prefix_fragment(tok: str) -> bool:
    return len(tok) >= 3 and any(w.startswith(tok) and w != tok for w in _PREFIX_VOCAB)


def tier1(text: str, *, final: bool = True, context: BargeInContext | None = None,
          today: date | None = None) -> Tier1Decision | None:
    """Pure, synchronous rule tier. Returns None when unsure (so tier 2 can decide)."""
    ctx = context or BargeInContext()
    raw = norm(text)
    clean = _clean(text)
    toks = _tokens(clean)
    if not toks:
        return Tier1Decision(L.HESITATION, 0.95, "empty")

    # -- CANCEL --------------------------------------------------------------------------
    body = _LEAD_MARKERS.sub("", clean).strip()
    btoks = _tokens(body)
    if not _NEGATED_CANCEL.search(body):
        if _CANCEL_STRONG.search(body):
            return Tier1Decision(L.CANCEL, 0.95, "cancel_phrase")
        if btoks and len(btoks) <= 3 and set(btoks) & _CANCEL_WEAK_TOKENS and \
                not re.search(r"\bstop (?:at|by|for|near)\b", body):
            return Tier1Decision(L.CANCEL, 0.9, "cancel_word_short")

    # -- BACKCHANNEL / pure hesitation ----------------------------------------------------
    if _BACKCHANNEL_RE.fullmatch(clean) and not (set(toks) <= _FILLERS):
        return Tier1Decision(L.BACKCHANNEL, 0.95, "backchannel")
    if set(toks) <= _FILLERS or _PURE_HESITATION_RE.fullmatch(clean + " "):
        return Tier1Decision(L.HESITATION, 0.95, "filler_only")

    # -- parse once for the content-bearing rules ------------------------------------------
    td = today or date.today()  # noqa: DTZ011 - local calendar day
    parsed = parse_rules(body, td)
    generic_slots = set(find_slots(body, None, td)) & {"time", "date", "party_size", "seat_pref",
                                                        "dest"}
    has_corr = bool(_CORRECTION.search(body))
    has_actually = bool(re.search(r"\b(?:actually|instead|rather)\b", body))
    has_add = bool(_ADD_LEAD.search(body) or _ADD_INLINE.search(body))
    complete_cue = has_corr or bool(generic_slots) or parsed.intent is not None

    # -- HESITATION: unfinished utterances ---------------------------------------------------
    last = toks[-1]
    if _ends_open(raw) and (not final or not complete_cue):
        return Tier1Decision(L.HESITATION, 0.9, "trailing_ellipsis")
    if last in _TRAILING_FILLERS and (not final or not complete_cue):
        return Tier1Decision(L.HESITATION, 0.85, "trailing_filler")
    is_question = bool(_QUESTION_LEAD.match(body) or raw.rstrip().endswith("?"))
    if (not final or len(btoks) <= 4) and last in _DANGLERS and not has_corr and not (
            is_question and (final or raw.rstrip().endswith("?"))):
        return Tier1Decision(L.HESITATION, 0.85, "dangling_word")
    if not final and _is_prefix_fragment(last):
        return Tier1Decision(L.HESITATION, 0.85, "truncated_word")

    # -- ADDITION -----------------------------------------------------------------------------
    if has_add and not has_corr and not has_actually and not raw.rstrip().endswith("?"):
        return Tier1Decision(L.ADDITION, 0.9, "addition_cue")

    # -- CORRECTION / GOAL_CHANGE -----------------------------------------------------------
    intent = parsed.intent
    same_task = intent is not None and ctx.current_intent is not None and intent == ctx.current_intent
    if has_corr:
        if intent is None or intent is Intent.CHANGE_BOOKING or same_task:
            return Tier1Decision(L.CORRECTION, 0.9, "correction_cue")
        return Tier1Decision(L.GOAL_CHANGE, 0.8, "correction_cue_new_intent")
    if has_actually or _GOAL_CUES.search(body):
        if intent is not None and not same_task:
            return Tier1Decision(L.GOAL_CHANGE, 0.9, "goal_change_cue")
        if generic_slots and (intent is None or same_task):
            return Tier1Decision(L.CORRECTION, 0.85, "actually_slot_only")
        if intent is not None:
            return Tier1Decision(L.CORRECTION, 0.75, "actually_same_intent")
    if intent is Intent.ADD_STOP and not is_question:
        return Tier1Decision(L.ADDITION, 0.8, "add_stop_no_reorder")
    if is_question:
        return Tier1Decision(L.CLARIFICATION_QUESTION, 0.9, "question")
    if intent is not None and ctx.current_intent is not None and not same_task and final:
        return Tier1Decision(L.GOAL_CHANGE, 0.75, "different_intent")
    if generic_slots and len(btoks) <= 5 and intent is None:
        return Tier1Decision(L.CORRECTION, 0.75, "slot_only_fragment")
    return None


# -------------------------------------------------------------------------- classifier -----
_SYSTEM = (
    "You label what a user said while a voice assistant was busy working on their request. "
    'Reply ONLY with JSON {"label": <one of correction, goal_change, addition, cancel, '
    'backchannel, hesitation, clarification_question>, "confidence": 0..1}. '
    "correction=tweaks a parameter of the current task; goal_change=replaces the objective; "
    "addition=adds to the task; cancel=abandon; backchannel=acknowledgement like 'okay'; "
    "hesitation=unfinished or filler; clarification_question=asks about the task.")


class BargeInClassifier:
    def __init__(self, llm: LLM | None = None, *, timeout_s: float = 0.25,
                 today: date | None = None, trace: ComponentTrace | None = None) -> None:
        self.llm, self.timeout_s, self._today, self._trace = llm, timeout_s, today, trace

    async def classify(self, text: str, *, final: bool = True,
                       context: BargeInContext | None = None, epoch: int = 0,
                       allow_llm: bool = True) -> BargeInResult:
        """`allow_llm=False` (agent idle: nothing to interrupt) skips tier 2 entirely; an
        undecided utterance then comes back as tier 0 / reason 'fallback:llm_disabled'."""
        t0 = time.perf_counter()
        d = tier1(text, final=final, context=context, today=self._today)
        if d is not None:
            return self._done(text, epoch, d.label, 1, d.confidence, d.reason, t0)
        reason = "no_llm" if self.llm is None else "llm_disabled"
        if self.llm is not None and allow_llm:
            payload = {"utterance": text, "final": final,
                       "current_intent": context.current_intent.value
                       if context and context.current_intent else None}
            try:
                raw = await asyncio.wait_for(
                    self.llm.chat_json("bargein", _SYSTEM, json.dumps(payload),
                                       timeout=self.timeout_s),
                    timeout=self.timeout_s)
                label = BargeInType(str(raw["label"]).lower())
                conf = min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0)
                return self._done(text, epoch, label, 2, conf, "llm", t0)
            except TimeoutError:
                reason = "llm_timeout"
            except LLMError:
                reason = "llm_error"
            except (KeyError, TypeError, ValueError):
                reason = "llm_invalid"
        return self._done(text, epoch, L.HESITATION, 0, 0.0, f"fallback:{reason}", t0)

    def _done(self, text: str, epoch: int, label: BargeInType, tier: int, conf: float,
              reason: str, t0: float) -> BargeInResult:
        res = BargeInResult(label, tier, conf, (time.perf_counter() - t0) * 1000, reason, text)
        if self._trace:
            self._trace.emit("classified", epoch=epoch, text=text, label=label.value, tier=tier,
                             confidence=conf, reason=reason, latency_ms=round(res.latency_ms, 3))
        return res
