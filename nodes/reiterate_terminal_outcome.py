"""reiterate_terminal_outcome: the terminal node for a turn on a thread
that already reached crisis or emergency on an EARLIER turn, where THIS
turn's crisis_check/check_red_flags found nothing new to escalate to
(CLAUDE.md > Non-negotiable safety rule 13).

Manual testing surfaced the gap this node closes: a thread that already
received a crisis or emergency directive had nothing preventing the next
message from silently re-entering ordinary symptom triage — a patient
telling the bot "ما بدي اتصل بالإسعاف" (I don't want to call the
ambulance) right after an emergency directive got processed as if it
were an unrelated fresh turn, with no structural guarantee the pipeline
wouldn't eventually produce a normal differential-diagnosis report on a
thread that had already told the patient to go to the ER. graph.py's
routing after check_red_flags now checks state["thread_outcome"]
(state.py) and reaches this node instead of assess_sufficiency when it's
already set and nothing new fired this turn.

No LLM call, deliberately, even though crisis_node calls one on its
FIRST firing. Once a thread already has a standing directive, a second
LLM call earns nothing (the fact to convey never changes) and only adds
risk: a fixed, hand-authored, previously-reviewed message cannot drift
into reassurance, negotiation, or analysis the way a fresh generation
theoretically could under repeated prompting. This mirrors why
emergency_node itself has never called an LLM at all.

--- Why thread_outcome, not severity: three concepts, not two.

state["severity"] is a graded clinical-severity judgment
(low/moderate/high/emergency) — content, meant to reflect "how severe is
the presenting picture," protected by safety rule 5 from being
downgraded by patient objection. state["thread_outcome"] is a thread-
lifecycle marker — routing, meant to answer "has this thread already
concluded via a terminal safety path, and which one." These are
different axes with different update disciplines: severity is (in
principle) re-evaluated as new clinical information arrives, while
thread_outcome is write-once-then-sticky per terminal firing. The
concrete proof they don't collapse into one field: "crisis" cannot be a
severity value at all. Safety rule 9 requires stopping symptom analysis
entirely on the crisis path — there is no clinical-severity judgment
being made there to grade, let alone record. Reusing/extending Severity
with a 5th "crisis" tier would assert a judgment rule 9 explicitly
forbids making. See state.py's own field comments for both, and
nodes/emergency_node.py / nodes/crisis_node.py for where each is
actually written.

A third candidate value was considered and rejected: "diagnosis_complete"
(a normal, non-emergency consultation concluding via generate_reports)
does not belong on thread_outcome either. Unlike crisis/emergency,
continuing to chat after an ordinary diagnosis is not unsafe — the
failure mode there is data-quality (a later unrelated complaint merging
into the same accumulated symptom set), and a sticky block risks
deflecting a genuinely useful addendum (e.g. a missed, possibly
red-flag-relevant detail) into "start a new conversation" instead of
letting it through. That risk profile doesn't match crisis/emergency's,
so it was left to the existing stage=="diagnosis" signal and a
consumer-side (UI) decision, not a new backend enforcement mechanism.
ThreadOutcome (state.py) is therefore scoped to exactly the two SAFETY
terminal outcomes.

--- Escalation is never blocked by this design. crisis_check and
check_red_flags still run on EVERY turn regardless of thread_outcome
(graph.py's edges are unconditional up to that point) — a genuine new
crisis signal or a genuine new red flag must always be able to reach the
REAL crisis_node/emergency_node, even on a thread that already reached
the other terminal outcome. graph.py's routing checks red_flags, then
is_crisis (via the earlier crisis_check branch), BEFORE thread_outcome —
this node is only reached when neither fired this turn. Both terminal
nodes overwrite thread_outcome unconditionally when they DO fire again,
so the field always reflects whichever terminal safety outcome most
recently applied, not necessarily the first one.
"""

from __future__ import annotations

from typing import Any

from nodes._shared import support_line_text
from state import HealixState

_CRISIS_REMINDER = (
    "زي ما حكينا قبل شوي، الموضوع الأهم هلق إنك تتواصل مع حد قادر يساعدك "
    "فعليًا وبسرعة."
)

_EMERGENCY_REMINDER = (
    "زي ما حكينا قبل شوي، هاد وضع طارئ وما بيتحمل التأجيل. توجه فورًا "
    "لأقرب قسم طوارئ أو اتصل بالإسعاف — الاستمرار بالحديث هون ما بيغني "
    "عن هيك."
)


def reiterate_terminal_outcome(state: HealixState) -> dict[str, Any]:
    """Re-issue the thread's standing safety directive without re-entering
    the pipeline. No LLM call — see module docstring."""
    outcome = state.get("thread_outcome")

    if outcome == "crisis":
        reply = f"{_CRISIS_REMINDER}\n\n{support_line_text()}"
    else:
        # "emergency" is the only other value graph.py's routing ever
        # reaches this node for (state.ThreadOutcome has no third value).
        # Also the safer default if this node were ever somehow reached
        # with thread_outcome unset (not expected — routing only reaches
        # it when the field is already set) rather than guessing which
        # directive applies.
        reply = _EMERGENCY_REMINDER

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "stage": outcome or "emergency",
    }
