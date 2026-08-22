"""reiterate_terminal_outcome: terminal node for a turn on a thread that
already reached crisis/emergency earlier, where THIS turn found nothing
new to escalate. Without it, a thread that already got an emergency
directive could silently re-enter ordinary triage on the next message
("ما بدي اتصل بالإسعاف").

No LLM call — a fixed, reviewed reminder can't drift into reassurance or
negotiation the way a fresh generation could.

thread_outcome (not severity) is the field checked here: a thread-
lifecycle marker ("did this thread already conclude via a terminal
safety path"), separate from severity's graded clinical judgment. crisis
can't be a severity value at all — safety rule 9 stops symptom analysis
entirely on that path, so there's no severity judgment to grade.

Escalation is never blocked: crisis_check/check_red_flags run every
turn regardless of thread_outcome, and are checked BEFORE it in graph.py
— this node only fires when neither found anything new.
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
    outcome = state.get("thread_outcome")

    if outcome == "crisis":
        reply = f"{_CRISIS_REMINDER}\n\n{support_line_text()}"
    else:
        # Safe default if ever reached with thread_outcome unset.
        reply = _EMERGENCY_REMINDER

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "stage": outcome or "emergency",
    }
