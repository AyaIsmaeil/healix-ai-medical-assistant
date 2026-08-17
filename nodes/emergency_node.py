"""emergency_node: the terminal node for the red-flag path (CLAUDE.md >
Graph flow, CLAUDE.md > Non-negotiable safety rule 4: the red-flag path
bypasses RAG and diagnosis entirely, routing straight to its terminal
node).

Deliberately has no LLM call, unlike crisis_node — nothing here needs to
vary per-turn or acknowledge what the patient said in their own words.
The patient-facing message is fixed: state urgency plainly, direct to an
ER or emergency services, and stop. Explaining *why* would mean
describing which red flag fired, which reads as a clinical explanation
bordering on diagnosis (CLAUDE.md > Non-negotiable safety rule 1) — the
opposite of what this node is for. A hand-authored, reviewed constant is
also the safest thing an emergency message can be: no risk of an LLM
drifting into reassurance, symptom analysis, or treatment suggestion
under time pressure, the same reasoning crisis_node's prompt has to fight
for with explicit "do not" instructions instead.

The clinical detail — why this fired — is not lost, just not put in
front of the patient: check_red_flags already populates
state["red_flags"] with each entry's "reason" (rule matches' reason_ar,
or the LLM's own reasoning for its entry). This node does not touch that
field; it simply persists as last-value state for the doctor report to
use once that node exists.

state["thread_outcome"] = "emergency": set unconditionally here, the
moment this node fires. Sticky — unlike state["stage"], nodes/reset_stage.py
never clears it — so a LATER turn that escalates nothing new is routed
by graph.py to reiterate_terminal_outcome instead of silently continuing
into normal symptom triage (CLAUDE.md > Non-negotiable safety rule 13).
See state.py's own field comment and
nodes/reiterate_terminal_outcome.py's module docstring for the full
reasoning.

state["severity"] = "emergency": also set unconditionally here — a red
flag firing IS an emergency-level clinical-severity judgment by
definition, so this is a direct, unambiguous mapping, not a new design
decision. This is the one place state["severity"] (previously a dormant
field nothing wrote) gets wired up; the low/moderate/high grading for
the normal, non-emergency diagnostic path remains unwired — see
CLAUDE.md > Known limitations. Contrast crisis_node, which deliberately
does NOT set severity — crisis is a different axis, not a point on this
scale at all.
"""

from __future__ import annotations

from typing import Any

from state import HealixState

_EMERGENCY_MESSAGE = (
    "الأعراض يلي ذكرتها بتستدعي تدخل طبي إسعافي فورًا. توجه حالًا لأقرب "
    "قسم طوارئ، أو اتصل بالإسعاف إذا ما قدرت توصل بسرعة. لا تأجل هاد "
    "الشي ولا تنتظر لبكرا."
)


def emergency_node(state: HealixState) -> dict[str, Any]:
    """Produce the red-flag-path response and mark this turn's stage."""
    return {
        "messages": [{"role": "assistant", "content": _EMERGENCY_MESSAGE}],
        "stage": "emergency",
        "thread_outcome": "emergency",
        "severity": "emergency",
    }
