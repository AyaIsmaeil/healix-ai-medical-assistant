"""assess_sufficiency: decides whether the accumulated symptom picture is
enough to attempt a preliminary differential, or whether one more
follow-up question would meaningfully narrow it down (CLAUDE.md > Graph
flow: runs after check_red_flags, once a turn has no crisis and no red
flag).

Calls the LLM on the "quality" tier — CLAUDE.md > Non-negotiable safety
rule 12: this reads state["symptoms"] / state["negated_symptoms"] /
state["unmatched_mentions"], all derived from the patient's raw Arabic
message, to decide something that shapes the rest of the turn.

Unlike crisis_check and check_red_flags there is no deterministic rule
layer to OR against here — "is this enough information" isn't a
fixed-pattern match, so the LLM's judgment (schemas.sufficiency.
SufficiencyAssessment) is the whole verdict.

Hard ceiling (CLAUDE.md > State): state["turn_count"] counts follow-up
questions already asked this differential cycle. This node increments
turn_count by exactly one at the moment IT decides a follow-up is
needed, not when ask_followup later phrases it. When turn_count has
already reached MAX_FOLLOW_UP_QUESTIONS, this node short-circuits before
calling the LLM at all: the ceiling is a code-enforced fact, not
something left for the model to respect, and skipping the call also
means this node's own decision never has a chance to accidentally
increment turn_count past the ceiling.

--- turn_count has a SECOND writer, nodes.rag_retrieve — a deliberate,
coordinated exception to what was originally a strict single-writer
rule (CLAUDE.md > State), not a drift risk reintroduced by accident.
Both writers represent the exact same conceptual event — "a follow-up
question was asked this turn" — just triggered by two different gaps
(this node: the symptom picture itself; rag_retrieve: an unconfirmed
patient_sex blocking a sex-restricted candidate). Both check the SAME
MAX_FOLLOW_UP_QUESTIONS ceiling before incrementing, so the two stay
coordinated against one shared budget rather than drifting into two
independent counters.

The two writers' increment branches cannot both fire in the same turn —
not by convention, but because graph.py's routing makes it structurally
unreachable: this node's own increment happens ONLY on the
is_sufficient=False branch below, and that exact branch is what routes
the turn to ask_followup and ENDS it — rag_retrieve is never even
invoked that turn. rag_retrieve, in turn, only ever runs when THIS
node's verdict was is_sufficient=True this same turn, which is
precisely the branch that does NOT increment turn_count here. The two
increment sites sit on two mutually exclusive branches of the same
boolean decision, so at most one of them can execute per turn — proven
by graph.py's own conditional edges, not just asserted here. See
nodes/rag_retrieve.py's own module docstring for the mirror of this
same argument from its side, and tests/unit/test_graph.py's
turn_count-related coverage for the test-level proof.

This node does not itself route (CLAUDE.md > Working style: one node at a
time) — graph.py's conditional edge reads state["is_sufficient"] to
choose between ask_followup and rag_retrieve. Nor does it touch
state["stage"]: this node never ends a turn on its own: ask_followup
does that when the question is actually sent to the patient.
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from prompts.base import build_prompt
from schemas.sufficiency import SufficiencyAssessment
from state import HealixState, Symptom

# Named per CLAUDE.md > State: "turn_count enforces a hard ceiling of 6
# follow-up questions." A named constant, not a magic number, so the limit
# has exactly one place to change.
MAX_FOLLOW_UP_QUESTIONS = 6

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def _format_symptom(symptom: Symptom) -> str:
    name = symptom.get("name", "")
    details = []
    if symptom.get("raw_mention"):
        # The patient's own wording for this symptom, verbatim — where a
        # spontaneous elaboration on an already-named symptom (location,
        # quality, anything not captured by the three structured fields
        # below) actually lands, per extract_symptoms.txt. Dropping this
        # was a real gap: assess_sufficiency could re-ask for detail the
        # patient already gave in their own words, just because it wasn't
        # in duration/severity/onset.
        details.append(f"raw_mention={symptom['raw_mention']}")
    if symptom.get("duration"):
        details.append(f"duration={symptom['duration']}")
    if symptom.get("severity"):
        details.append(f"severity={symptom['severity']}")
    if symptom.get("onset"):
        details.append(f"onset={symptom['onset']}")
    if details:
        return f"- {name} ({', '.join(details)})"
    return f"- {name}"


def _format_symptoms(symptoms: list[Symptom]) -> str:
    if not symptoms:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(_format_symptom(symptom) for symptom in symptoms)


def _format_mentions(mentions: list[str]) -> str:
    if not mentions:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(f"- {mention}" for mention in mentions)


def assess_sufficiency(state: HealixState) -> dict[str, Any]:
    """Judge whether the accumulated symptom picture is ready for a differential."""
    turn_count = state.get("turn_count", 0)

    if turn_count >= MAX_FOLLOW_UP_QUESTIONS:
        # The ceiling is already spent — proceed regardless of what the
        # model would say, without asking it at all. turn_count is not
        # incremented further: no new question is being asked.
        return {
            "is_sufficient": True,
            "next_question": None,
            "information_limited": True,
        }

    symptoms = state.get("symptoms", [])
    negated_symptoms = state.get("negated_symptoms", [])
    unmatched_mentions = state.get("unmatched_mentions", [])
    medical_record_summary = state.get("medical_record_summary", "")

    prompt = build_prompt(
        "assess_sufficiency",
        symptoms=_format_symptoms(symptoms),
        negated_symptoms=_format_symptoms(negated_symptoms),
        unmatched_mentions=_format_mentions(unmatched_mentions),
        medical_record_summary=medical_record_summary or _NO_ENTRIES_PLACEHOLDER,
    )
    result = call_llm(
        prompt,
        schema=SufficiencyAssessment,
        tier="quality",
        prompt_name="assess_sufficiency",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(result, SufficiencyAssessment)

    if result.is_sufficient:
        return {
            "is_sufficient": True,
            "next_question": None,
            "information_limited": False,
        }

    return {
        "is_sufficient": False,
        "next_question": result.next_question,
        "information_limited": False,
        "turn_count": turn_count + 1,
    }
