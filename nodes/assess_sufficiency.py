"""assess_sufficiency: decides whether the accumulated symptom picture is
enough for a preliminary differential, or whether one more follow-up
question would meaningfully narrow it. Runs once a turn has no crisis
and no red flag. No deterministic rule layer here — the LLM's judgment
is the whole verdict (schemas.sufficiency.SufficiencyAssessment).

turn_count enforces a hard ceiling (MAX_FOLLOW_UP_QUESTIONS) on
follow-up questions per differential cycle. Past the ceiling, this node
skips the LLM call entirely and proceeds. nodes.rag_retrieve is the only
other writer of turn_count (its own sex-clarification question) — the
two never fire in the same turn, since rag_retrieve only runs on this
node's is_sufficient=True branch, which doesn't increment.
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from prompts.base import build_prompt
from schemas.sufficiency import SufficiencyAssessment
from state import HealixState, Symptom

MAX_FOLLOW_UP_QUESTIONS = 6

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def _format_symptom(symptom: Symptom) -> str:
    name = symptom.get("name", "")
    details = []
    if symptom.get("raw_mention"):
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
    turn_count = state.get("turn_count", 0)

    if turn_count >= MAX_FOLLOW_UP_QUESTIONS:
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
