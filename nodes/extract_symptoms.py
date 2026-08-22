"""extract_symptoms: reads the patient's latest message for confirmed and
negated symptoms, plus unmatched_mentions (symptom-like phrases with no
canonical vocabulary match yet — dropped otherwise).

The prompt also gets one turn of context: the previous assistant question
and the already-known symptoms. Without this, a bare follow-up answer
("من الصبح ومستمر") has no referent and the model returns nothing useful.

Negation: rule-based (rules.negation, raw message) combined with the
LLM's own negated_symptoms via OR — the rule layer only adds names the
LLM missed.

Duration: each confirmed symptom's free-text `duration` is also parsed
into `duration_days` (vocabulary.duration.parse_duration_days) as an
added field, never replacing the original text; None when unparseable.
"""

from __future__ import annotations

from typing import Any

from audit.logger import log_negation_detection
from llm_client import call_llm
from nodes._shared import latest_user_message, previous_assistant_message
from prompts.base import build_prompt
from rules.crisis import normalize
from rules.negation import detect_negated_symptoms, merge_negations
from schemas.symptoms import ExtractedSymptom, NegatedSymptom, SymptomExtraction
from state import HealixState, Symptom
from vocabulary.duration import parse_duration_days

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def _format_known_symptoms(symptoms: list[Symptom]) -> str:
    if not symptoms:
        return _NO_ENTRIES_PLACEHOLDER
    lines = []
    for symptom in symptoms:
        name = symptom.get("name", "")
        details = [
            f"{field}={symptom[field]}"
            for field in ("raw_mention", "duration", "severity", "onset")
            if symptom.get(field)
        ]
        lines.append(f"- {name} ({', '.join(details)})" if details else f"- {name}")
    return "\n".join(lines)


def _with_duration_days(symptom: ExtractedSymptom) -> dict[str, Any]:
    data = symptom.model_dump()
    data["duration_days"] = parse_duration_days(symptom.duration) if symptom.duration else None
    return data


def _merge_negated_symptoms(
    llm_negated: list[NegatedSymptom], message: str, thread_id: str | None
) -> list[Symptom]:
    """OR-combine the LLM's negated_symptoms with the rule layer. A
    rule-only name is appended as a bare {"name": ...} (no raw_mention —
    that layer has no free-text capture); an LLM-only name is never
    dropped just because the rule layer missed it too."""
    llm_entries = [symptom.model_dump() for symptom in llm_negated]
    llm_names = {normalize(entry["name"]) for entry in llm_entries}

    rule_names = detect_negated_symptoms(message)
    combined_names = merge_negations(llm_names, rule_names)

    log_negation_detection(
        thread_id=thread_id,
        rule_negated=sorted(rule_names),
        llm_negated=sorted(llm_names),
        combined=sorted(combined_names),
    )

    for name in combined_names:
        if name not in llm_names:
            llm_entries.append({"name": name})

    return llm_entries


def extract_symptoms(state: HealixState) -> dict[str, Any]:
    message = latest_user_message(state)
    previous_question = previous_assistant_message(state)
    known_symptoms = state.get("symptoms", [])

    prompt = build_prompt(
        "extract_symptoms",
        message=message,
        previous_question=previous_question or _NO_ENTRIES_PLACEHOLDER,
        known_symptoms=_format_known_symptoms(known_symptoms),
    )
    result = call_llm(
        prompt,
        schema=SymptomExtraction,
        tier="quality",
        prompt_name="extract_symptoms",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(result, SymptomExtraction)

    return {
        "symptoms": [_with_duration_days(symptom) for symptom in result.symptoms],
        "negated_symptoms": _merge_negated_symptoms(
            result.negated_symptoms, message, state.get("thread_id")
        ),
        "unmatched_mentions": result.unmatched_mentions,
    }
