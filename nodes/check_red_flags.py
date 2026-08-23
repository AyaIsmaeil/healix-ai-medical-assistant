
from __future__ import annotations

from typing import Any

from audit.logger import log_red_flag_detection
from llm_client import call_llm
from prompts.base import build_prompt
from rules.red_flags import check_red_flags as run_red_flag_rules
from rules.red_flags import find_incomplete_combination_candidates
from schemas.red_flags import RedFlagAssessment
from state import HealixState, RedFlag, SafetyDecision, Symptom

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def _format_symptom_names(symptoms: list[Symptom]) -> str:
    names = [name for symptom in symptoms if (name := symptom.get("name"))]
    if not names:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(f"- {name}" for name in names)


def _format_mentions(mentions: list[str]) -> str:
    if not mentions:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(f"- {mention}" for mention in mentions)


def _candidate_to_state_dict(candidate: Any) -> dict[str, Any]:
    # Plain dict, not the dataclass — checkpointed state must stay
    # JSON-serializable, hence sorted lists rather than frozensets.
    return {
        "rule_id": candidate.rule_id,
        "category": candidate.category,
        "reason_ar": candidate.reason_ar,
        "source": candidate.source,
        "matched_symptoms": sorted(candidate.matched_symptoms),
        "missing_any_of": sorted(candidate.missing_any_of),
    }


def check_red_flags(state: HealixState) -> dict[str, Any]:
    symptoms = state.get("symptoms", [])
    negated_symptoms = state.get("negated_symptoms", [])
    medical_record_summary = state.get("medical_record_summary", "")
    unmatched_mentions = state.get("unmatched_mentions", [])

    rule_matches = run_red_flag_rules(symptoms, medical_record_summary, negated_symptoms)
    incomplete_candidates = find_incomplete_combination_candidates(
        symptoms, medical_record_summary, negated_symptoms
    )
    unresolved_candidates = [c for c in incomplete_candidates if not c.rejected]

    prompt = build_prompt(
        "check_red_flags",
        symptoms=_format_symptom_names(symptoms),
        unmatched_mentions=_format_mentions(unmatched_mentions),
    )
    llm_result = call_llm(
        prompt,
        schema=RedFlagAssessment,
        tier="quality",
        prompt_name="check_red_flags",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(llm_result, RedFlagAssessment)

    red_flags: list[RedFlag] = [
        {"id": match.rule_id, "reason": match.reason_ar} for match in rule_matches
    ]

    safety_decision: SafetyDecision
    if red_flags:
        safety_decision = "HARD_EMERGENCY"
    elif unresolved_candidates:
        safety_decision = "NEEDS_CLARIFICATION"
    else:
        safety_decision = "NO_RED_FLAG"

    log_red_flag_detection(
        thread_id=state.get("thread_id"),
        rule_matched=bool(rule_matches),
        rule_ids=[match.rule_id for match in rule_matches],
        llm_matched=llm_result.potential_red_flag,
        llm_reasoning=llm_result.reasoning,
        combined=bool(red_flags),
        candidate_rule_ids=[c.rule_id for c in unresolved_candidates],
        safety_decision=safety_decision,
    )

    return {
        "red_flags": red_flags,
        "red_flag_candidates": [_candidate_to_state_dict(c) for c in unresolved_candidates],
        "safety_decision": safety_decision,
    }
