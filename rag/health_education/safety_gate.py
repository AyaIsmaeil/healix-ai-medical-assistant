"""Routes a health-education question BEFORE any retrieval happens.

Per docs/AHD_DATA_PROVENANCE.md's binding constraints: this module never
declares an emergency itself and never diagnoses. It only decides where a
question belongs — educational answer, or a redirect toward the existing,
unmodified triage system (rules/crisis.py, rules/red_flags.py, and
ultimately POST /chat) for anything that looks personal or urgent.

Order (deterministic layers first, exactly mirroring nodes/crisis_check.py
and nodes/check_red_flags.py's own "rule-based never removed or made
conditional on an LLM" discipline):

1. rules.crisis.detect_crisis on the raw question — unmodified import,
   same phrase set POST /chat's crisis_check node uses.
2. A raw-text pass over rules.red_flags.RED_FLAG_RULES's own cited
   symptom combinations (e.g. "عندي ألم صدر شديد وضيق نفس" contains both
   acs_chest_pain's all_of and any_of terms) — reusing the exact same
   rule data and word-boundary matcher red_flags.py already validates
   its own symptom names against, applied here to raw free text instead
   of an extracted-symptom set (the same mechanism already used for
   medical_record_summary's chronic-condition keywords, which is also
   free text).
3. Only once both of the above find nothing: an LLM screen
   (schemas.health_education.HealthQuestionClassification) distinguishes
   an educational question from a personal-symptom report or a
   medication/dosage request. This LLM is never asked to detect an
   emergency — that already happened in steps 1-2.
"""

from __future__ import annotations

from llm_client import call_llm
from prompts.base import build_prompt
from rules.crisis import detect_crisis, normalize
from rules.red_flags import RED_FLAG_RULES, RedFlagRule, _compile_keyword_matcher
from schemas.health_education import HealthQuestionClassification

from api.health_qa_contracts import Category


def _rule_matches_text(rule: RedFlagRule, normalized_text: str) -> bool:
    """Same all_of/any_of semantics as SymptomRequirement.satisfied_by
    (rules/red_flags.py), applied to raw text via word-boundary substring
    matching instead of a confirmed-symptom set."""
    requirement = rule.requirement

    for term in requirement.all_of:
        matcher = _compile_keyword_matcher(frozenset({term}))
        if matcher is None or not matcher.search(normalized_text):
            return False

    if requirement.any_of:
        matcher = _compile_keyword_matcher(requirement.any_of)
        if matcher is None or not matcher.search(normalized_text):
            return False
    elif not requirement.all_of:
        return False

    return True


def _matches_a_red_flag_combination(question: str) -> bool:
    normalized = normalize(question)
    return any(_rule_matches_text(rule, normalized) for rule in RED_FLAG_RULES)


def classify(question: str, *, thread_id: str | None = None) -> Category:
    """Where this question belongs. Never raises for a "no match" case —
    only for a genuine LLM failure (llm_client.LLMError), same as any
    other LLM-backed node in this project."""
    if detect_crisis(question).matched:
        return "emergency_redirect"

    if _matches_a_red_flag_combination(question):
        return "emergency_redirect"

    prompt = build_prompt("health_qa_classify", question=question)
    llm_result = call_llm(
        prompt,
        schema=HealthQuestionClassification,
        tier="fast",
        prompt_name="health_qa_classify",
        thread_id=thread_id,
    )
    assert isinstance(llm_result, HealthQuestionClassification)

    if llm_result.category == "personal_symptom":
        return "triage_redirect"
    if llm_result.category == "medication_dosage":
        return "medication_safety"
    return "educational"
