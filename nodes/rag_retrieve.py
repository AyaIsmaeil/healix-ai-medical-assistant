"""rag_retrieve: matches confirmed symptoms against rag/knowledge_base/
by set overlap, producing a ranked candidate list. No LLM call — this is
where the computed match score (matched / reference symptoms) that the
eventual diagnosis cites actually gets computed, never an LLM confidence.

Both sides are normalized before comparing (KB entries use natural
spelling, state["symptoms"] is already normalized).

Negation is counter-evidence, not just absence: net_matched = matched -
negated, floored at zero.

MIN_MATCHED_SYMPTOMS=2 is an absolute floor, not a percentage — a KB
entry with only 2 common/nonspecific symptoms (e.g. hypertension:
headache, dizziness) would otherwise let a single vague symptom score as
a strong match.

match_score is only comparable WITHIN one disease's own match, not
across different diseases, while the knowledge base stays this small and
uneven — a known limitation, not something this floor fixes.

Sex-specific gating (rag.schema.KnowledgeBaseEntry.applicable_sex): a
qualifying candidate whose entry is sex-restricted and patient_sex is
unknown is neither included nor excluded — it triggers a follow-up
question via the same next_question/turn_count mechanism
assess_sufficiency uses, not a new one. A confirmed mismatch excludes it
outright. turn_count is a second, coordinated writer here (see
assess_sufficiency's own docstring) — the two can never fire in the same
turn, since this node only runs when assess_sufficiency already said
sufficient. Once the shared budget is spent, this stops asking and sets
information_limited=True rather than guessing patient_sex from text.
"""

from __future__ import annotations

from typing import Any

from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS
from rag.schema import KnowledgeBaseEntry, load_all
from rules.crisis import normalize
from rules.red_flags import _SUBSUMES
from state import CandidateDisease, HealixState, PatientSex, Symptom

MIN_MATCHED_SYMPTOMS = 2

# Same generic-term -> specific-siblings mapping rules/red_flags.py
# applies to red-flag rules (reused DATA, not that module's private
# cache) — a KB entry listing generic "حمى" should still match a patient
# who described the specific "حمى مرتفعة مفاجئة".
_SUBSUMES_N: dict[str, frozenset[str]] = {
    normalize(generic): frozenset(normalize(specific) for specific in specifics)
    for generic, specifics in _SUBSUMES.items()
}


def _subsumed_evidence(kb_symptoms: frozenset[str], present: set[str]) -> dict[str, str]:
    """kb_symptom -> the patient's actual term satisfying it (direct or
    via a specific sibling) — reports what the patient really said, not
    the KB's generic wording."""
    evidence: dict[str, str] = {}
    for kb_symptom in kb_symptoms:
        if kb_symptom in present:
            evidence[kb_symptom] = kb_symptom
            continue
        siblings = _SUBSUMES_N.get(kb_symptom, frozenset()) & present
        if siblings:
            evidence[kb_symptom] = next(iter(siblings))
    return evidence

# Fixed, disease-silent — naming the candidate would announce a
# diagnosis before diagnose() has even run.
_SEX_CLARIFICATION_QUESTION = (
    "قبل ما نكمل، ممكن تحكيلي إذا كنت رجل أو امرأة؟ هيك بقدر أعطيك تقييم أدق."
)


def _normalized_names(symptoms: list[Symptom]) -> set[str]:
    return {normalize(symptom["name"]) for symptom in symptoms if symptom.get("name")}


def _match_entry(
    entry: KnowledgeBaseEntry,
    confirmed: set[str],
    negated: set[str],
    patient_sex: PatientSex | None,
) -> tuple[CandidateDisease | None, bool]:
    """One disease's match detail, or (None, needs_sex_clarification).

    The second value is True only when this entry would otherwise
    qualify but is sex-restricted and patient_sex is unknown. A confirmed
    sex mismatch returns (None, False) — excluded, not ambiguous.
    """
    kb_symptoms = frozenset(normalize(name) for name in entry.symptoms)

    confirmed_evidence = _subsumed_evidence(kb_symptoms, confirmed)
    if not confirmed_evidence:
        return None, False

    negated_evidence = _subsumed_evidence(kb_symptoms, negated)
    net_matched = max(0, len(confirmed_evidence) - len(negated_evidence))
    if net_matched < MIN_MATCHED_SYMPTOMS:
        return None, False

    if entry.applicable_sex is not None:
        if patient_sex is None:
            return None, True  # would qualify, but sex-restricted and unconfirmed
        if patient_sex != entry.applicable_sex:
            return None, False  # confirmed mismatch — excluded outright, not ambiguous

    missing = kb_symptoms - confirmed_evidence.keys() - negated_evidence.keys()

    return {
        "name": entry.name,
        "name_ar": entry.name_ar,
        "match_score": round(net_matched / len(kb_symptoms), 2),
        "matched_symptoms": sorted(confirmed_evidence.values()),
        "missing_symptoms": sorted(missing),
        "negated_symptoms": sorted(negated_evidence.values()),
        # name_ar/specialties carried forward for route_specialty/generate_reports.
        "specialties": list(entry.specialties),
    }, False


def rag_retrieve(state: HealixState) -> dict[str, Any]:
    confirmed = _normalized_names(state.get("symptoms", []))
    negated = _normalized_names(state.get("negated_symptoms", []))
    patient_sex = state.get("patient_sex")

    candidates: list[CandidateDisease] = []
    needs_sex_clarification = False
    for entry in load_all():
        match, ambiguous = _match_entry(entry, confirmed, negated, patient_sex)
        if match is not None:
            candidates.append(match)
        if ambiguous:
            needs_sex_clarification = True

    candidates.sort(key=lambda candidate: candidate["match_score"], reverse=True)

    if needs_sex_clarification:
        turn_count = state.get("turn_count", 0)
        if turn_count < MAX_FOLLOW_UP_QUESTIONS:
            return {
                "candidate_diseases": candidates,
                "next_question": _SEX_CLARIFICATION_QUESTION,
                "turn_count": turn_count + 1,
            }
        return {"candidate_diseases": candidates, "information_limited": True}

    return {"candidate_diseases": candidates}
