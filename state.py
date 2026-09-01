"""HealixState: the LangGraph state shared across all nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from audit.logger import log_malformed_output


Symptom = dict[str, Any]

RedFlag = dict[str, str]

CandidateDisease = dict[str, Any]


def _valid_name(symptom: Symptom) -> str | None:
    """Return symptom["name"], or None (and audit-log it) if it's missing/empty.

    LLM extraction output is not schema-validated before it reaches this
    reducer (schemas/ validates the raw LLM call, not what a graph merge
    step receives) — a malformed entry here must not raise mid-conversation.
    """
    name = symptom.get("name")
    if name in (None, ""):
        reason = "missing_name" if "name" not in symptom else "empty_name"
        log_malformed_output(node="merge_symptoms", reason=reason, payload=dict(symptom))
        return None
    return name


def merge_symptoms(existing: list[Symptom] | None, new: list[Symptom] | None) -> list[Symptom]:
    """Accumulating reducer for `symptoms` / `negated_symptoms`.

    Symptoms persist and merge across turns instead of being overwritten
    each time extract_symptoms runs (CLAUDE.md > State), deduplicated by
    the "name" key. Merging a symptom already present fills in whatever
    fields the new entry has rather than replacing it outright — a sparse
    later mention (e.g. just a name) never erases detail such as duration
    or severity gathered on an earlier turn. None/"" values on the new
    side are skipped for that reason. Entries missing a usable "name" are
    skipped and audit-logged rather than failing the turn.
    """
    merged: dict[str, Symptom] = {}
    order: list[str] = []

    for symptom in existing or []:
        name = _valid_name(symptom)
        if name is None:
            continue
        order.append(name)
        merged[name] = dict(symptom)

    for symptom in new or []:
        name = _valid_name(symptom)
        if name is None:
            continue
        if name not in merged:
            order.append(name)
            merged[name] = dict(symptom)
            continue
        current = merged[name]
        for key, value in symptom.items():
            if value not in (None, ""):
                current[key] = value

    return [merged[name] for name in order]


def merge_unmatched_mentions(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Accumulating reducer for `unmatched_mentions` — list, deduped by exact string.

    Unlike merge_symptoms, entries here are plain phrases, not records with
    optional sub-fields to fill in — so there is nothing to merge field by
    field, and exact-string dedup (no normalization, no fuzzy matching) is
    enough, preserving first-seen order across turns. See
    schemas.symptoms.SymptomExtraction.unmatched_mentions for what produces
    these: a symptom-like phrase the patient used that extract_symptoms
    recognized but could not map to vocabulary/symptoms.py, and refused to
    force a near-match name onto rather than drop.
    """
    seen: dict[str, None] = {}
    for mention in existing or []:
        seen.setdefault(mention, None)
    for mention in new or []:
        seen.setdefault(mention, None)
    return list(seen)


Severity = Literal["low", "moderate", "high", "emergency"]

PatientSex = Literal["male", "female"]

Stage = Literal["followup", "crisis", "emergency", "diagnosis"]

ThreadOutcome = Literal["crisis", "emergency"]

SafetyDecision = Literal["HARD_EMERGENCY", "NEEDS_CLARIFICATION", "NO_RED_FLAG"]


class HealixState(TypedDict):
   
    thread_id: str
    medical_record_summary: str  # filtered summary from Laravel, never a raw record dump
 
    patient_sex: PatientSex | None


    messages: Annotated[list[dict[str, str]], operator.add]

    symptoms: Annotated[list[Symptom], merge_symptoms]
    # Carries diagnostic weight equal to confirmed symptoms — not a lesser signal.
    negated_symptoms: Annotated[list[Symptom], merge_symptoms]
   
    unmatched_mentions: Annotated[list[str], merge_unmatched_mentions]

    # How many follow-up questions have been asked so far this differential
    turn_count: int

    is_crisis: bool
    # Confirmed / hard-emergency hits ONLY — populated exclusively from
    red_flags: list[RedFlag]
    # Cited combination rules whose all_of core is present but whose
    red_flag_candidates: list[dict[str, Any]]
    # This turn's explicit red-flag disposition — see SafetyDecision above
    safety_decision: SafetyDecision | None
    # Only new clinical information may change this — never a patient objection
  
    severity: Severity | None

    # Which terminal SAFETY outcome ("crisis" or "emergency") this thread
    thread_outcome: ThreadOutcome | None


    is_sufficient: bool
    # The question to ask next, Syrian colloquial Arabic. Consumed by.
    next_question: str | None
    # True when nodes.assess_sufficiency.MAX_FOLLOW_UP_QUESTIONS was
    information_limited: bool

    # rag_retrieve's ranked output (nodes/rag_retrieve.py) — each entry a
    candidate_diseases: list[CandidateDisease]
    diagnosis: dict | None  # ranked possibilities with uncertainty, or insufficient_information

    specialty: str | None

    specialty_laravel: str | None
    reports: dict | None  # {"patient": ..., "doctor": ...}

    stage: Stage | None
