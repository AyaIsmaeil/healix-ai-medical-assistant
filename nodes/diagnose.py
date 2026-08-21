"""diagnose: ranks state["candidate_diseases"] (from rag_retrieve) into a
differential, or returns insufficient_information. Reads only the
already-computed candidate set, never re-derives it from raw symptoms.

The model may only select from the RAG-retrieved candidates, never
free-generate a name — enforced structurally via a per-turn Literal enum
(schemas.diagnosis.build_diagnosis_schema), not a prompt instruction.

No numeric confidence from the LLM: ranking uses rag_retrieve's own
match_score, and the certainty band (high/medium/low) is computed in
code from it (_certainty_band), never asked of the model.

MAX_CANDIDATES_CONSIDERED / MAX_DIFFERENTIAL_SIZE are absolute caps,
independent of the knowledge base's current size — bound the LLM-facing
surface area and the final differential's length regardless of how large
the KB grows.

Final order is always code-sorted by match_score, never the model's own
listed order.
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from prompts.base import build_prompt
from schemas.diagnosis import build_diagnosis_schema
from state import CandidateDisease, HealixState

MAX_CANDIDATES_CONSIDERED = 10
MAX_DIFFERENTIAL_SIZE = 5

# Certainty bands derived from match_score in code, never asked of the
# model. "low" starts below the midpoint since a large reference symptom
# list can legitimately clear MIN_MATCHED_SYMPTOMS as low as ~0.29.
_HIGH_CERTAINTY_THRESHOLD = 0.7
_MEDIUM_CERTAINTY_THRESHOLD = 0.4

_INSUFFICIENT_INFORMATION: dict[str, Any] = {
    "status": "insufficient_information",
    "differential": [],
    "reasoning": None,
}


def _certainty_band(match_score: float) -> str:
    if match_score >= _HIGH_CERTAINTY_THRESHOLD:
        return "high"
    if match_score >= _MEDIUM_CERTAINTY_THRESHOLD:
        return "medium"
    return "low"


def _format_candidate(candidate: CandidateDisease) -> str:
    matched = "، ".join(candidate["matched_symptoms"]) or "لا يوجد"
    missing = "، ".join(candidate["missing_symptoms"]) or "لا يوجد"
    negated = "، ".join(candidate["negated_symptoms"]) or "لا يوجد"
    return (
        f"- {candidate['name']} (match_score={candidate['match_score']})\n"
        f"  matched: {matched}\n"
        f"  missing: {missing}\n"
        f"  negated: {negated}"
    )


def _format_candidates(candidates: list[CandidateDisease]) -> str:
    return "\n".join(_format_candidate(candidate) for candidate in candidates)


def diagnose(state: HealixState) -> dict[str, Any]:
    candidates = state.get("candidate_diseases", [])

    if not candidates:
        # No candidates -> no LLM call, no schema enum to build.
        return {"diagnosis": dict(_INSUFFICIENT_INFORMATION)}

    considered = candidates[:MAX_CANDIDATES_CONSIDERED]  # already match_score-sorted
    by_name = {candidate["name"]: candidate for candidate in considered}
    candidate_names = tuple(by_name)

    schema = build_diagnosis_schema(candidate_names)
    prompt = build_prompt("diagnose", candidates=_format_candidates(considered))
    result = call_llm(
        prompt,
        schema=schema,
        tier="quality",
        prompt_name="diagnose",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(result, schema)

    if result.status == "insufficient_information":
        return {
            "diagnosis": {
                "status": "insufficient_information",
                "differential": [],
                "reasoning": result.reasoning,
            }
        }

    selected_names: list[str] = []
    seen: set[str] = set()
    for name in result.differential:
        if name not in seen:
            seen.add(name)
            selected_names.append(name)

    differential = [
        {
            "name": name,
            "name_ar": by_name[name]["name_ar"],
            "match_score": by_name[name]["match_score"],
            "certainty": _certainty_band(by_name[name]["match_score"]),
            "matched_symptoms": by_name[name]["matched_symptoms"],
            "missing_symptoms": by_name[name]["missing_symptoms"],
            "negated_symptoms": by_name[name]["negated_symptoms"],
            "specialties": by_name[name]["specialties"],
            # Carried through unchanged when ml_corroborate set it; never shown to the LLM.
            **(
                {"ml_corroboration": by_name[name]["ml_corroboration"]}
                if "ml_corroboration" in by_name[name]
                else {}
            ),
        }
        for name in selected_names
    ]
    differential.sort(key=lambda entry: entry["match_score"], reverse=True)
    differential = differential[:MAX_DIFFERENTIAL_SIZE]

    return {
        "diagnosis": {
            "status": "differential",
            "differential": differential,
            "reasoning": result.reasoning,
        }
    }
