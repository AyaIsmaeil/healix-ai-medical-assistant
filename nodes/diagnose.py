"""diagnose: ranks state["candidate_diseases"] (from rag_retrieve) into a
differential, or returns insufficient_information (CLAUDE.md > Graph flow:
runs after ml_corroborate, before route_specialty — wired in graph.py as
ml_corroborate -> diagnose -> route_specialty -> generate_reports).

Reads only state["candidate_diseases"] — not the raw symptom lists or
patient message. rag_retrieve already did the symptom-matching; this
node's job is judging the ALREADY-COMPUTED candidate set as a whole, not
re-deriving it.

Calls the LLM on the "quality" tier — CLAUDE.md > Non-negotiable safety
rule 12 names this node explicitly among those reading/interpreting
clinically-relevant content to decide something safety-relevant, not
just crisis/red-flag detection.

--- Safety rule 6: the model may only select from the RAG-retrieved
candidates, never free-generate a disease name. Enforced structurally,
not by prompt instruction: schemas.diagnosis.build_diagnosis_schema
constructs a Literal enum from THIS TURN's candidate names and passes it
to call_llm's native schema parameter — same mechanism as
schemas.symptoms.SymptomName, just built fresh per call instead of once
at import time, because the valid-value set itself changes every turn.
insufficient_information is a first-class, valid outcome of this
judgment (safety rule 6's own text) — see the empty-candidates
short-circuit and the schema's own status field.

--- Safety rule 7: no numeric confidence, generated or otherwise. The
LLM's output carries no score of any kind — ranking uses the match_score
rag_retrieve already computed (never re-derived or second-guessed here),
and the qualitative certainty band (high/medium/low) attached to each
differential entry is COMPUTED IN CODE from that same match_score
(_certainty_band below), never asked of the model. This mirrors why
rag_retrieve computes match_score in code rather than asking an LLM to
estimate it — the same principle applied one layer further down the
pipeline.

--- CLAUDE.md > Known limitations: match_score is only meaningfully
comparable WITHIN one disease's own match, not across different
diseases, while the knowledge base stays this small and uneven in
symptom-list length. The prompt (prompts/templates/diagnose.txt) tells
the model this explicitly, so its inclusion/exclusion judgment does not
silently treat the score as a precise cross-disease ranking it isn't yet.

--- Two named caps, both deliberately independent of the knowledge
base's CURRENT size (10 diseases) — CLAUDE.md > Working style: this node
must behave correctly at any knowledge-base size without changes, which
requires bounding its LLM-facing surface area by something other than
"however many happen to exist today":

  * MAX_CANDIDATES_CONSIDERED bounds how many of rag_retrieve's
    (already match_score-sorted) candidates are even shown to the model
    / included in the per-call enum. Without this, a much larger future
    knowledge base could in principle hand this node a very large
    candidate list, growing both the prompt and the enum unboundedly.
  * MAX_DIFFERENTIAL_SIZE bounds how many candidates the FINAL,
    code-sorted differential presents, after the model's own selection.
    A real differential diagnosis is a short, clinically useful list
    regardless of how many total candidates a large knowledge base could
    theoretically retrieve — this is a clinical-presentation choice, not
    a knowledge-base-size workaround, so it applies identically whether
    the knowledge base has 10 entries or 10,000.

Both are named constants for exactly the reason every other threshold in
this project is: one place to change, never a magic number repeated
inline. Neither is tuned to "what happens to work for the current 10
entries" — MAX_CANDIDATES_CONSIDERED=10 and MAX_DIFFERENTIAL_SIZE=5 would
behave identically at any knowledge-base size; the current knowledge
base just happens to rarely produce enough candidates to make either cap
bind in practice yet. Tests against the real 10-entry knowledge base
verify this node's LOGIC, not a size-10-specific behavior — nothing here
assumes exactly 10 entries exist.

Final differential order is always code-sorted by match_score,
descending, regardless of what order the model listed its selections
in — ranking itself is never left to the model (see safety rule 7 note
above).
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from prompts.base import build_prompt
from schemas.diagnosis import build_diagnosis_schema
from state import CandidateDisease, HealixState

# See module docstring's "Two named caps" section for why these are
# absolute counts independent of the knowledge base's current size.
MAX_CANDIDATES_CONSIDERED = 10
MAX_DIFFERENTIAL_SIZE = 5

# Certainty bands, derived from match_score in code (never asked of the
# model — see module docstring's safety rule 7 note). Chosen against the
# achievable score range given rag_retrieve.MIN_MATCHED_SYMPTOMS=2: a
# large reference symptom list (e.g. influenza's 7) can legitimately
# clear that floor as low as ~0.29, so "low" starts below the midpoint
# rather than requiring near-zero to ever apply.
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
    """Judge state["candidate_diseases"] into a ranked differential, or insufficient_information."""
    candidates = state.get("candidate_diseases", [])

    if not candidates:
        # Nothing to rank and nothing for a schema enum to enforce at all
        # (a zero-length Literal is not constructible) — code-level
        # short-circuit, no LLM call, same pattern as
        # assess_sufficiency's ceiling short-circuit.
        return {"diagnosis": dict(_INSUFFICIENT_INFORMATION)}

    # Already sorted descending by match_score (nodes.rag_retrieve) —
    # taking a prefix keeps the highest-scoring candidates.
    considered = candidates[:MAX_CANDIDATES_CONSIDERED]
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
            # Carried through unchanged when nodes.ml_corroborate set it —
            # never shown to the LLM above (the prompt/schema in this
            # node are untouched by that field), never a new candidate,
            # never a numeric value. See CLAUDE.md's XGBoost
            # corroboration-signal section.
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
