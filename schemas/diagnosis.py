"""Pydantic model for diagnose's LLM output.

Built dynamically, unlike every other schema in this package: which
disease names are even valid depends on THIS TURN's
state["candidate_diseases"] (from rag_retrieve), so the enum has to be
constructed fresh per call from that live candidate set.
build_diagnosis_schema(candidate_names) is the entry point
nodes/diagnose.py uses, in place of importing one fixed class the way
every other node imports e.g. schemas.symptoms.SymptomExtraction.

CLAUDE.md > Non-negotiable safety rule 6: diagnose may only select from
RAG-retrieved candidate diseases, never free-generate a name. The
per-call Literal enum below is what makes that structural, not a prompt
instruction the model could ignore — same mechanism as
schemas.symptoms.SymptomName, just built at call time instead of import
time, because the valid-value set itself changes every turn (and even in
size — see nodes/diagnose.py's own docstring on staying correct at any
knowledge-base size).

status="insufficient_information" is a first-class, valid outcome
(safety rule 6's own text), not an error path — differential stays
empty in that case. This schema does not carry a numeric confidence at
all (safety rule 7); nor does it carry a qualitative certainty level —
that is derived in nodes/diagnose.py from the already-computed
match_score, not asked of the model, for the same reason
nodes/rag_retrieve.py computes match_score in code rather than asking an
LLM to estimate it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from audit.logger import log_malformed_output


def build_diagnosis_schema(candidate_names: tuple[str, ...]) -> type[BaseModel]:
    """Construct this turn's DiagnosisAssessment, enum-constrained to candidate_names.

    candidate_names must be non-empty. nodes.diagnose.diagnose never calls
    this for an empty candidate_diseases list — it short-circuits to
    insufficient_information in code instead, without any LLM call, since
    a zero-length Literal has no valid values for a provider's structured
    output to enforce at all.
    """
    if not candidate_names:
        raise ValueError("build_diagnosis_schema requires at least one candidate name.")

    DiseaseName = Literal[candidate_names]  # type: ignore[valid-type]

    class DiagnosisAssessment(BaseModel):
        """One call's verdict: which of the given candidates, if any, form
        a differential worth presenting.
        """

        status: Literal["differential", "insufficient_information"] = Field(
            description=(
                "'differential' if any retrieved candidates form a clinically "
                "sensible, if uncertain, list worth presenting to the patient's "
                "doctor; 'insufficient_information' if none of them do — this "
                "is a normal, expected outcome, not a failure."
            )
        )
        differential: list[DiseaseName] = Field(
            default_factory=list,
            description=(
                "Which of the candidate diseases given to you belong in the "
                "differential. Any order — the final presented order is "
                "re-ranked afterward by the already-computed match score, not "
                "by the order you list them in. Empty when status is "
                "insufficient_information. Every entry must be exactly one of "
                "the candidate names given to you, spelled exactly as given — "
                "never a disease that is not in that list, no matter how well "
                "it seems to fit."
            ),
        )
        reasoning: str | None = Field(
            default=None,
            description=(
                "One short clause explaining the overall verdict, for the "
                "audit trail only."
            ),
        )

        @model_validator(mode="after")
        def _keep_differential_consistent_with_status(self) -> "DiagnosisAssessment":
            if self.status == "insufficient_information" and self.differential:
                log_malformed_output(
                    node="DiagnosisAssessment",
                    reason="differential_present_while_insufficient",
                    payload={"differential": list(self.differential)},
                )
                self.differential = []
            elif self.status == "differential" and not self.differential:
                log_malformed_output(
                    node="DiagnosisAssessment",
                    reason="differential_status_with_no_candidates_selected",
                    payload={},
                )
                self.status = "insufficient_information"
            return self

    return DiagnosisAssessment
