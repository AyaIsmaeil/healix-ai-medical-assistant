"""Pydantic model for assess_sufficiency's LLM output.

CLAUDE.md > Graph flow: assess_sufficiency judges whether the accumulated
symptom picture is enough to attempt a differential, or whether one more
follow-up question would meaningfully narrow it. Unlike crisis_check and
check_red_flags, there is no deterministic rule layer here to OR against —
"is this enough information" is not a fixed-pattern match, so the LLM's
judgment is the whole verdict for this turn.

is_sufficient and next_question are coupled: a question only makes sense
when the picture is insufficient, and the picture being sufficient leaves
nothing to ask. The validator below enforces that coupling itself rather
than trusting the model to keep the two consistent, and repairs a
malformed response instead of failing the turn over it — same pattern as
schemas.symptoms.SymptomExtraction._dedupe_by_name: a formatting slip here
is not a safety violation worth crashing a medical turn over.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from audit.logger import log_malformed_output

# Deliberately generic and non-diagnostic: this is what the model falls
# back to when it says "insufficient" but forgets to write the question
# itself, so the patient still gets asked *something* sensible rather than
# the turn producing an empty follow-up. Syrian colloquial Arabic, matching
# CLAUDE.md > Language.
FALLBACK_QUESTION = "ممكن تحكيلي أكتر عن الأعراض يلي عندك، ومتى بلشت بالضبط؟"


class SufficiencyAssessment(BaseModel):
    """assess_sufficiency's LLM-side judgment.

    No category enum, no score: CLAUDE.md > Non-negotiable safety rule 7
    keeps numeric confidence to the RAG match score, and this call happens
    before RAG retrieval even runs.
    """

    is_sufficient: bool = Field(
        description=(
            "Whether the confirmed symptoms, negated symptoms, and "
            "unmatched mentions gathered so far are enough to attempt a "
            "preliminary differential — not a diagnosis, just enough "
            "signal to retrieve and rank candidate conditions."
        )
    )
    next_question: str | None = Field(
        default=None,
        description=(
            "Required when is_sufficient is false, otherwise omit it. One "
            "specific, focused question in Syrian colloquial Arabic that "
            "would most narrow down the differential — never a generic "
            "'tell me more', never more than one question at once. Open "
            "with a brief one-clause acknowledgment of what the patient "
            "just described — e.g. reflecting back the main symptom just "
            "mentioned — before the question itself, so it reads as part "
            "of a conversation rather than an interrogation; the "
            "acknowledgment must not restate the full symptom list or "
            "turn into a recap."
        ),
    )
    reasoning: str | None = Field(
        default=None,
        description="One short clause explaining the verdict, for the audit trail only.",
    )

    @model_validator(mode="after")
    def _keep_question_consistent_with_verdict(self) -> "SufficiencyAssessment":
        if self.is_sufficient:
            if self.next_question:
                log_malformed_output(
                    node="SufficiencyAssessment",
                    reason="next_question_present_while_sufficient",
                    payload={"next_question": self.next_question},
                )
            self.next_question = None
        elif not self.next_question or not self.next_question.strip():
            log_malformed_output(
                node="SufficiencyAssessment",
                reason="missing_next_question_while_insufficient",
                payload={},
            )
            self.next_question = FALLBACK_QUESTION
        return self
