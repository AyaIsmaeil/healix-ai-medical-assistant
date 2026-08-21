"""Pydantic models for the LLM half of red-flag screening and verification.

The deterministic layer (rules/red_flags.py) never goes through an LLM
schema. The LLM may surface a *potential* concern the rule engine cannot
see (unmatched mentions). It is not authorized to confirm an emergency
or to invent verification criteria — disposition is decided in
nodes/check_red_flags.py from cited rules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RedFlagAssessment(BaseModel):
    potential_red_flag: bool = Field(
        description=(
            "Whether anything in the listed symptoms or unmatched mentions "
            "is a *candidate* red-flag concern that the deterministic rules "
            "might have missed. This is a screen, not an emergency decision "
            "and not a diagnosis. Isolated common symptoms (e.g. chest pain "
            "or fever alone) may be potential concerns; they are not "
            "confirmed emergencies."
        )
    )
    reasoning: str | None = Field(
        default=None,
        description="One short clause explaining the concern, for the audit trail only.",
    )


class VerificationQuestion(BaseModel):
    question: str = Field(
        description=(
            "One Syrian-colloquial Arabic question that asks ONLY about the "
            "listed missing information items. Do not add other medical "
            "questions, diagnoses, or advice."
        ),
        min_length=1,
    )
