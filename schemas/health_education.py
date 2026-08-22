"""Pydantic models for the Health Education Q&A module's own LLM calls.

Separate from schemas/red_flags.py, schemas/symptoms.py etc. — this
module belongs to rag/health_education/, not to the triage graph, and
must not be confused with or imported by nodes/. See
rag/health_education/safety_gate.py and service.py for how these are
used.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthQuestionClassification(BaseModel):
    """LLM screen for what kind of question this is, once the deterministic
    crisis/red-flag layers (rules.crisis, rules.red_flags) have already
    cleared it. This model is never authorized to declare an emergency —
    that is decided by the deterministic layers alone, before this schema
    is ever used (see safety_gate.classify)."""

    category: Literal["educational", "personal_symptom", "medication_dosage"] = Field(
        description=(
            "'educational': a general question about a health topic, not "
            "about the asker's own current condition (e.g. 'ما هو الربو؟'). "
            "'personal_symptom': the asker is describing their own current "
            "symptom(s) or complaint, even mild (e.g. 'عندي ضيق نفس'). "
            "'medication_dosage': the question asks for a medication name, "
            "dose, or prescription/treatment recommendation."
        )
    )


class HealthEducationAnswer(BaseModel):
    """One grounded educational summary, written from retrieved AHD
    excerpts only. The model must not add information beyond what the
    excerpts support — see prompts/templates/health_qa_answer.txt."""

    answer: str = Field(
        min_length=1,
        description=(
            "A short Arabic educational summary grounded in the provided "
            "excerpts. Never a diagnosis, dosage, or personalized "
            "treatment recommendation."
        ),
    )
