"""Pydantic models for the two crisis-path LLM outputs: detection
(crisis_check) and response (crisis_node).

CLAUDE.md > Non-negotiable safety rule 9: on any sign of acute
psychological distress or self-harm, analysis stops and only the crisis
marker is emitted. Both schemas below exist to hold the LLM to that —
neither has any field for a diagnosis, a symptom, or anything derived
from analyzing what the patient said.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CrisisCheckResult(BaseModel):
    """crisis_check's LLM-side detector.

    The deterministic half (rules/crisis.py) matches raw text and never
    goes through an LLM schema at all. Combined with it by OR, never by
    AND or as a tie-breaker (safety rule 3): is_crisis=False here never
    overrides a rule-layer match, and is_crisis=True here is never held
    back waiting for rule-layer agreement. See nodes/crisis_check.py.
    """

    is_crisis: bool = Field(
        description=(
            "Whether the message shows signs of acute psychological distress, "
            "suicidal ideation, or self-harm intent."
        )
    )
    reasoning: str | None = Field(
        default=None,
        description="One short clause explaining the signal, for the audit trail only.",
    )


class CrisisResponse(BaseModel):
    """crisis_node's LLM-generated acknowledgment.

    Deliberately one field: there is nothing else for the model to
    contribute here. In particular there is no phone-number-shaped field —
    support-line numbers come from support_lines.py, spliced in by code
    after this call returns, never written by the model (CLAUDE.md >
    Non-negotiable safety rules: a model-generated helpline number is
    unacceptable). See nodes/crisis_node.py and
    prompts/templates/crisis_node.txt for what this message may and may
    not contain.
    """

    message: str = Field(
        description=(
            "A single short, calm acknowledgment in Syrian colloquial Arabic. "
            "No analysis, no diagnosis, no reassurance, no follow-up question, "
            "no coping suggestion, and no phone number, hotline, or "
            "organization name."
        )
    )
