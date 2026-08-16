"""Pydantic model for check_red_flags' LLM-side output.

CLAUDE.md > Non-negotiable safety rule 3: red-flag detection is
rule-based first, LLM second, combined with OR. This schema is the LLM
half — rules/red_flags.py's deterministic half never goes through an LLM
schema at all, and is never weakened by what this returns.

Deliberately minimal, mirroring schemas.crisis.CrisisCheckResult: a
boolean the node ORs with the rule engine's result, plus free-text
reasoning for the audit trail only. No category enum: unlike the rule
engine's fixed, code-defined category set, this layer exists precisely
to catch what that fixed set doesn't model (including a symptom-like
phrase that never matched a canonical name at all) — constraining it to
the same categories would defeat the point of having it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RedFlagAssessment(BaseModel):
    has_red_flag: bool = Field(
        description=(
            "Whether anything in the symptoms — including the free-text "
            "ones that didn't match a canonical name — suggests a "
            "potential medical emergency needing immediate evaluation. "
            "This is a screen, not a diagnosis."
        )
    )
    reasoning: str | None = Field(
        default=None,
        description="One short clause explaining the concern, for the audit trail only.",
    )
