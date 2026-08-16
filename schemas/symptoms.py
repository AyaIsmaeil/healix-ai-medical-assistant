"""Pydantic model for extract_symptoms output.

CLAUDE.md > Symptom vocabulary: symptom names are constrained here, at the
schema layer, so the provider's native structured output makes it
*impossible* for the model to return a name outside the canonical list.
That is what makes the exact-string set membership in rules/red_flags.py
safe — without it, a hallucinated near-miss name would parse fine and then
silently fail to match any red-flag rule.

The constraint is generated from vocabulary/symptoms.py at import time.
Adding or correcting a vocabulary entry requires no edit to this file; a
hand-copied second list is exactly the drift this design removes.

Because `name` is required on ExtractedSymptom/NegatedSymptom, a symptom
that IS real but ISN'T in the vocabulary has no valid slot: the model
cannot invent a name (rejected by the enum) and cannot attach the mention
to a wrong-but-valid name either (raw_mention only exists alongside a real
`name`). Observed directly, not assumed: given "طنين بالأذن" (tinnitus,
not yet in the vocabulary), the model correctly refused to force a
near-match name — and the mention vanished, because there was nowhere to
put it. SymptomExtraction.unmatched_mentions below is that missing slot.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from audit.logger import log_malformed_output
from vocabulary.severity import SymptomSeverity
from vocabulary.symptoms import CANONICAL_SYMPTOMS

# Sorted for a stable JSON Schema: the generated enum ordering should not
# churn between runs, or every prompt-version diff becomes unreadable.
SYMPTOM_NAMES: tuple[str, ...] = tuple(sorted(CANONICAL_SYMPTOMS))

# Runtime tuple-subscript: Literal[("a", "b")] is Literal["a", "b"]. Static
# type checkers cannot follow this (the values are only known at import),
# which is the accepted cost of not duplicating the vocabulary by hand.
SymptomName = Literal[SYMPTOM_NAMES]  # type: ignore[valid-type]

# Symptom-level severity, distinct from the case-level `Severity` in
# state.py. Defined in vocabulary/ alongside the Arabic words that map onto
# it — closed vocabularies live there, schemas consume them, same as symptom
# names. English values, matching state.py's convention; the Arabic surface
# form is a presentation concern for the report layer, not a storage format.
SymptomOnset = Literal["sudden", "gradual"]


class ExtractedSymptom(BaseModel):
    """One symptom the patient confirmed.

    Every optional field defaults to None and must stay None unless the
    patient actually supplied it — an absent detail is information, and a
    guessed one corrupts both the match score and the doctor report.
    """

    name: SymptomName = Field(description="Canonical symptom name.")
    raw_mention: str | None = Field(
        default=None,
        description=(
            "The patient's own wording for this symptom, verbatim. Free text "
            "belongs here and never in `name`."
        ),
    )
    duration: str | None = Field(
        default=None, description="Patient-reported duration, in their own words."
    )
    severity: SymptomSeverity | None = Field(default=None)
    onset: SymptomOnset | None = Field(default=None)


class NegatedSymptom(BaseModel):
    """A symptom the patient explicitly denied.

    Carries diagnostic weight equal to a confirmed symptom (CLAUDE.md >
    State), so it is constrained to the same vocabulary.
    """

    name: SymptomName = Field(description="Canonical symptom name.")
    raw_mention: str | None = Field(
        default=None, description="The patient's own wording for the denial, verbatim."
    )


def _collapse_duplicates(
    entries: list[ExtractedSymptom] | list[NegatedSymptom], field_name: str
) -> list:
    """Collapse repeated names, keeping first position and filling gaps.

    merge_symptoms downstream keys on `name`, so two entries sharing a name
    within one response would silently become one there, arbitrarily losing
    whichever fields the survivor lacked. Collapsing here makes that
    deterministic and leaves an audit trail.

    Collapsing rather than rejecting is deliberate: a model listing the
    same symptom twice is a benign quirk, and failing the whole extraction
    over it would fail a medical turn for no clinical reason.

    First-wins on conflicting values, unlike state.merge_symptoms, which is
    last-wins. The situations differ: across turns a later value is newer
    information, but within a single response a duplicate is a model error,
    so the conservative choice is not to let it overwrite.
    """
    merged: dict[str, Any] = {}
    order: list[str] = []

    for entry in entries:
        if entry.name not in merged:
            merged[entry.name] = entry
            order.append(entry.name)
            continue

        surviving = merged[entry.name]
        filled = {}
        for attr, value in entry.model_dump().items():
            if value is not None and getattr(surviving, attr) is None:
                setattr(surviving, attr, value)
                filled[attr] = value

        log_malformed_output(
            node="SymptomExtraction",
            reason=f"duplicate_symptom_name:{field_name}",
            payload={"name": entry.name, "merged_fields": filled},
        )

    return [merged[name] for name in order]


class SymptomExtraction(BaseModel):
    """Full extract_symptoms output."""

    symptoms: list[ExtractedSymptom] = Field(default_factory=list)
    negated_symptoms: list[NegatedSymptom] = Field(default_factory=list)
    unmatched_mentions: list[str] = Field(
        default_factory=list,
        description=(
            "Symptom-like phrases the patient used that do NOT match any of "
            "the canonical symptom names available to you — verbatim, in the "
            "patient's own words. Use this ONLY for something you recognize "
            "as a genuine physical or mental complaint the patient is "
            "reporting or denying; never for greetings, small talk, or "
            "anything unrelated to a health complaint. Never force a "
            "near-match canonical name onto a phrase like this — leave "
            "`symptoms`/`negated_symptoms` alone and list the phrase here "
            "instead, unedited."
        ),
    )

    @model_validator(mode="after")
    def _dedupe_by_name(self) -> "SymptomExtraction":
        self.symptoms = _collapse_duplicates(self.symptoms, "symptoms")
        self.negated_symptoms = _collapse_duplicates(self.negated_symptoms, "negated_symptoms")
        return self
