"""KnowledgeBaseEntry: the schema for rag/knowledge_base/*.json.

One file per disease (mirrors nodes/ being one file per node). Each file
is loaded and validated against KnowledgeBaseEntry below — malformed
entries fail loudly at load time rather than reaching rag_retrieve/diagnose
with a silently wrong shape.

symptoms is deliberately NOT constrained to vocabulary/symptoms.py's
enum, unlike schemas/symptoms.py's ExtractedSymptom.name. That enum
exists to stop an LLM from free-generating a name at extraction time;
this is reference data going the other direction — the whole point of
rag/coverage.py is to audit which of these strings the vocabulary is
still missing, which requires being able to store one that isn't in it
yet. See CLAUDE.md > Symptom vocabulary and rag/coverage.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"


class KnowledgeBaseEntry(BaseModel):
    name: str = Field(description="Disease name, as given by the source — not translated.")
    name_ar: str = Field(
        min_length=1,
        description=(
            "Patient-facing Arabic name — plain language a patient would "
            "recognize, not a clinical/diagnostic label (same standard as "
            "`symptoms` below, not the same standard as `name` above). "
            "This is what nodes/generate_reports.py's patient register "
            "actually shows; `name` stays reserved for the doctor "
            "register's clinical cross-referencing. Not automatically "
            "derivable from `name` (it is not a translation exercise — "
            "e.g. conjunctivitis's clinically-literal Arabic term is "
            "not what a patient says; see rag/knowledge_base/conjunctivitis.json's "
            "name_ar), so this is authored per entry, same as symptoms."
        ),
    )
    symptoms: list[str] = Field(
        min_length=1,
        description="Reference symptoms, in the exact Arabic wording provided.",
    )
    specialties: list[str] = Field(
        min_length=1, description="Specialty/specialties a matching diagnosis would route to."
    )
    source: str = Field(description="Citation for this entry's clinical content.")
    note: str | None = Field(
        default=None,
        description=(
            "Free-text clinical nuance that doesn't fit a symptom list — an "
            "asymptomatic-presentation pattern, a distinguishing feature "
            "against a look-alike diagnosis, a negative finding, a duration "
            "threshold. Never a substitute for adding/correcting a symptom."
        ),
    )
    translation_reviewed: bool = Field(
        description="Whether this entry's Arabic content has been human-reviewed."
    )
    applicable_sex: Literal["male", "female"] | None = Field(
        default=None,
        description=(
            "Set only when this condition is anatomically restricted to "
            "one sex (e.g. dysmenorrhea, PCOS, vaginal candidiasis) — "
            "None means applicable to either, which is the default and "
            "correct for almost every entry. Do NOT set this for a "
            "condition that is merely more common in one sex (e.g. "
            "urinary tract infection, iron deficiency anaemia) — that is "
            "a prevalence fact, not an anatomical exclusion, and gating "
            "on it would incorrectly exclude a real case of the other "
            "sex. nodes/rag_retrieve.py excludes an entry outright when "
            "state['patient_sex'] contradicts this field, and treats an "
            "otherwise-qualifying match as needing patient confirmation "
            "(a follow-up question, never a guess) when patient_sex is "
            "unknown — see that node's module docstring."
        ),
    )


def load_all(directory: Path = KNOWLEDGE_BASE_DIR) -> list[KnowledgeBaseEntry]:
    """Load and validate every *.json file in `directory`.

    Sorted by filename for a deterministic order — callers (rag.coverage,
    tests) should not depend on filesystem iteration order.
    """
    entries = []
    for path in sorted(directory.glob("*.json")):
        entries.append(KnowledgeBaseEntry.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return entries
