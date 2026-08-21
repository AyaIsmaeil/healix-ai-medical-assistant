"""Request/response contract for POST /health-questions.

Separate from api/contracts.py's ChatRequest/ChatResponse on purpose —
this is a different feature (general health education, not triage) with
a different safety shape. Do not import ChatRequest/ChatResponse here or
reuse this module's models on POST /chat; see docs/AHD_DATA_PROVENANCE.md
and rag/health_education/ for the feature this contract belongs to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "educational", "triage_redirect", "emergency_redirect", "medication_safety", "out_of_scope"
]
RetrievalStatus = Literal["sufficient", "insufficient"]


class HealthQuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str | None = None
    locale: str = "ar"


class SourceReference(BaseModel):
    """A category-level pointer into the AHD corpus, never the raw
    retrieved question/answer text (that would leak raw patient-submitted
    dataset content into the API response — see
    tests/unit/test_health_qa.py's no-raw-leak tests)."""

    dataset: str = "AHD: Arabic Healthcare Dataset"
    category: str
    license: str = "CC BY 4.0"


class HealthQuestionResponse(BaseModel):
    answer: str
    category: Category
    sources: list[SourceReference] = Field(default_factory=list)
    grounded: bool
    disclaimer: str
    retrieval_status: RetrievalStatus
