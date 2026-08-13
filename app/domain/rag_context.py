"""
Healix — سياق RAG للتقييم الهجين (قواعد + ML + أدبيات PubMed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.domain.explanation import AssessmentExplanation


@dataclass(frozen=True)
class RagSource:
    """مقطع معرفة مسترجَع من ChromaDB (PubMed)."""

    doc_id: str
    pmid: str
    disease_name: str
    medical_specialty: str
    triage_level: str
    snippet: str
    pubmed_url: str
    relevance_score: Optional[float] = None


@dataclass(frozen=True)
class HybridExplainResult:
    """نتيجة التفسير الهجين: شرط عربي + مصادر RAG (إن وُجدت)."""

    explanation: AssessmentExplanation
    rag_enabled: bool
    rag_sources: List[RagSource] = field(default_factory=list)
