"""
استخراج المعرفة الطبية من ملخصات PubMed عبر LLM (OpenRouter) أو وضع تجريبي.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from app.config import config
from app.exceptions import LLMProviderError
from app.llm.factory import build_llm_provider
from app.rag.extraction_schema import (
    KNOWLEDGE_EXTRACTION_JSON_SCHEMA,
    KNOWLEDGE_JSON_NUDGE,
    ExtractedMedicalKnowledge,
    validate_knowledge_extraction_json,
    validate_knowledge_extraction_shape,
)
from app.rag.pubmed_client import PubMedArticle

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a medical knowledge extraction assistant for a clinical decision support "
    "system (NOT a diagnostic tool). Given a PubMed abstract, extract structured "
    "clinical knowledge in English JSON only.\n\n"
    "Rules:\n"
    "- Extract ONLY what is supported by the abstract; do not invent symptoms or specialties.\n"
    "- associated_symptoms: list of symptom/sign phrases mentioned or strongly implied.\n"
    "- medical_specialty: the most relevant clinical specialty (e.g. Cardiology, General Surgery).\n"
    "- triage_level: Emergency (life-threatening, immediate care), Urgent (same-day care), "
    "or Routine (non-urgent outpatient).\n"
    "- clinical_summary: 1-3 sentences summarizing key clinical points from the abstract.\n"
    "- red_flags: warning signs requiring urgent/emergency attention if mentioned.\n"
    "- If a field cannot be inferred from the abstract, use empty list or a conservative "
    "triage_level of Routine with a brief summary stating limited evidence.\n"
    "Return JSON only."
)


class MedicalKnowledgeExtractor:
    """يستخرج حقولاً منظَّمة من ملخص بحث."""

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm
        self._provider = None
        if use_llm and (config.LLM_PROVIDER or "mock").strip().lower() != "mock":
            self._provider = build_llm_provider(
                response_schema=KNOWLEDGE_EXTRACTION_JSON_SCHEMA,
                response_validator=validate_knowledge_extraction_shape,
                response_format_hint=KNOWLEDGE_JSON_NUDGE,
            )

    def extract(self, article: PubMedArticle) -> ExtractedMedicalKnowledge:
        """يستخرج المعرفة من مقالة واحدة."""
        if self._provider is not None:
            try:
                knowledge = self._extract_with_llm(article)
                mode = "llm"
            except LLMProviderError as exc:
                logger.warning(
                    "فشل LLM لـ PMID %s (%s) — الرجوع للاستخراج التجريبي.",
                    article.pmid,
                    exc,
                )
                knowledge = self._extract_heuristic(article)
                mode = "heuristic_fallback"
        else:
            knowledge = self._extract_heuristic(article)
            mode = "heuristic"

        return knowledge.model_copy(
            update={
                "source_pmid": article.pmid,
                "source_title": article.title,
                "source_journal": article.journal,
                "source_year": article.year,
                "source_query_disease": article.query_disease,
                "extraction_mode": mode,
            }
        )

    def extract_batch(self, articles: List[PubMedArticle]) -> List[ExtractedMedicalKnowledge]:
        results: List[ExtractedMedicalKnowledge] = []
        for index, article in enumerate(articles, start=1):
            logger.info(
                "استخراج [%d/%d] PMID=%s",
                index,
                len(articles),
                article.pmid,
            )
            results.append(self.extract(article))
        return results

    def _extract_with_llm(self, article: PubMedArticle) -> ExtractedMedicalKnowledge:
        user_prompt = self._build_user_prompt(article)
        completion = self._provider.generate(_SYSTEM_PROMPT, user_prompt)
        return validate_knowledge_extraction_json(completion.text)

    @staticmethod
    def _build_user_prompt(article: PubMedArticle) -> str:
        mesh = ", ".join(article.mesh_terms[:15]) if article.mesh_terms else "N/A"
        authors = ", ".join(article.authors[:5]) if article.authors else "N/A"
        return (
            f"PMID: {article.pmid}\n"
            f"Title: {article.title}\n"
            f"Journal: {article.journal or 'N/A'} ({article.year or 'N/A'})\n"
            f"Authors: {authors}\n"
            f"MeSH: {mesh}\n"
            f"Search context disease: {article.query_disease or 'N/A'}\n\n"
            f"Abstract:\n{article.abstract or '(no abstract available)'}\n\n"
            "Extract structured clinical knowledge as JSON."
        )

    def _extract_heuristic(self, article: PubMedArticle) -> ExtractedMedicalKnowledge:
        """
        استخراج تجريبي بلا شبكة — للتطوير والاختبار فقط.
        لا يُستخدم كبديل إنتاجي عن LLM.
        """
        text = f"{article.title}\n{article.abstract}".lower()
        symptoms = _match_symptoms(text)
        specialty = _infer_specialty(text, article.query_disease)
        triage = _infer_triage(text)
        red_flags = _match_red_flags(text)

        disease_name = article.query_disease or article.title.split(":")[0].strip()
        summary = article.abstract[:500] if article.abstract else article.title

        return ExtractedMedicalKnowledge(
            disease_name=disease_name,
            associated_symptoms=symptoms,
            medical_specialty=specialty,
            triage_level=triage,
            clinical_summary=summary,
            red_flags=red_flags,
        )


_SYMPTOM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bchest pain\b", "chest pain"),
    (r"\bdyspnea\b|\bshortness of breath\b", "shortness of breath"),
    (r"\bfever\b|\bpyrexia\b", "fever"),
    (r"\babdominal pain\b", "abdominal pain"),
    (r"\bnausea\b", "nausea"),
    (r"\bvomiting\b", "vomiting"),
    (r"\bcough\b", "cough"),
    (r"\bwheez", "wheezing"),
    (r"\bpolyuria\b|\bfrequent urination\b", "polyuria"),
    (r"\bpolydipsia\b|\bthirst\b", "increased thirst"),
    (r"\bheadache\b", "headache"),
    (r"\bhypertension\b|\belevated blood pressure\b", "elevated blood pressure"),
)

_RED_FLAG_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bshock\b|\bhypotension\b", "shock or hypotension"),
    (r"\bunconscious\b|\bloss of consciousness\b", "loss of consciousness"),
    (r"\bsevere\b.*\bpain\b", "severe pain"),
    (r"\brespiratory failure\b|\bsevere hypox", "respiratory failure"),
    (r"\bST elevation\b|\bSTEMI\b", "ST-elevation myocardial infarction"),
)

_SPECIALTY_HINTS: tuple[tuple[str, str], ...] = (
    ("cardio", "Cardiology"),
    ("myocardial", "Cardiology"),
    ("hypertension", "Cardiology"),
    ("appendic", "General Surgery"),
    ("asthma", "Pulmonology"),
    ("diabet", "Endocrinology"),
    ("pulmon", "Pulmonology"),
    ("gastro", "Gastroenterology"),
)


def _match_patterns(text: str, patterns: tuple[tuple[str, str], ...]) -> List[str]:
    found: Set[str] = set()
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(label)
    return sorted(found)


def _match_symptoms(text: str) -> List[str]:
    return _match_patterns(text, _SYMPTOM_PATTERNS)


def _match_red_flags(text: str) -> List[str]:
    return _match_patterns(text, _RED_FLAG_PATTERNS)


def _infer_specialty(text: str, query_disease: Optional[str]) -> str:
    combined = f"{text} {query_disease or ''}".lower()
    for hint, specialty in _SPECIALTY_HINTS:
        if hint in combined:
            return specialty
    return "General Medicine"


def _infer_triage(text: str) -> str:
    emergency_markers = (
        "life-threatening",
        "mortality",
        "emergency",
        "cardiac arrest",
        "shock",
        "stemi",
        "rupture",
    )
    urgent_markers = ("urgent", "acute", "severe", "hospitalization", "admission")

    lowered = text.lower()
    if any(marker in lowered for marker in emergency_markers):
        return "Emergency"
    if any(marker in lowered for marker in urgent_markers):
        return "Urgent"
    return "Routine"
