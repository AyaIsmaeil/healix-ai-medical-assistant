"""
Healix - Assessment Explanation Domain Model (Phase 3.8)
نموذج المجال الخالص للتفسير النهائي للتقييم — العقد الثابت الذي يُنتجه أي
adapter مستقبلي (LLM الآن، أو أي مولّد شرح آخر) عبر ``AssessmentExplainerPort``
(انظر ``domain.ports``). تبديل الـadapter لا يغيّر هذا العقد إطلاقاً.

هذا هو الناتج النهائي لخطّ التقييم: شرح عربي واضح لما حُسِب سابقاً فقط — لا
تشخيص، ولا تعديل لأي تنبؤ/استعجال/تخصّص/ثقة.

بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على مزوّد LLM —
dataclass خالص فقط، بنفس فلسفة ``domain.prediction`` و``domain.urgency``
و``domain.specialty`` و``domain.confidence``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssessmentExplanation:
    """التفسير النهائي للتقييم بالعربية — يُعاد دائماً من أي adapter.

    ``disclaimer`` يجب أن ينصّ دائماً على أنّ هذا التقييم ليس تشخيصاً طبياً
    نهائياً (يُفرَض بالتعليمات وبالبديل الحتمي عند تعذّر الـLLM).
    """

    summary: str
    medical_reasoning: str
    recommendation: str
    disclaimer: str
