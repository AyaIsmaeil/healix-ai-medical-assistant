"""
Healix - Confidence Assessment Domain Model (Phase 3.7)
نموذج المجال الخالص لتقدير موثوقية التقييم الكامل — العقد الثابت الذي
يستهلكه أي adapter مستقبلي (قاعدي الآن، نموذج مُعايَرة ML لاحقاً) عبر
``ConfidenceEstimatorPort`` (انظر ``domain.ports``). تبديل الـadapter لا
يغيّر هذا العقد إطلاقاً.

هذا المكوّن يُقدّر مدى موثوقية كل ما أُنتج قبله فقط — لا يتنبّأ بمرض، ولا
يغيّر الاستعجال، ولا يغيّر التخصّص. مجرّد حكم على الثقة.

بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على ML حقيقي —
dataclass خالص فقط، بنفس فلسفة ``domain.prediction`` و``domain.urgency``
و``domain.specialty``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceAssessment:
    """تقدير موثوقية التقييم الكامل — يُعاد دائماً من أي adapter.

    ``requires_human_review`` علم صريح لطبقة التطبيق (Laravel) بأنّ الحالة
    تحتاج مراجعة بشرية قبل الاعتماد على مخرجاتها.
    """

    overall_confidence: float
    requires_human_review: bool
    explanation: str
