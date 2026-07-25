"""
Healix - Specialty Recommendation Domain Model (Phase 3.6)
نموذج المجال الخالص لتوصية التخصّص الطبي — العقد الثابت الذي يستهلكه أي
adapter مستقبلي (قاعدي الآن، نموذج ML لاحقاً) عبر ``SpecialtyRecommenderPort``
(انظر ``domain.ports``). تبديل الـadapter لا يغيّر هذا العقد إطلاقاً.

بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على ML حقيقي —
dataclass خالص فقط، بنفس فلسفة ``domain.prediction`` و``domain.urgency``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpecialtyRecommendation:
    """توصية التخصّص الطبي المناسب لاستقبال المريض — تُعاد دائماً من أي adapter."""

    specialty: str
    confidence: float
    explanation: str
